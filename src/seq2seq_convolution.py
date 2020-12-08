# -*- coding: utf-8 -*-
import os
import torchsnooper
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import math
import time
from apex import amp
from configparser import ConfigParser
import argparse

from util import build_field_dataset_vocab, save_vocab, load_vocab, init_weights, epoch_time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 利用卷积网络实现seq2seq model
'''
encoder将会输出两个张量：一个是卷积张量;一个是联合了卷积张量和embedding(token embedding + position embedding)张量的输出（计算attention）
'''
# 利用残差网络构建编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout, kernel_size, position_length):
        super(Encoder, self).__init__()

        assert kernel_size % 2 == 1, '卷积核大小应为奇数，为了保证序列两端padding时对称，保证锚点在中间。'

        #确保整个网络的变化不会发生太大变化
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        # 词的embedding
        self.token_embedding = nn.Embedding(input_dim, emb_dim)
        # 对词的位置进行embedding
        self.position_embedding = nn.Embedding(position_length, emb_dim)

        # 以下是通过将token与position的embedding按元素加和后经过一层线性变换
        self.embedding_to_hidden = nn.Linear(emb_dim, hidden_dim)
        # 以下是通过将经过残差网络后的张量再经过一层线性变换
        self.hidden_to_embedding = nn.Linear(hidden_dim, emb_dim)


        # 以下是卷积块
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim,
                                              out_channels=2*hidden_dim,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size-1)//2) for _ in range(n_layers)]) # padding=(kernel_size-1)//2 保证padding在序列两边
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src=[batch_size, seq_len]
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # 构建位置tensor -> [batch_size, src_len]，位置序号从(0)开始到(src_len-1)
        position = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        # 对词和其位置进行embedding -> [batch_size,seq_len,embdim]
        token_embeded = self.token_embedding(src)
        position_embeded = self.position_embedding(position.long())

        # 对词和其位置的embedding进行按元素加和 -> [batch_size, seq_len, embdim]
        token_position_embedding = self.dropout(token_embeded + position_embeded)

        # 对以上加和后的tensor经过一层线性变换作为卷积的输入 -> [batch_size, seq_len, hidden_dim]
        linear_conv_input = self.embedding_to_hidden(token_position_embedding)

        #利用permute转换维度 -> [batch_size, hidden_size, seq_len]
        linear_conv_input = linear_conv_input.permute(0, 2, 1)

        #开始进入卷积块
        for i, conv in enumerate(self.convs):
            # 经过一层卷积 -> [batch_size, 2*hidden_dim, src_len]
            conved = conv(self.dropout(linear_conv_input))
            #通过门控线性单元激活，沿dim分成两半A和B，A为经过卷积，B为经过卷积后再经过sigmoid，最后A * B
            conved = F.glu(conved, dim = 1) # [batch_size, hidden_dim, src_len]
            # 利用残差连接,[batch_size, hidden_dim, src_learn]
            conved = (conved + linear_conv_input) * self.scale
            # 以下作为下一层卷积块的输入
            linear_conv_input = conved

        # 转换维度，并经过一个线性层[batch_size, seq_len ,hidden_dim] -> [batch_size, seq_len, embdim]
        conved = self.hidden_to_embedding(conved.permute(0, 2, 1))

        # 将经过残差的conved和输入token_position_embedding按元素加和，用作后面的attention
        combined = (conved + token_position_embedding) * self.scale # [batch_size, seq_len, embdim]
        return conved, combined

# 构建attention权重计算方式
class Attention(nn.Module):
    def __init__(self, hidden_dim, emb_dim):
        super(Attention, self).__init__()

        # 确保整个网络的变化不会发生太大变化
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        # attention经过线性变换
        self.attention_hidden2embed = nn.Linear(hidden_dim, emb_dim)
        self.attention_embed2hidden = nn.Linear(emb_dim, hidden_dim)

    def forward(self, embedded, conved, encoder_conved, encoder_combined):
        '''
        :param embedded: [batch_size, trg_len, emb_dim]
        :param conved: [batch_size, hidden_dim, trg_len]，经过卷积后的张量
        :param encoder_conved: [batch_size, src_len, emb_dim]
        :param encoder_combined: [batch_size, src_len, emb_dim]
        :return:
        '''
        # [batch_size, trg_len, emb_dim]
        conved_emb = self.attention_hidden2embed(conved.permute(0, 2, 1))

        # [batch_size, trg_len, emb_dim]，decoder的卷积后的张量和decoder的embedding = token_embedding +position_embedding做一个残差连接
        combined = (conved_emb + embedded) * self.scale

        # [batch_size, trg_len, emb_dim] * [batch_size, emb_dim, src_len] = [batch_size, trg_len, src_len]
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1)) # matmul = bmm

        # [batch_size, trg_len, src_len]
        attention = F.softmax(energy, dim=-1)

        # [batch_size, trg_len, emb_dim]
        attention_encoding = torch.bmm(attention, encoder_combined)

        # [batch_size, trg_len, hidden_dim]
        attention_encoding = self.attention_embed2hidden(attention_encoding)

        # 做一个残差连接，[batch_size，hidden_dim，trg_len]
        attention_combined = (conved + attention_encoding.permute(0, 2, 1)) * self.scale

        return attention, attention_combined

# 构建解码器
'''
与编码器模型结构有稍微不同：
1.embedding没有被用在卷积块之后的残差连接。而是，embedding被输到卷积块中作为残差连接使用。
2.编码器的conved和组合输出被用在decoder的卷积块内。
3.解码器的输出为一个线性层，输出维度应是词典大小。这是用来预测句子中的下一个单词是什么。
'''
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, kernel_size, trg_pad_idx, position_length):
        super(Decoder, self).__init__()
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx

        # 确保整个网络的变化不会发生太大变化
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        # 词的embedding
        self.token_embedding = nn.Embedding(output_dim, emb_dim)
        # 对词的位置进行embedding
        self.position_embedding = nn.Embedding(position_length, emb_dim)

        # 以下是通过将token与position的embedding按元素加和后经过一层线性变换
        self.embedding_to_hidden = nn.Linear(emb_dim, hidden_dim)
        # 以下是通过将经过残差网络后的张量再经过一层线性变换
        self.hidden_to_embedding = nn.Linear(hidden_dim, emb_dim)

        # 最后的预测输出，词典中词的概率
        self.fc_out = nn.Linear(emb_dim, output_dim)

        # 卷积块
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim,
                                              out_channels=2*hidden_dim,
                                              kernel_size=kernel_size) for _ in range(n_layers)])
        # attention计算
        self.atten = Attention(hidden_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, encoder_conved, encoder_combined):
        '''
        :param trg: [batch_size, trg_len]
        :param encoder_conved: [batch_size, seq_len ,emb_dim]
        :param encoder_combined: [batch_size, seq_len ,emb_dim]
        :return:
        '''
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # 构建位置tensor -> [batch_size, trg_len]，位置序号从(0)开始到(trg_len-1)
        position = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        # 对词和其位置进行embedding -> [batch_size,trg_len,embdim]
        token_embeded = self.token_embedding(trg)
        position_embeded = self.position_embedding(position.long())

        # 对词和其位置的embedding进行按元素加和 -> [batch_size, trg_len, embdim]
        token_position_embedding = self.dropout(token_embeded + position_embeded)

        # 对以上加和后的tensor经过一层线性变换作为卷积的输入 -> [batch_size, trg_len, hidden_dim]
        linear_conv_input = self.embedding_to_hidden(token_position_embedding)

        # 利用permute转换维度 -> [batch_size, hidden_size, trg_len]
        linear_conv_input = linear_conv_input.permute(0, 2, 1)

        batch_size = linear_conv_input.shape[0]
        hidden_dim = linear_conv_input.shape[1]

        # 开始卷积
        for i, conv in enumerate(self.convs):
            linear_conv_input = self.dropout(linear_conv_input)

            # padding
            padding = torch.zeros(batch_size, hidden_dim, self.kernel_size-1).fill_(self.trg_pad_idx).to(device)

            # [batch_size, hidden_dim, trg_len+kernel_size-1],padding放在一侧
            padded_conv_input = torch.cat((padding, linear_conv_input), dim=-1)

            # 进入卷积，[batch_size, 2 * hidden_dim, trg_len]
            conved = conv(padded_conv_input)

            # 通过门控线性单元激活，沿dim分成两半A和B，A为经过卷积，B为经过卷积后再经过sigmoid，最后A * B
            conved = F.glu(conved, dim=1)  # [batch_size, hidden_dim, trg_len]

            # 计算attention=[batch_size, trg_len, src_len]，conved=[batch_size，hidden_dim，trg_len]
            attention, conved = self.atten(token_position_embedding, conved, encoder_conved, encoder_combined)

            # 残差连接,[batch_size，hidden_dim，trg_len]，卷积后经过glu激活后再经过attention后与decoder的输入embedding进行残差连接
            conved = (conved + linear_conv_input) * self.scale

            # 作为下一层卷积块的输入
            linear_conv_input = conved

        # 经过线性层，[batch_size, trg_len, emd_dim]
        conved = self.hidden_to_embedding(conved.permute(0, 2, 1))

        # 预测输出，[batch_size, trg_len, output_dim]
        output = self.fc_out(self.dropout(conved))
        return output, attention

# 利用Encoder与Decoder构建seq2seq模型
class Seq2Seq(nn.Module):
    '''
    接收source句子
    利用编码器encoder中的卷积生成上下文向量attention和attention_combined
    利用解码器decoder中的卷积做预测

    由于解码是并行完成的，所以不需要循环解码。所有的target序列一次性被输入到解码器，并且使用填充来确保解码器中的每个卷积过滤器只能看到序列中当前和之前的token，因为序列在句子中滑动。
    无法使用teacher forcing，因为是并行预测的。
    '''

    def __init__(self, predict_flag, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.predict_flag = predict_flag

    def forward(self, src, trg, max_length=50):
        '''
        :param src: [batch_size, src_len]
        :param trg: [batch_size, trg_len-1] 切除<eos>
        :return:
        '''
        # 预测，一次输入一句话
        if self.predict_flag:
            # 在预测模型，trg初始化是列表:['<sos>']，所以在forward()中trg是['<sos>'的索引号]
            encoder_conved, encoder_combined = self.encoder(src)
            for i in range(max_length):
                # [1,trg_len]
                target_tensor = torch.LongTensor(trg).unsqueeze(0)
                # output = [batch_size, trg_len, output_dim]
                output, _ = self.decoder(target_tensor, encoder_conved, encoder_combined)
                prob_max_index = output.argmax(-1)[:,-1].item() # 每次拿到最后一个预测概率最大的那一个的索引
                trg.append(prob_max_index)
                if prob_max_index == 3: # <eos>=3
                    break
            return trg

        # 训练
        else:
            # encoder将会输出两个张量：一个是卷积张量;一个是联合了卷积张量和embedding(token embedding + position embedding)张量的输出（计算attention）
            # [batch_size, src_len, emb_dim]
            encoder_conved, encoder_combined = self.encoder(src)

            # output=[batch_size, trg_len-1, output_dim], attention=[batch_size, trg_len-1, src_len]
            output, attention = self.decoder(trg, encoder_conved, encoder_combined)
            return output, attention


# 构建模型，优化函数，损失函数，学习率衰减函数
def build_model(source, encoder_embedding_dim, decoder_embedding_dim, hidden_dim, en_conv_layers, de_conv_layers,
                en_kernel_size, de_kernel_size, max_position_length, encoder_dropout,
                decoder_dropout, lr, gamma, weight_decay):

    input_dim = output_dim = len(source.vocab)
    # target <pad>
    target_pad_index = source.vocab.stoi[source.pad_token]
    encoder = Encoder(input_dim, encoder_embedding_dim, hidden_dim, en_conv_layers, encoder_dropout, en_kernel_size, max_position_length)
    decoder = Decoder(output_dim, decoder_embedding_dim, hidden_dim, de_conv_layers, decoder_dropout, de_kernel_size, target_pad_index, max_position_length)

    model = Seq2Seq(False, encoder, decoder).to(device)

    model.apply(init_weights)

    # 定义优化函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 定义lr衰减
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # 定义损失函数,这里忽略<pad>的损失。
    criterion = nn.CrossEntropyLoss(ignore_index=target_pad_index)
    return model, optimizer, scheduler, criterion


# 训练
def train(model, iterator, optimizer, criterion, clip):
    '''
    开始训练：
        1.得到source与target句子
        2.上一批batch的计算梯度归0
        3.给模型喂source与target，并得到输出output
        4.由于损失函数只适用于带有1维target和2维的input，我们需要用view进行flatten(在计算损失时，从output与target中忽略了第一列<sos>)
        5.反向传播计算梯度loss.backward()
        6.梯度裁剪，防止梯度爆炸
        7.更新模型参数
        8.损失值求和(返回所有batch的损失的均值)
    '''
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src  # src=[batch_size, seq_len]
        trg = batch.trg  # trg=[batch_size, trg_len]
        src = src.to(device)
        trg = trg.to(device)

        output, _ = model(src, trg[:,:-1])  # 这里忽略了trg的<eos>，output=[batch_size, trg_len-1, output_dim]
        # 以下在计算损失时，忽略了每个tensor的第一个元素及<sos>
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)  # output=[batch_size * (trg_len - 1), output_dim]，这个预测输出包括<eos>
        trg = trg[:, 1:].reshape(-1)  # 这里忽略了<sos>，trg=[batch_size * (trg_len - 1)]

        loss = criterion(output, trg)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += float(loss.item())
        # print('epoch_loss:{}'.format(float(loss.item())))
    return epoch_loss / len(iterator)

'''
评估
'''
def evaluate(model, iterator, criterion):
    model.eval()  # 评估模型，切断dropout与batchnorm
    epoch_loss = 0
    with torch.no_grad():  # 不更新梯度
        for i, batch in enumerate(iterator):
            src = batch.src  # src=[batch_size, seq_len]
            trg = batch.trg  # trg=[batch_size, seq_len]
            src = src.to(device)
            trg = trg.to(device)

            # output=[batch_size, trg_len-1, output_dim]
            output, _ = model(src, trg[:, :-1])  # 这里忽略了trg的<eos>

            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)  # output=[batch_size * (trg_len - 1), output_dim]
            trg = trg[:, 1:].reshape(-1)  # trg=[batch_size * (trg_len - 1)]
            loss = criterion(output, trg)
            epoch_loss += float(loss.item())
    return epoch_loss / len(iterator)


def train_model(model, train_iterator, val_iterator, optimizer, scheduler, criterion, n_epochs, clip, model_path):
    '''
    开始训练我们的模型：
    1.每一次epoch，都会检查模型是否达到的最佳的validation loss，如果达到了，就更新
    最好的validation loss以及保存模型参数
    2.打印每个epoch的loss以及困惑度。
    '''
    best_valid_loss = float('inf')
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, clip)
        valid_loss = evaluate(model, val_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)
        scheduler.step()
        print('epoch:{},time-mins:{},time-secs:{}'.format(epoch + 1, epoch_mins, epoch_secs))
        print('train loss:{},train perplexity:{}'.format(train_loss, math.exp(train_loss)))
        print('val loss:{}, val perplexity:{}'.format(valid_loss, math.exp(valid_loss)))


# 预测
def predict(sentence, vocab, model):
    with torch.no_grad():
        tokenized = list(sentence)  # tokenize the sentence
        tokenized.insert(0, '<sos>')
        tokenized.append('<eos>')
        indexed = [vocab.stoi[t] for t in tokenized]  # convert to integer sequence
        src_tensor = torch.LongTensor(indexed)  # convert to tensor
        src_tensor = src_tensor.unsqueeze(0)  # reshape in form of batch,no. of words
        trg = [vocab.stoi['<sos>']]
        prediction = model(src_tensor, trg)  # prediction
        return prediction

if __name__ == '__main__':
    parent_directory = os.path.dirname(os.path.abspath("."))
    # 加载配置文件
    config = ConfigParser()
    config.read(os.path.join(parent_directory, 'resource') + '/config.cfg')
    section = config.sections()[0]

    parser = argparse.ArgumentParser(description='seq2seq_attention_chatbot')
    parser.add_argument('-type', default='train', help='train or predict with seq2seq!', type=str)
    args = parser.parse_args()
    if args.type == 'train':
        data_directory = os.path.join(parent_directory, 'data') + '\\'

        #如果词典存在，则加载
        vocab = None
        if os.path.exists(config.get(section, 'vocab')):
            vocab = load_vocab(config.get(section, 'vocab'))

        #加载训练数据
        source, train_iterator, val_iterator = build_field_dataset_vocab(data_directory,
                                                                         config.get(section, 'chat_source_name'),
                                                                         config.get(section, 'chat_target_name'),
                                                                         vocab,
                                                                         field_include_length=False)
        #保存source的词典
        if vocab is None:
            save_vocab(source.vocab, config.get(section, 'vocab'))

        model, optimizer, scheduler, criterion = build_model(source,
                                                          config.getint(section, 'encoder_embedding_dim'),
                                                          config.getint(section, 'decoder_embedding_dim'),
                                                          config.getint(section, 'hidden_dim'),
                                                          config.getint(section, 'en_conv_layers'),
                                                          config.getint(section, 'de_conv_layers'),
                                                          config.getint(section, 'en_kernel_size'),
                                                          config.getint(section, 'de_kernel_size'),
                                                          config.getint(section, 'max_position_length'),
                                                          config.getfloat(section, 'encoder_dropout'),
                                                          config.getfloat(section, 'decoder_dropout'),
                                                          config.getfloat(section, 'lr'),
                                                          config.getfloat(section, 'gamma'),
                                                          config.getfloat(section, 'weight_decay'))

        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        train_model(model,
                    train_iterator,
                    val_iterator,
                    optimizer,
                    scheduler,
                    criterion,
                    config.getint(section, 'n_epochs'),
                    config.getfloat(section, 'clip'),
                    config.get(section, 'seq2seq_convolution_model'))

    elif args.type == 'predict':
        device = torch.device('cpu')
        vocab = load_vocab(config.get(section, 'vocab'))
        input_dim = output_dim = len(vocab)

        target_pad_index = vocab['<pad>']

        encoder = Encoder(input_dim,
                          config.getint(section, 'encoder_embedding_dim'),
                          config.getint(section, 'hidden_dim'),
                          config.getint(section, 'en_conv_layers'),
                          config.getfloat(section, 'encoder_dropout'),
                          config.getint(section, 'en_kernel_size'),
                          config.getint(section, 'max_position_length'))

        decoder = Decoder(output_dim,
                          config.getint(section, 'decoder_embedding_dim'),
                          config.getint(section, 'hidden_dim'),
                          config.getint(section, 'de_conv_layers'),
                          config.getfloat(section, 'decoder_dropout'),
                          config.getint(section, 'de_kernel_size'),
                          target_pad_index,
                          config.getint(section, 'max_position_length'))

        model = Seq2Seq(True, encoder, decoder)

        model.load_state_dict(torch.load(config.get(section, 'seq2seq_convolution_model')))
        model.eval()
        while True:
            sentence = input('you:')
            if sentence == 'exit':
                break
            prediction = predict(sentence, vocab, model)
            prediction = [vocab.itos[t] for t in prediction]
            print('bot:{}'.format(''.join(prediction)))