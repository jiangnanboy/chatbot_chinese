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

# 利用transformer实现seq2seq model
'''
encoder产生序列上下文向量
    1.输入是序列中token的embedding与位置embedding
    2.token的embedding与其位置embedding相加，得到一个vector(这个向量融合了token与position信息)
    3.在2之前，token的embedding乘上一个scale(防止点积变大，造成梯度过小)向量[sqrt(emb_dim)]，这个假设为了减少embedding中的变化，没有这个scale，很难稳定的去训练model。
    4.加入dropout
    5.通过N个encoder layer，得到Z。此输出Z被用于decoder中。
    src_mask对于非<pad>值为1,<pad>为0。为了计算attention而遮挡<pad>这个无意义的token。与source 句子shape一致。
'''
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, n_layers, n_heads, pf_dim, dropout, position_length):
        super(Encoder, self).__init__()

        self.scale = torch.sqrt(torch.FloatTensor([emb_dim])).to(device)

        # 词的embedding
        self.token_embedding = nn.Embedding(input_dim, emb_dim)
        # 对词的位置进行embedding
        self.position_embedding = nn.Embedding(position_length, emb_dim)
        # encoder层，有几个encoder层，每个encoder有几个head
        self.layers = nn.ModuleList([EncoderLayer(emb_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src=[batch_size, seq_len]
        # src_mask=[batch_size, 1, 1, seq_len]
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # 构建位置tensor -> [batch_size, seq_len]，位置序号从(0)开始到(src_len-1)
        position = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        # 对词和其位置进行embedding -> [batch_size,seq_len,embdim]
        token_embeded = self.token_embedding(src) * self.scale
        position_embeded = self.position_embedding(position)

        # 对词和其位置的embedding进行按元素加和 -> [batch_size, seq_len, embdim]
        src = self.dropout(token_embeded + position_embeded)

        for layer in self.layers:
            src = layer(src, src_mask)

        # [batch_size, seq_len, emb_dim]
        return src

'''
encoder layers：
    1.将src与src_mask传入多头attention层(multi-head attention)
    2.dropout
    3.使用残差连接后传入layer-norm层(输入+输出后送入norm)后得到的输出
    4.输出通过前馈网络feedforward层
    5.dropout
    6.一个残差连接后传入layer-norm层后得到的输出喂给下一层
    注意：
        layer之间不共享参数
        多头注意力层用到的是多个自注意力层self-attention
'''
class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, n_heads, pf_dim, dropout):
        super(EncoderLayer, self).__init__()
        # 注意力层后的layernorm
        self.self_attn_layer_norm = nn.LayerNorm(emb_dim)
        # 前馈网络层后的layernorm
        self.ff_layer_norm = nn.LayerNorm(emb_dim)
        # 多头注意力层
        self.self_attention = MultiHeadAttentionLayer(emb_dim, n_heads, dropout)
        # 前馈层
        self.feedforward = FeedforwardLayer(emb_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        #src=[batch_size, seq_len, emb_dim]
        #src_mask=[batch_size, 1, 1, seq_len]

        # self-attention
        # _src=[batch size, query_len, emb_dim]
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, 残差连接以及layer-norm
        # src=[batch_size, seq_len, emb_dim]
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # 前馈网络
        # _src=[batch_size, seq_len, emb_dim]
        _src = self.feedforward(src)

        # dropout, 残差连接以及layer-norm
        # src=[batch_size, seq_len, emb_dim]
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src
'''
多头注意力层的计算:
    1.q,k,v的计算是通过线性层fc_q,fc_k,fc_v
    2.对query,key,value的emb_dim split成n_heads
    3.通过计算Q*K/scale计算energy
    4.利用mask遮掩不需要关注的token
    5.利用softmax与dropout
    6.5的结果与V矩阵相乘
    7.最后通过一个前馈fc_o输出结果
注意:Q,K,V的长度一致
'''
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, emb_dim, n_heads, dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        assert emb_dim % n_heads == 0
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim//n_heads

        self.fc_q = nn.Linear(emb_dim, emb_dim)
        self.fc_k = nn.Linear(emb_dim, emb_dim)
        self.fc_v = nn.Linear(emb_dim, emb_dim)

        self.fc_o = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # query=[batch_size, query_len, emb_dim]
        # key=[batch_size, key_len, emb_dim]
        # value=[batch_size, value_len, emb_dim]
        batch_size = query.shape[0]

        # Q=[batch_size, query_len, emb_dim]
        # K=[batch_size, key_len, emb_dim]
        # V=[batch_size, value_len, emb_dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        '''
        view与reshape的异同：
        
        torch的view()与reshape()方法都可以用来重塑tensor的shape，区别就是使用的条件不一样。view()方法只适用于满足连续性条件的tensor，并且该操作不会开辟新的内存空间，
        只是产生了对原存储空间的一个新别称和引用，返回值是视图。而reshape()方法的返回值既可以是视图，也可以是副本，当满足连续性条件时返回view，
        否则返回副本[ 此时等价于先调用contiguous()方法在使用view() ]。因此当不确能否使用view时，可以使用reshape。如果只是想简单地重塑一个tensor的shape，
        那么就是用reshape，但是如果需要考虑内存的开销而且要确保重塑后的tensor与之前的tensor共享存储空间，那就使用view()。
        '''

        # Q=[batch_size, n_heads, query_len, head_dim]
        # K=[batch_size, n_heads, key_len, head_dim]
        # V=[batch_size, n_heads, value_len, head_dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # 注意力打分矩阵 [batch_size, n_heads, query_len, head_dim] * [batch_size, n_heads, head_dim, key_len] = [batch_size, n_heads, query_len, key_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        # [batch_size, n_heads, query_len, key_len]
        attention = torch.softmax(energy , dim = -1)

        # [batch_size, n_heads, query_len, key_len]*[batch_size, n_heads, value_len, head_dim]=[batch_size, n_heads, query_len, head_dim]
        x = torch.matmul(self.dropout(attention), V)

        # [batch_size, query_len, n_heads, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()

        # [batch_size, query_len, emb_dim]
        x = x.view(batch_size, -1, self.emb_dim)

        # [batch_size, query_len, emb_dim]
        x = self.fc_o(x)

        return x, attention

'''
前馈层
'''
class FeedforwardLayer(nn.Module):
    def __init__(self, emb_dim, pf_dim, dropout):
        super(FeedforwardLayer, self).__init__()
        self.fc_1 = nn.Linear(emb_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x=[batch_size, seq_len, emb_dim]

        # x=[batch_size, seq_len, pf_dim]
        x = self.dropout(torch.relu(self.fc_1(x)))

        # x=[batch_size, seq_len, emb_dim]
        x = self.fc_2(x)

        return x


# 构建解码器
'''
目标是将编码器encoder对source sentence的编码表示Z，转为对target sentence的预测token。
然后比较预测token与真实token，计算loss。

和编码器encoder类似，不同的是解码器有两个多头注意力层。一个是target sentence的masked multi-head attention层；一个是decoder表示瓣query以及encoder表示的key与value的multi-head attention层;

'''
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, n_layers, n_heads, pf_dim, dropout, position_length):
        super(Decoder, self).__init__()

        # 确保整个网络的变化不会发生太大变化
        self.scale = torch.sqrt(torch.FloatTensor([emb_dim])).to(device)

        # 词的embedding
        self.token_embedding = nn.Embedding(output_dim, emb_dim)
        # 对词的位置进行embedding
        self.position_embedding = nn.Embedding(position_length, emb_dim)

        self.layers = nn.ModuleList([DecoderLayer(emb_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])

        # 最后的预测输出，词典中词的概率
        self.fc_out = nn.Linear(emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, encoder_src, trg_mask, src_mask):
        '''
        trg: [batch_size, trg_len]
        encoder_src: [batch_size, src_len ,emb_dim]
        trg_mask: [batch_size, 1, trg_len ,trg_len]
        src_mask: [batch_size, 1, 1, src_len]
        '''
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # 构建位置tensor -> [batch_size, trg_len]，位置序号从(0)开始到(trg_len-1)
        position = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        # 对词和其位置进行embedding -> [batch_size,trg_len,embdim]
        token_embeded = self.token_embedding(trg) * self.scale
        position_embeded = self.position_embedding(position)

        # 对词和其位置的embedding进行按元素加和 -> [batch_size, trg_len, embdim]
        trg = self.dropout(token_embeded + position_embeded)

        for layer in self.layers:
            # trg: [batch_size, trg_len, emb_dim]
            # attention: [batch_size, n_heads, trg_len, src_len]
            trg, attention = layer(trg, encoder_src, trg_mask, src_mask)

        # output: [batch_size, trg_len, output_dim]
        output = self.fc_out(trg)

        return output, attention

'''
解码器层没有引入任何新的概念，只是以略微不同的方式使用与编码器相同的一组层。
decoder层与encoder层类似：
    decoder层有两个multi-head attention layers：self-attention与encoder-attention：
    1.self-attention与encoder中的self-attention一样：利用decoder表示query,key与value -> dropout -> 残差连接 -> layer-norm
    2.encoder-attention：decoder表示作为query，encoder的表示作为key与value -> dropout -> 残差连接 -> layer-norm
    3.送入前馈层 -> dropout -> 残差连接 -> layer-norm
'''
class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, n_heads, pf_dim, dropout):
        super(DecoderLayer, self).__init__()

        # 三个子模块的layer-norm层
        # self-attention的layer-norm层
        self.self_attn_layer_norm = nn.LayerNorm(emb_dim)
        # encoder-attention的layer-norm层
        self.enc_attn_layer_norm = nn.LayerNorm(emb_dim)
        # 前馈后的layer-norm层
        self.ff_layer_norm = nn.LayerNorm(emb_dim)

        # self-attention的多头注意力层
        self.self_attention = MultiHeadAttentionLayer(emb_dim, n_heads, dropout)

        # encoder-attention的多头注意力层
        self.encoder_attention = MultiHeadAttentionLayer(emb_dim, n_heads, dropout)

        # 前馈层
        self.feedforward = FeedforwardLayer(emb_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, encoder_src, trg_mask, src_mask):
        '''
        trg: [batch_size, trg_len, emb_dim]
        encoder_src: [batch_size, src_len, emb_dim]
        trg_mask: [batch_size, 1, trg_len, trg_len]
        src_mask: [batch_size, 1, 1, src_len]
        '''
        # self-attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout -> 残差连接 -> layer-norm
        # trg=[batch_size, trg_len, emb_dim]
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # encoder-attention
        _trg, attention = self.encoder_attention(trg, encoder_src, encoder_src, src_mask)

        # dropout -> 残差连接 -> layer-norm
        # trg=[batch_size, trg_len, emb_dim]
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # 前馈层
        _trg = self.feedforward(trg)

        # dropout -> 残差连接 -> layer-norm
        # trg=[batch_size, trg_len, emb_dim]
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention


# 利用Encoder与Decoder构建seq2seq模型
class Seq2Seq(nn.Module):

    def __init__(self, predict_flag, encoder, decoder, src_pad_idx, trg_pad_idx):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.predict_flag = predict_flag

    def mask_src_mask(self, src):
        # src=[batch_size, src_len]

        # src_mask=[batch_size, 1, 1, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def mask_trg_mask(self, trg):
        # trg=[batch_size, trg_len]

        #trg_pad_mask=[batch_size, 1, 1, trg_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[1]

        '''
        对角矩阵，如：
        [[ True, False, False, False],
        [ True,  True, False, False],
        [ True,  True,  True, False],
        [ True,  True,  True,  True]]
        '''
        # trg_sub_mask=[trg_len, trg_len]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()

        '''
        利用逻辑与结合trg_sub_mask与trg_pad_mask，确保后续的token与padding token不参与
        '''
        # trg_mask=[batch_size, 1, trg_len, trg_len]
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, src, trg, max_length=50):
        '''
        :param src: [batch_size, src_len]
        :param trg: [batch_size, trg_len]
        :return:
        '''
        # 预测，一次输入一句话
        if self.predict_flag:
            # 在预测模型，trg初始化是列表:['<sos>']，所以在forward()中trg是['<sos>'的索引号]
            src_mask = model.mask_src_mask(src)
            encoder_src = self.encoder(src, src_mask)

            for i in range(max_length):
                # [1,trg_len]
                target_tensor = torch.LongTensor(trg).unsqueeze(0)
                trg_mask = model.mask_trg_mask(target_tensor)

                # output = [batch_size, trg_len, output_dim]
                output, _ = self.decoder(target_tensor, encoder_src, trg_mask, src_mask)
                prob_max_index = output.argmax(-1)[:,-1].item() # 每次拿到最后一个预测概率最大的那一个的索引
                trg.append(prob_max_index)
                if prob_max_index == 3: # <eos>=3
                    break
            return trg

        # 训练
        else:
            # src_mask = [batch_size, 1, 1, src_len]
            # trg_mask = [batch_size, 1, trg_len, trg_len]
            src_mask = self.mask_src_mask(src)
            trg_mask = self.mask_trg_mask(trg)

            # encoder_src=[batch_size, src_len, emb_dim]
            encoder_src = self.encoder(src, src_mask)

            # output=[batch_size, trg_len, output_dim]
            # attention=[batch_size, n_heads, trg_len, src_len]
            output, attention = self.decoder(trg, encoder_src, trg_mask, src_mask)

            return output, attention


# 构建模型，优化函数，损失函数，学习率衰减函数
def build_model(source, embedding_dim, encoder_layers, decoder_layers, encoder_heads, decoder_heads, encoder_pf_dim, decoder_pf_dim,
                max_position_length, encoder_dropout,
                decoder_dropout, lr, gamma, weight_decay):

    input_dim = output_dim = len(source.vocab)
    # <pad>
    pad_index = source.vocab.stoi[source.pad_token]

    encoder = Encoder(input_dim, embedding_dim, encoder_layers, encoder_heads, encoder_pf_dim, encoder_dropout, max_position_length)
    decoder = Decoder(output_dim, embedding_dim, decoder_layers, decoder_heads, decoder_pf_dim, decoder_dropout, max_position_length)

    model = Seq2Seq(False, encoder, decoder, pad_index, pad_index).to(device)

    #初始化权重
    model.apply(init_weights)

    # 定义优化函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 定义lr衰减
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # 定义损失函数,这里忽略<pad>的损失。
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

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

        scheduler.step()

        train_loss = train(model, train_iterator, optimizer, criterion, clip)
        valid_loss = evaluate(model, val_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)

        print('epoch:{},time-mins:{},time-secs:{}'.format(epoch + 1, epoch_mins, epoch_secs))
        print('train loss:{},train perplexity:{}'.format(train_loss, math.exp(train_loss)))
        print('val loss:{}, val perplexity:{}'.format(valid_loss, math.exp(valid_loss)))


'''
预测：
    1.分词
    2.加入<sos>与<eos>
    3.数字化
    4.传为tensor并增加batch这个维度
    5.创建source mask
    6.将source sentence与source mask喂入encoder
    7.创建一个output list，初始化为<sos>
    8.未达到最大长度或是<eos> token:
        a.将目前输出的预测句子转为带batch维的tensor
        b.创建target mask
        c.将目录的输出，encoder output 以及两个mask传入decoder
        d.得到下一个预测输出token
        e.输出的预测token加入预测句子序列
    9.将预测输出的序列转为token
'''
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
                                                          config.getint(section, 'embedding_dim'),
                                                          config.getint(section, 'encoder_layers'),
                                                          config.getint(section, 'decoder_layers'),
                                                          config.getint(section, 'encoder_heads'),
                                                          config.getint(section, 'decoder_heads'),
                                                          config.getint(section, 'encoder_pf_dim'),
                                                          config.getint(section, 'decoder_pf_dim'),
                                                          config.getint(section, 'max_position_length'),
                                                          config.getfloat(section, 'encoder_dropout'),
                                                          config.getfloat(section, 'decoder_dropout'),
                                                          config.getfloat(section, 'lr'),
                                                          config.getfloat(section, 'gamma'),
                                                          config.getfloat(section, 'weight_decay'))

        # 注意这里的opt_level设为其它值会出现错误：“RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublassGemm”，没看到好的解释
        model, optimizer = amp.initialize(model, optimizer, opt_level='O0')

        train_model(model,
                    train_iterator,
                    val_iterator,
                    optimizer,
                    scheduler,
                    criterion,
                    config.getint(section, 'n_epochs'),
                    config.getfloat(section, 'clip'),
                    config.get(section, 'seq2seq_transformer_model'))

    elif args.type == 'predict':
        device = torch.device('cpu')
        vocab = load_vocab(config.get(section, 'vocab'))
        input_dim = output_dim = len(vocab)

        pad_index = vocab['<pad>']

# input_dim, emb_dim, n_layers, n_heads, pf_dim, dropout, position_length
        encoder = Encoder(input_dim,
                          config.getint(section, 'embedding_dim'),
                          config.getint(section, 'encoder_layers'),
                          config.getint(section, 'encoder_heads'),
                          config.getint(section, 'encoder_pf_dim'),
                          config.getfloat(section, 'encoder_dropout'),
                          config.getint(section, 'max_position_length'))

# output_dim, emb_dim, n_layers, n_heads, pf_dim, dropout, position_length
        decoder = Decoder(output_dim,
                          config.getint(section, 'embedding_dim'),
                          config.getint(section, 'decoder_layers'),
                          config.getint(section, 'decoder_heads'),
                          config.getint(section, 'decoder_pf_dim'),
                          config.getfloat(section, 'decoder_dropout'),
                          config.getint(section, 'max_position_length'))

        model = Seq2Seq(True, encoder, decoder, pad_index, pad_index)

        model.load_state_dict(torch.load(config.get(section, 'seq2seq_transformer_model')))
        model.eval()
        while True:
            sentence = input('you:')
            if sentence == 'exit':
                break
            prediction = predict(sentence, vocab, model)
            prediction = [vocab.itos[t] for t in prediction]
            print('bot:{}'.format(''.join(prediction)))