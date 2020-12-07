# -*- coding: utf-8 -*-
from torchtext import data,datasets
from torchtext.vocab import Vectors
from tqdm import tqdm
import os
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
import numpy as np

from util import save_vocab, load_vocab, init_weights, epoch_time

from pointer_generator.pointer_genearator_dataset import get_dataset, build_field_dataset_vocab, get_dataset_to_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# https://zhuanlan.zhihu.com/p/265319703
# 构建编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout, pad_index):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.pad_index = pad_index

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_index)
        self.gru = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # 初始化
        # h0 = torch.zeros(self.n_layers, src.size(1), self.hidden_dim).to(device)
        # c0 = torch.zeros(self.n_layers, src.size(1), self.hidden_dim).to(device)
        # nn.init.kaiming_normal_(h0)
        # nn.init.kaiming_normal_(c0)
        # src=[batch_size, seq_len]

        #以下三行为计算batch中src的长度，不包括pad(之前pad的索引映射为1，这里是去除pad后的词数)
        exist = (src != 1) * 1.0
        factor = np.ones(src.shape[1])
        src_len = np.dot(exist.cpu(), factor) #这里从cuda转为cpu
        src_len = torch.from_numpy(src_len)
        src_len = src_len.long()

        embedded = self.dropout(self.embedding(src))
        # embedd=[batch_size,seq_len,embdim]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_len, batch_first=True, enforce_sorted=False) #enforce_sorted=False会自动根据每个行词数进行降序排序
        output, hidden = self.gru(packed)
        # output=[batch_size, seq_len, hidden_size*n_directions]
        # hidden=[n_layers*n_directions, batch_size, hidden_size]
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=self.pad_index, total_length=len(src[0])) #这个会返回output以及压缩后的legnths
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        # et(energy函数)的计算经过4层线性变换
        self.w_h = nn.Linear(hidden_dim, hidden_dim) #对encoder_output进行线性变换
        self.w_s = nn.Linear(hidden_dim, hidden_dim) #对deocder_output进行线性变换
        self.w_c = nn.Linear(1, hidden_dim) #对之前所有注意力权重之和coverage进行线性变换
        self.v = nn.Linear(hidden_dim, 1) #这是最后一层线性变换(在这之前又经过tahn变换)

    # encoder_output ：来自encoder的output => [batch_size, seq_len, n_directions * hidden_size]
    # decoder_hidden : 来自decoder的hidden => [batch_size, n_directions*hidden_dim]
    # coverage : sum of attention score (batch,seq_len) 之前每一步attention权重之和(第一步初始为全0) => [batch_size, seq_len]
    def forward(self, encoder_output, decoder_hidden, coverage):
        # 对encoder_output进行线性变换
        encoder_feature = self.w_h(encoder_output)   # (batch_size, seq_len, hidden_dim)

        # 对deocder_output进行线性变换
        decoder_feature = self.w_s(decoder_hidden).unsqueeze(1) # (batch_size, 1, hidden_dim)

        # broadcast 广播运算
        attention_feature = encoder_feature + decoder_feature  # (batch,seq_len,hidden)

        #对之前所有注意力权重之和coverage进行线性变换
        coverage_feature = self.w_c(coverage.unsqueeze(2))  # [batch_size, seq_len, 1]  - > (batch, seq_len,hidden)

        #energy的计算中加入coverage机制，缓解生成重复词问题
        attention_feature += coverage_feature

        #energy的计算
        e_t = self.v(torch.tanh(attention_feature)).squeeze(dim = 2)  # [batch_size, seq_len, hidden] -> (batch_size, seq_len, 1) -> (batch_size, seq_len)

        # 经过softmax归一化后的attention权重 -> (batch_size, seq_len)
        a_t = torch.softmax(e_t,dim=-1)

        #将所有attention权重加和，这是为了实现coverage机制，缓解生成重复词问题
        sum_coverage = coverage + a_t # 用来对之前decoder 每一步的attention distribution 进行求和

        return a_t, sum_coverage  # attention权重以及所有attention权重之和 a_t=[batch_size, seq_len]，next_coverage=[batch_size, seq_len]

# pgen是公式中计算genearation的概率
class GeneratorProbability(nn.Module):
    def __init__(self,hidden_dim,embed_dim):
        super(GeneratorProbability,self).__init__()

        self.w_h = nn.Linear(hidden_dim, 1)
        self.w_s = nn.Linear(hidden_dim, 1)
        self.w_x = nn.Linear(embed_dim, 1)

    # context : (batch, hidden)，注意力上下文
    # hidden : decoder hidden，(batch,hidden)，decoder的hidden状态
    # input : decoder input，(batch,embed)，decoder的输入
    def forward(self,context, hidden, input):
        h_feature = self.w_h(context)     # (batch,1)
        s_feature = self.w_s(hidden)     # (batch,1)
        x_feature = self.w_x(input)     # (batch,1)

        gen_feature = h_feature + s_feature + x_feature  # (batch,1)

        gen_p = torch.sigmoid(gen_feature)

        return gen_p

# 构建解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, pad_index):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_index)

        #这里将上一步的注意力上下文和本步的embedding进行拼接，搞一个线性层
        self.get_gru_input = nn.Linear(hidden_dim + emb_dim, emb_dim)

        self.gru = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        self.attention = Attention(hidden_dim)

        #pgen是公式中计算genearation的概率
        self.pgen = GeneratorProbability(hidden_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

        #以下是Pvocab计算词表中词的分布，按公式是先拼接注意力上下文和decoder的hidden或者ouput，再经过2层线性变换，得到词表中词的分布概率
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), #将decoder_output与context拼接，拼接后的维度是hidden_dim * 2
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    '''
    intput: (batch,1) 真实的target，带unk
    hidden: 初始化为encoder最后一层输出的hidden ->  [n_layers*n_directions, batch_size, hidden_size]。后面都是decoder的hidden
    encoder_output: encoder的output, [batch_size, seq_len, n_directions * hidden_size]
    context_vec: => [batch_size, hidden_size], 一开始初始化全0 ,时间步step的记忆，含上一步的语义
    oovs_zero: =>[batch_size, 这个batch中oov词数最大的那行]
    encoder_with_oov: => [batch_size, encoder_seq_len] 带上oov索引
    coverage: 注意力权重之和，一开始初始为全0 => [batch_size, encoder_seq_len]
    '''
    def forward(self, input, hidden, encoder_output, context_vec, oovs_zero, encoder_with_oov, coverage):

        # embedded=[batch_sze, emb_dim]
        embedded = self.dropout(self.embedding(input))

        # 注意力权重以及之前步所有注意力权重之和，这里利用decoder的hidden最后一层计算attention
        # 因为层数n_layers=2，所以拿到hidden最后一层hidden[-1, :, :]
        # attention_weight = [batch_size, seq_len], sum_coverage = [batch_size, seq_len]
        attention_weight, sum_coverage = self.attention(encoder_output, hidden[-1, :, :], coverage)

        # gru的输入，将embedding和上一步的context进行拼接，做一个线性变换， [batch_size, emb_dim] -> [batch_size, 1, emb_dim]
        gru_input = self.get_gru_input(torch.cat([context_vec, embedded], dim=-1)).unsqueeze(1)

        # decoder_output=[batch_size, 1, hidden_size*n_directions]
        # hidden=[n_layers*n_directions, batch_size, hidden_size]
        decoder_output, hidden = self.gru(gru_input, hidden)

        '''
        以下context是计算上下文：根据注意力权重得分和编码器的输出进行矩阵乘法得到
        注意力权重分布用于产生编码器隐藏状态的加权和，加权平均的过程。得到的向量称为上下文向量
        '''
        # [batch_size, 1, seq_len] * [batch_size, seq_len, n_directions*hidden_size] = [batch_size, 1, n_directions*hidden_size]
        context = torch.bmm(attention_weight.unsqueeze(1), encoder_output)

        # 拼接注意力上下文context与decoder_output的hidden_dim =[batch_size, 1, 2 * hidden_dim]
        decoder_output_context = torch.cat([decoder_output, context], 2)

        context = context.squeeze() # [batch_size, n_directions*hidden_size]
        #计算生成概率Pgen -> [batch_size, 1]
        Pgen = self.pgen(context, hidden[-1, :, :], gru_input.squeeze())

        # 计算Pvocab的词典生成概率 -> (batch,vob_size)
        pvocab = self.fc_out(decoder_output_context.squeeze(1))
        pvocab = torch.softmax(pvocab, dim = -1)
        pvocab = pvocab * Pgen

        # 词典外oov词的概率，attention_weight是注意力权重 -> [batch_size, seq_len]
        poovcab= attention_weight * (1 - Pgen)

        if oovs_zero is not None:
            # [batch_size, vob_size + 这个batch中oov词数最大的那行]
            pvocab = torch.cat([pvocab, oovs_zero], dim = -1)
            #最终的预测概率包括词典中的词分布与input时的oov词分布，【src中的值根据index放入pvocab中(dim=1 or -1)列索引，按行scatter操作】
            final_word_p = pvocab.scatter_add(dim=-1, index=encoder_with_oov, src=poovcab)
        else:
            final_word_p = pvocab

        '''
        final_word_p = [batch_size, vob_size + 这个batch中oov词数最大的那行] 最终的预测概率包括词典中的词分布与input时的oov词分布
        hidden = [n_layers*n_directions, batch_size, hidden_size] 经过lstm的hidden与cell状态
        context = [batch_size, n_directions*hidden_size] 这一步的上下文
        attention_weight = [batch_size, seq_len] 注意力权重
        sum_coverage = [batch_size, seq_len] 记录所有attention权重之和
        '''
        return final_word_p, hidden, context, attention_weight, sum_coverage

# 利用Encoder与Decoder构建seq2seq模型
class Seq2Seq(nn.Module):
    '''
    接收source句子
    利用编码器encoder生成上下文向量
    利用解码器decoder生成预测target句子

    每次迭代中：
    传入input以及先前的hidden与cell状态给解码器decoder
    从解码器decoder中接收一个prediction以及下一个hidden与下一个cell状态
    保存这个prediction作为预测句子中的一部分
    决定是否使用"teacher force":
        如果使用：解码器的下一次input是真实的token
        如果不使用：解码器的下一次input是预测prediction（使用output tensor的argmax）的token
    '''

    def __init__(self, predict_flag, encoder, decoder, eps, coverage_loss_lambda):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.eps = eps
        self.coverage_loss_lambda = coverage_loss_lambda

        self.predict_flag = predict_flag # train or predict? true->predict; false->train
        assert encoder.hidden_dim == decoder.hidden_dim, 'encoder与decoder的隐藏状态维度必须相等！'
        assert encoder.n_layers == decoder.n_layers, 'encoder与decoder的层数必须相等！'

    '''
     src = encoder_input => [batch_size, encoder_seq_len]
     src_with_oov = encoder_with_oov => [batch_size, encoder_seq_len]
     trg = decoder_input => [batch_size, decode_seq_len] #带unk的target
     trg_with_oov = decoder_target => [batch_size, decode_seq_len]  # 经过处理带索引的oov，all_decoder_target_with_oov
     oov_zero => [batch_size, 这个batch中oov词数最大的那行]，主要为了decoder阶段扩展词典和oov词典做词生成概率分布
     context_vec => [batch_size, hidden_size] 保存上一步注意力上下文
     coverage 初始化全0 => [batch_size, encoder_seq_len] 此张量保存所有步的attention权重
     max_len 生成的最大词数
    '''
    def forward(self, src, src_with_oov, trg, trg_with_oov, oov_zeros, context_vec, coverage, max_len=50):

        # 预测，一次输入一句话
        if self.predict_flag:
            assert len(src) == 1, '预测时一次输入一句话'

            # encoder的最后一层hidden state作为decoder的初始隐状态
            encoder_output, hidden = self.encoder(src)  # hidden=[n_layers*n_directions, batch_size, hidden_size]


            output_tokens = []

            input = torch.tensor(2).unsqueeze(0)  # 预测阶段解码器输入第一个token-> <sos>
            while True:
                final_word_p, hidden, context_vec, attention_weight, sum_coverage = self.decoder(input, hidden, encoder_output, context_vec, oov_zeros, src_with_oov, coverage)
                context_vec = context_vec.unsqueeze(0)
                input = final_word_p.argmax(1)
                output_token = input.squeeze().detach().item()
                if output_token == 3 or len(output_tokens) == max_len:  # 输出最终结果是<eos>或达到最大长度则终止
                    break
                output_tokens.append(output_token)
            return output_tokens

        # 训练
        else:
            '''
            src=[batch_size, seq_len]
            trg=[batch_size, seq_len]
            '''
            trg_len = trg.shape[1]


            # 每行样本的长度，去除开始token
            decoder_lens = trg.sum(dim=-1) - 1
            # encoder的最后一层hidden state作为decoder的初始隐状态
            encoder_output, hidden = self.encoder(src)  # hidden=[n_layers*n_directions, batch_size, hidden_size]
            all_step_loss = []
            for t in range(1, trg_len):
                # 排除<sos>
                target = trg[:, t]  # [batch_size, 1]
                '''
                解码器输入token的embedding，为之前的hidden与cell状态
                接收输出即predictions和新的hidden与cell状态
                '''
                final_word_p, hidden, context_vec, attention_weight, sum_coverage = self.decoder(target, hidden, encoder_output, context_vec, oov_zeros, src_with_oov, coverage)

                # target_oov = [batch_size, 1], 带oov索引的目标target
                target_oov = trg_with_oov[:, t].unsqueeze(1)


                # 获得这个词分布下，目标词的概率,利用index来索引input特定位置的数值,dim=-1按行操作
                # [batch_size, 1] -> [batch_size]
                probs = torch.gather(input=final_word_p, dim=-1, index=target_oov).squeeze()

                # 计算概率 -log(P)
                step_loss = -torch.log(probs + self.eps)

                # coverage loss，以下将attention_weight与coverage从float32转为float16，不转会出错。
                coverage_loss = self.coverage_loss_lambda * torch.sum(torch.min(attention_weight.to(torch.float16), coverage.to(torch.float16)), dim = -1)

                #最终的损失
                step_loss += coverage_loss

                coverage = sum_coverage

                all_step_loss.append(step_loss)

            token_loss = torch.stack(all_step_loss, dim = 1)

            batch_loss_sum_token = token_loss.sum(dim=-1)
            batch_loss_mean_token = batch_loss_sum_token / decoder_lens.float()
            result_loss = batch_loss_mean_token.mean()

            return result_loss



# 构建模型，优化函数，损失函数，学习率衰减函数
def build_model(source, encoder_embedding_dim, decoder_embedding_dim, hidden_dim, n_layers, encoder_dropout,
                decoder_dropout, lr, gamma, weight_decay, eps, coverage_loss_lambda):
    '''
    训练seq2seq model
    input与output的维度是字典的大小。
    encoder与decoder的embedding与dropout可以不同
    网络的层数与hiden/cell状态的size必须相同
    '''
    input_dim = output_dim = len(source.vocab)

    encoder = Encoder(input_dim, encoder_embedding_dim, hidden_dim, n_layers, encoder_dropout, source.vocab.stoi[source.pad_token])
    decoder = Decoder(output_dim, decoder_embedding_dim, hidden_dim, n_layers, decoder_dropout, source.vocab.stoi[source.pad_token])

    model = Seq2Seq(False, encoder, decoder, eps, coverage_loss_lambda).to(device)

    model.apply(init_weights)

    # 定义优化函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 定义lr衰减
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    return model, optimizer, scheduler


# 训练
def train(model, iterator, hidden_dim, optimizer, clip):
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
        encoder_input, encoder_input_with_oov, decoder_input, decoder_input_with_oov, oov_zeros, init_coverage, init_context_vec = get_dataset_to_model(batch, hidden_dim, device)
        loss = model(encoder_input, encoder_input_with_oov, decoder_input, decoder_input_with_oov, oov_zeros, init_context_vec, init_coverage)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += float(loss.item())

    return epoch_loss

'''
评估
'''
def evaluate(model, iterator, hidden_dim):
    model.eval()  # 评估模型，切断dropout与batchnorm
    epoch_loss = 0
    with torch.no_grad():  # 不更新梯度
        for i, batch in enumerate(iterator):
            encoder_input, encoder_input_with_oov, decoder_input, decoder_input_with_oov, oov_zeros, init_coverage, init_context_vec = get_dataset_to_model(
                batch, hidden_dim, device)
            loss = model(encoder_input, encoder_input_with_oov, decoder_input, decoder_input_with_oov, oov_zeros,
                         init_context_vec, init_coverage)
            epoch_loss += float(loss.item())
    return epoch_loss / len(iterator)


def train_model(model, train_iterator, val_iterator, hidden_dim, optimizer, scheduler, n_epochs, clip, model_path):
    '''
    开始训练我们的模型：
    1.每一次epoch，都会检查模型是否达到的最佳的validation loss，如果达到了，就更新
    最好的validation loss以及保存模型参数
    2.打印每个epoch的loss以及困惑度。
    '''
    best_valid_loss = float('inf')
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = train(model, train_iterator, hidden_dim, optimizer, clip)
        valid_loss = evaluate(model, val_iterator, hidden_dim)
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
def predict(sentence, vocab, model, hidden_dim):
    with torch.no_grad():
        word_to_index = vocab.stoi
        tokenized = list(sentence)  # tokenize the sentence
        tokenized.append('<eos>')
        input_indexed = []
        input_indexed_with_oov = []
        oovs = []

        for token in tokenized:
            input_indexed.append(word_to_index[token])
            if token not in word_to_index.keys():
                if token not in oovs:
                    oovs.append(token)
                input_indexed_with_oov.append(len(word_to_index) + oovs.index(token))  # 这里将src中的oov词也编上索引
            else:
                input_indexed_with_oov.append(word_to_index[token])

        oov_zeros = None
        if len(oovs) > 0:  # 使用指针时，并且在这个input中存在oov词汇，oov_zeros才不是None
            oov_zeros = torch.zeros((1, len(oovs)), dtype=torch.float32)
        init_coverage = torch.zeros((1, len(input_indexed)), dtype=torch.float32)
        # 注意力上下文
        init_context_vec = torch.zeros(hidden_dim, dtype=torch.float32)
        init_context_vec = init_context_vec.unsqueeze(0)
        encoder_input = torch.LongTensor(input_indexed)  # convert to tensor
        encoder_input = encoder_input.unsqueeze(0)  # reshape in form of batch,no. of words

        encoder_input_with_oov = torch.LongTensor(input_indexed_with_oov)
        encoder_input_with_oov = encoder_input_with_oov.unsqueeze(0)

        prediction = model(src=encoder_input, src_with_oov=encoder_input_with_oov, trg=None, trg_with_oov=None, oov_zeros=oov_zeros, context_vec=init_context_vec, coverage=init_coverage)  # prediction

        index_to_word = vocab.itos
        # 将oov词放到index_to_word词典中
        for word in oovs:
            index_to_word[len(index_to_word) + oovs.index(word)] = word

        prediction = [index_to_word[t] for t in prediction]
        return prediction

'''
if __name__ == '__main__':
    parent_directory = os.path.dirname(os.path.abspath(".."))

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
        source, train_data = build_field_dataset_vocab(data_directory,
                                                      config.get(section, 'chat_source_name'),
                                                      config.get(section, 'chat_target_name'),
                                                      vocab)

        train_iterator, val_iterator = get_dataset(source, train_data, config.get(section, 'batch_size'))

        #保存source的词典
        if vocab is None:
            save_vocab(source.vocab, config.get(section, 'vocab'))

        model,optimizer,scheduler = build_model(source,
                                                          config.getint(section, 'encoder_embedding_dim'),
                                                          config.getint(section, 'decoder_embedding_dim'),
                                                          config.getint(section, 'hidden_dim'),
                                                          config.getint(section, 'n_layers'),
                                                          config.getfloat(section, 'encoder_dropout'),
                                                          config.getfloat(section, 'decoder_dropout'),
                                                          config.getfloat(section, 'lr'),
                                                          config.getfloat(section, 'gamma'),
                                                          config.getfloat(section, 'weight_decay'),
                                                          config.getfloat(section, 'eps'),
                                                          config.getfloat(section, 'coverage_loss_lambda'))

        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        train_model(model,
                    train_iterator,
                    val_iterator,
                    config.getint(section, 'hidden_dim'),
                    optimizer,
                    scheduler,
                    config.getint(section, 'n_epochs'),
                    config.getfloat(section, 'clip'),
                    config.get(section, 'seq2seq_pointer_attention_model'))

    elif args.type == 'predict':
        vocab = load_vocab(config.get(section, 'vocab'))
        input_dim = output_dim = len(vocab)
        encoder = Encoder(input_dim, config.getint(section, 'encoder_embedding_dim'), config.getint(section, 'hidden_dim'), config.getint(section, 'n_layers'), config.getfloat(section, 'encoder_dropout'))
        decoder = Decoder(output_dim, config.getint(section, 'decoder_embedding_dim'), config.getint(section, 'hidden_dim'), config.getint(section, 'n_layers'), config.getfloat(section, 'decoder_dropout'))
        model = Seq2Seq(True, encoder, decoder)
        model.load_state_dict(torch.load(config.get(section, 'seq2seq_pointer_attention_model')))
        model.eval()
        while True:
            sentence = input('you:')
            if sentence == 'exit':
                break
            prediction = predict(sentence, vocab, model)
            print('bot:{}'.format(''.join(prediction)))
            
'''

# 测试
if __name__ == '__main__':
    parent_directory = os.path.dirname(os.path.abspath(".."))

    chat_source = 'E:\project\pycharm workspace\chatbot_chinese\data\chat_source.src'
    chat_target = 'E:\project\pycharm workspace\chatbot_chinese\data\chat_target.trg'

    chat_source_name = 'chat_source.src'
    chat_target_name = 'chat_target.trg'

    vocab_file = 'E:\project\pycharm workspace\chatbot_chinese\\vocab\\vocab.pk'

    seq2seq_pointer_attention_model = 'E:\project\pycharm workspace\chatbot_chinese\model\seq2seq - pointer - attention - model.pt'

    batch_size = 256
    max_length = 50
    encoder_embedding_dim = 128
    decoder_embedding_dim = 128
    hidden_dim = 256
    n_layers = 2
    encoder_dropout = 0.5
    decoder_dropout = 0.5
    lr = 0.01
    weight_decay = 0.01
    gamma = 0.1
    n_epochs = 10
    clip = 1
    teacher_forcing_ration = 1
    coverage_loss_lambda = 1.0
    eps = 0.0000001

    type = 'predict'

    if type == 'train':
        data_directory = os.path.join(parent_directory, 'data') + '\\'

        # 如果词典存在，则加载
        vocab = None
        if os.path.exists(vocab_file):
            vocab = load_vocab(vocab_file)

        # 加载训练数据
        source, train_data = build_field_dataset_vocab(data_directory,
                                                       chat_source_name,
                                                       chat_target_name,
                                                       vocab)

        train_iterator, val_iterator = get_dataset(source, train_data, batch_size)

        # 保存source的词典
        if vocab is None:
            save_vocab(source.vocab, vocab_file)

        model, optimizer, scheduler = build_model(source,
                                                  encoder_embedding_dim,
                                                  decoder_embedding_dim,
                                                  hidden_dim,
                                                  n_layers,
                                                  encoder_dropout,
                                                  decoder_dropout,
                                                  lr,
                                                  gamma,
                                                  weight_decay,
                                                  eps,
                                                  coverage_loss_lambda)

        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        train_model(model,
                    train_iterator,
                    val_iterator,
                    hidden_dim,
                    optimizer,
                    scheduler,
                    n_epochs,
                    clip,
                    seq2seq_pointer_attention_model)

    elif type == 'predict':
        vocab = load_vocab(vocab_file)
        input_dim = output_dim = len(vocab)
        encoder = Encoder(input_dim, encoder_embedding_dim, hidden_dim, n_layers, encoder_dropout,
                          vocab['<pad>'])
        decoder = Decoder(output_dim, decoder_embedding_dim, hidden_dim, n_layers, decoder_dropout,
                          vocab['<pad>'])
        model = Seq2Seq(True, encoder, decoder, eps, coverage_loss_lambda)
        model.load_state_dict(torch.load(seq2seq_pointer_attention_model))
        model.eval()
        while True:
            sentence = input('you:')
            if sentence == 'exit':
                break
            prediction = predict(sentence, vocab, model, hidden_dim)
            print('bot:{}'.format(''.join(prediction)))