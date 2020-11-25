# -*- coding: utf-8 -*-
from torchtext import data,datasets
from torchtext.vocab import Vectors
from tqdm import tqdm
import os
import sys
import torchsnooper
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import math
import time
from apex import amp
from configparser import ConfigParser
import argparse

from util import build_field_dataset_vocab, save_vocab, load_vocab, init_weights, count_parameters, epoch_time



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#构建编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        #初始化
        #h0 = torch.zeros(self.n_layers, src.size(1), self.hidden_dim).to(device)
        #c0 = torch.zeros(self.n_layers, src.size(1), self.hidden_dim).to(device)
        #nn.init.kaiming_normal_(h0)
        #nn.init.kaiming_normal_(c0)
        # src=[batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))
        # embedd=[batch_size,seq_len,embdim]
        #output,(hidden,cell) = self.lstm(embedded, (h0,c0))
        output,(hidden,cell) = self.lstm(embedded)
        #output=[batch_size, seq_len, hidden_size*n_directions]
        #hidden=[batch_size, n_layers*n_directions,hidden_size]
        #cell=[batch_size, n_layers*n_directions,hidden_size]
        return hidden, cell

#构建解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        #input=[batch_size]
        #hidden=[batch_size, n_layers*n_directions, hidden_size]
        #cell=[batch_size, n_layers*n_directions, hidden_size]
        input = input.unsqueeze(1)
        #input=[batch_size, 1]
        embedded = self.dropout(self.embedding(input))
        #embedded=[batch_sze, 1, emb_dim]
        output,(hidden,cell) = self.lstm(embedded, (hidden,cell))
        #output=[batch_size, 1, hidden_size*n_directions]
        #hidden=[batch_size, n_layers*n_directions,hidden_size]
        #cell=[batch_size, n_layers*n_directions,hidden_size]
        '''
        seq_len在decoder阶段是1，如果单向则n_directions=1因此：
        output = [batch_size, 1, hidden_size]
        hidden = [batch_size, n_layers, hidden_size]
        cell = [batch_size, n_layers, hidden_size]
        '''
        prediction = self.fc_out(output.squeeze(1))
        #prediction=[batch_size, output_dim]
        return prediction, hidden, cell

#利用Encoder与Decoder构建seq2seq模型
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
    def __init__(self, predict_flag, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.predict_flag = predict_flag
        assert encoder.hidden_dim == decoder.hidden_dim,'encoder与decoder的隐藏状态维度必须相等！'
        assert encoder.n_layers == decoder.n_layers,'encoder与decoder的层数必须相等！'
    
    def forward(self, src, trg, teacher_forcing_ration=0.8, max_len=50):

        #预测，一次输入一句话
        if self.predict_flag:
            assert len(src) == 1, '预测时一次输入一句话'
            output_tokens = []
            hidden, cell = self.encoder(src)
            input = torch.tensor(2).unsqueeze(0)  # 预测阶段输入第一个token-> <sos>
            while True:
                output, hidden, cell = self.decoder(input, hidden, cell)
                input = output.argmax(1)
                output_token = input.squeeze().detach().item()
                if output_token == 3 or len(output_tokens) == max_len:  # 输出最终结果是<eos>或达到最大长度则终止
                    break
                output_tokens.append(output_token)
            return output_tokens

        #训练
        else:
            '''
            src=[batch_size, seq_len]
            trg=[batch_size, seq_len]
            teacher_forcing_ration是使用teacher forcing的概率,例如teacher_forcing_ration=0.8，则输入的时间步有80%的真实值。
            '''
            batch_size = trg.shape[0]
            trg_len = trg.shape[1]
            trg_vocab_size = self.decoder.output_dim
            # 存储decoder outputs
            outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)
            # encoder的最后一层hidden state作为decoder的初始隐状态
            hidden, cell = self.encoder(
                src)  # hidden=[batch_size, n_layers*n_directions,hidden_size]; cell=[batch_size, n_layers*n_directions,hidden_size]
            # 输入到decoder的第一个是<sos>
            input = trg[:, 0]
            for t in range(1, trg_len):
                '''
                解码器输入token的embedding，为之前的hidden与cell状态
                接收输出即predictions和新的hidden与cell状态
                '''
                output, hidden, cell = self.decoder(input, hidden, cell)
                # 存入decoder的预测值
                outputs[:, t, :] = output
                # 是否使用teacher forcing
                teacher_force = random.random() < teacher_forcing_ration
                # 获取预测的最大概率的token
                predict_max = output.argmax(1)
                '''
                如果是teacher forcing则下一步使用真实token作为解码的输入
                否则使用decoder的预测值作为下一步的解码输入
                '''
                input = trg[:, t] if teacher_force else predict_max
            return outputs


#构建模型，优化函数，损失函数，学习率衰减函数
def build_model(source, encoder_embedding_dim, decoder_embedding_dim, hidden_dim, n_layers, encoder_dropout, decoder_dropout, lr, gamma, weight_decay):
    '''
    训练seq2seq model
    input与output的维度是字典的大小。
    encoder与decoder的embedding与dropout可以不同
    网络的层数与hiden/cell状态的size必须相同
    '''
    input_dim = output_dim = len(source.vocab)

    encoder = Encoder(input_dim, encoder_embedding_dim, hidden_dim, n_layers, encoder_dropout)
    decoder = Decoder(output_dim, decoder_embedding_dim, hidden_dim, n_layers, decoder_dropout)

    model = Seq2Seq(False, encoder, decoder).to(device)

    model.apply(init_weights)
    
    #定义优化函数 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #定义lr衰减
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    #这里忽略<pad>的损失。
    target_pad_index = source.vocab.stoi[source.pad_token]
    #定义损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=target_pad_index)
    return model,optimizer,scheduler,criterion

#训练
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
        src = batch.src # src=[batch_size, seq_len]
        trg = batch.trg # trg=[batch_size, seq_len]
        src = src.to(device)
        trg = trg.to(device)
        output = model(src, trg, 1) # output=[batch_size, seq_len, output_dim]
        #以下在计算损失时，忽略了每个tensor的第一个元素及<sos>
        output_dim = output.shape[-1]
        output = output[:,1:,:].reshape(-1, output_dim) # output=[batch_size * (seq_len - 1), output_dim]
        trg = trg[:,1:].reshape(-1) # trg=[batch_size * (seq_len - 1)]
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
    model.eval() #评估模型，切断dropout与batchnorm
    epoch_loss = 0
    with torch.no_grad():#不更新梯度
        for i, batch in enumerate(iterator):
            src = batch.src # src=[batch_size, seq_len]
            trg = batch.trg # trg=[batch_size, seq_len]
            src = src.to(device)
            trg = trg.to(device)
            # output=[batch_size, seq_len, output_dim]
            output = model(src, trg, 0) #评估的时候不使用teacher force，使用预测作为每一步的输入
            
            output_dim = output.shape[-1]
            output = output[:,1:,:].reshape(-1, output_dim) # output=[batch_size * (seq_len - 1), output_dim]
            trg = trg[:,1:].reshape(-1) # trg=[batch_size * (seq_len - 1)]
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
        print('epoch:{},time-mins:{},time-secs:{}'.format(epoch + 1,epoch_mins, epoch_secs))
        print('train loss:{},train perplexity:{}'.format(train_loss, math.exp(train_loss)))
        print('val loss:{}, val perplexity:{}'.format(valid_loss, math.exp(valid_loss)))
 
#预测
def predict(sentence, vocab, model):
    with torch.no_grad():
        tokenized = list(sentence) #tokenize the sentence
        tokenized.append('<eos>')
        indexed = [vocab.stoi[t] for t in tokenized]          #convert to integer sequence
        tensor = torch.LongTensor(indexed)              #convert to tensor
        tensor = tensor.unsqueeze(0)                           #reshape in form of batch,no. of words
        prediction = model(tensor, None)                  #prediction
        return prediction

if __name__ == '__main__':
    parent_directory = os.path.dirname(os.path.abspath("."))
    # 加载配置文件
    config = ConfigParser()
    config.read(os.path.join(parent_directory, 'resource') + '/config.cfg')
    section = config.sections()[0]

    parser = argparse.ArgumentParser(description='seq2seq_chatbot')
    parser.add_argument('-type', default='train', help='train or predict with seq2seq!', type=str)
    args = parser.parse_args()
    if args.type == 'train':
        data_directory = os.path.join(parent_directory, 'data') + '\\'

        #加载训练数据
        source, train_iterator, val_iterator = build_field_dataset_vocab(data_directory,
                                                                         config.get(section, 'chat_source_name'),
                                                                         config.get(section, 'chat_target_name'))
        #保存source的词典
        save_vocab(source.vocab, config.get(section, 'vocab'))

        model,optimizer,scheduler,criterion = build_model(source,
                                                          config.getint(section, 'encoder_embedding_dim'),
                                                          config.getint(section, 'decoder_embedding_dim'),
                                                          config.getint(section, 'hidden_dim'),
                                                          config.getint(section, 'n_layers'),
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
                    config.get(section, 'seq2seq_model'))

    elif args.type == 'predict':
        vocab = load_vocab(config.get(section, 'vocab'))
        input_dim = output_dim = len(vocab)
        encoder = Encoder(input_dim, config.getint(section, 'encoder_embedding_dim'), config.getint(section, 'hidden_dim'), config.getint(section, 'n_layers'), config.getfloat(section, 'encoder_dropout'))
        decoder = Decoder(output_dim, config.getint(section, 'decoder_embedding_dim'), config.getint(section, 'hidden_dim'), config.getint(section, 'n_layers'), config.getfloat(section, 'decoder_dropout'))
        model = Seq2Seq(True, encoder, decoder)
        model.load_state_dict(torch.load(config.get(section, 'seq2seq_model')))
        model.eval()
        while True:
            sentence = input('you:')
            if sentence == 'exit':
                break
            prediction = predict(sentence, vocab, model)
            prediction = [vocab.itos[t] for t in prediction]
            print('bot:{}'.format(''.join(prediction)))



