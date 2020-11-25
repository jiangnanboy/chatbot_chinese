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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import math
import time
from apex import amp
from configparser import ConfigParser
import argparse

from util import build_field_dataset_vocab, save_vocab, load_vocab, init_weights, count_parameters, epoch_time



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 构建编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # 初始化
        # h0 = torch.zeros(self.n_layers, src.size(1), self.hidden_dim).to(device)
        # c0 = torch.zeros(self.n_layers, src.size(1), self.hidden_dim).to(device)
        # nn.init.kaiming_normal_(h0)
        # nn.init.kaiming_normal_(c0)
        # src=[batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))
        # embedd=[batch_size,seq_len,embdim]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_len, batch_first=True)
        output, hidden = self.gru(packed)
        # output=[batch_size, seq_len, hidden_size*n_directions]
        # hidden=[batch_size, n_layers*n_directions,hidden_size]
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True) #这个会返回output以及压缩后的legnths
        return output, hidden

class Attention(nn.Module):
    def __init__(self, method, hidden_dim):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, '请选择合适的流程意计算方法')
        self.hidden_dim = hidden_dim
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_dim, hidden_dim)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
            self.v = nn.Parameter(torch.FloatTensor(hidden_dim))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2) #[batch_size, seq_len]

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output) #[batch_size, seq_len, hidden_dim]
        return torch.sum(hidden * energy, dim=2) #[batch_size, seq_len]

    def concat_score(self, hidden, encoder_output):
        #hidden.expand(-1, encoder_output.size(1), -1) -> [batch_size, seq_len, n]
        energy = self.attn(torch.cat((hidden.expand(-1, encoder_output.size(1), -1), encoder_output), 2)).tanh()
        #energy=[batch_size, seq_len, hidden_dim]
        return torch.sum(self.v * energy, dim=2) #[batch_size, seq_len]

    def forward(self, hidden, encoder_output):
        #hidden=[batch_size, 1, n_directions*hidden_dim]
        #encoder_output=[batch_size, seq_len, hidden_dim*n_ndirections]
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_output)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_output)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_output)
        return F.softmax(attn_energies, dim=1).unsqueeze(1) #softmax归一化，[batch_size, 1, seq_len]

# 构建解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input=[batch_size]
        # hidden=[batch_size, n_layers*n_directions, hidden_size]
        # cell=[batch_size, n_layers*n_directions, hidden_size]
        input = input.unsqueeze(1)
        # input=[batch_size, 1]
        embedded = self.dropout(self.embedding(input))
        # embedded=[batch_sze, 1, emb_dim]
        output, hidden = self.gru(embedded, hidden)
        # output=[batch_size, 1, hidden_size*n_directions]
        # hidden=[batch_size, n_layers*n_directions,hidden_size]
        # cell=[batch_size, n_layers*n_directions,hidden_size]
        '''
        seq_len在decoder阶段是1，如果单向则n_directions=1因此：
        output = [batch_size, 1, hidden_size]
        hidden = [batch_size, n_layers, hidden_size]
        '''
        prediction = self.fc_out(output.squeeze(1))
        # prediction=[batch_size, output_dim]
        return prediction, hidden


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
                                                                         vocab)
        #保存source的词典
        if vocab == None:
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