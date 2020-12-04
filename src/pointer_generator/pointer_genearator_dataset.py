# -*- coding: utf-8 -*-

from torchtext import data,datasets
import torch
from random import shuffle
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler

'''
这个方法为【pointer generator】指针生成模型提供相关数据,
利用torchtext构建词典
'''
def build_field_dataset_vocab(data_directory, src_name, trg_name, vocab):

    tokenize = lambda x: x.split()

    # 定义field，这里source与target共用vocab字典
    source = data.Field(sequential=True, tokenize=tokenize,
                        lower=True, use_vocab=True,
                        init_token='<sos>', eos_token='<eos>',
                        pad_token='<pad>', unk_token='<unk>',
                        batch_first=True, fix_length=50) #include_lengths=True为方便之后使用torch的pack_padded_sequence

    # 定义数据集
    train_data = datasets.TranslationDataset(path=data_directory,
                                             exts=(src_name, trg_name),
                                             fields=(source, source))  # source与target共用vocab可使用同一个Fields
    # 创建词汇表
    if vocab is None:
        source.build_vocab(train_data, min_freq=2)
    else:
        source.vocab = vocab

    return source, train_data

# 对来自经torchtext处理的数据进行再处理成模型所需数据
def get_dataset(source, data, batch_size):
    # 词与索引的映射词典
    word_to_index = source.vocab.stoi
    # 索引与词的映射词典
    index_to_word = source.vocab.itos
    # unk的索引
    index_to_unk = source.vocab.stoi['<unk>']
    # pad的索引
    index_to_pad = source.vocab.stoi['<pad>']
    # sos索引
    index_to_sos = source.vocab.stoi['<sos>']
    # eos索引
    index_to_eos = source.vocab.stoi['<eos>']
    # 每行最大词个数
    max_length = source.fix_length

    # 保存每行样本oov词个数，这个oov词是不在词典中的词，这个oov词是来自src中的
    src_oov_lens = []

    '''
    encoder_input : 不带oov词索引号的encoder输入
    encoder_input_with_oov : 带oov词索引号的encoder输入
    '''
    encoder_input = []
    encoder_input_with_oov = []

    '''
    decoder_input : 不带oov词索引号的decoder输入
    decoder_input_with_oov : 带oov词索引号的decoder输入
    '''
    decoder_input = []
    decoder_input_with_oov = []

    #遍历所有data数据
    for i in range(len(data.examples)):
        src_trg_dict = vars(data.examples[i]) #一行包含src与trg

        oov_word = []  # src中oov单词

        '''
        以下处理src数据
        '''
        src_word_index = [] # 将src中的词映射为数字
        src_word_index.append(index_to_sos) #一行的开头
        src_word_index_with_oov = [] #将src中的词映射为数字，并将oov词也映射成索引
        src_word_index_with_oov.append(index_to_eos) #一行的开头

        for w in src_trg_dict['src']:
            src_word_index.append(word_to_index[w])
            if w not in word_to_index.keys():
                if w not in oov_word:
                    oov_word.append(w)
                src_word_index_with_oov.append(len(word_to_index) + oov_word.index(w)) #这里将src中的oov词也编上索引
            else:
                src_word_index_with_oov.append(word_to_index[w])
        encoder_input.append(add_padding(src_word_index, max_length, index_to_pad, index_to_eos))
        encoder_input_with_oov.append(add_padding(src_word_index_with_oov, max_length, index_to_pad, index_to_eos))

        '''
        以下处理trg数据
        '''
        trg_word_index = []  # 将trg中的词映射为数字
        trg_word_index.append(index_to_sos) #一行的开头
        trg_word_index_with_oov = []  # 将trg中的词映射为数字，并将oov词也映射成索引(oov词典来自src中的oov词典)
        trg_word_index_with_oov.append(index_to_sos) #一行的开头

        for w in src_trg_dict['trg']:
            trg_word_index.append(word_to_index[w])
            if w not in word_to_index.keys():
                if w in oov_word:
                    trg_word_index_with_oov.append(len(word_to_index) + oov_word.index(w))
                else:
                    trg_word_index_with_oov.append(index_to_unk)
            else:
                trg_word_index_with_oov.append(word_to_index[w])
        decoder_input.append(add_padding(trg_word_index, max_length, index_to_pad, index_to_eos))
        decoder_input_with_oov.append(add_padding(trg_word_index_with_oov, max_length, index_to_pad, index_to_eos))

        # 保存每个样本src中的oov词数
        src_oov_lens.append(len(oov_word))

    encoder_input = torch.tensor(encoder_input, dtype=torch.long)
    encoder_input_with_oov = torch.tensor(encoder_input_with_oov, dtype=torch.long)
    decoder_input = torch.tensor(decoder_input, dtype=torch.long)
    decoder_input_with_oov = torch.tensor(decoder_input_with_oov, dtype=torch.long)
    src_oov_lens = torch.tensor(src_oov_lens, dtype=torch.int)

    '''
    TensorDataset可以用来对tensor进行打包，就好像python中的zip功能。该类通过每一个tensor的第一个维度进行索引。因此，该类中的tensor第一维度必须相等。
    '''
    data_set = TensorDataset(encoder_input, encoder_input_with_oov, decoder_input, decoder_input_with_oov, src_oov_lens)

    train_len = int(len(data_set) * 0.95)
    train_set, val_set = torch.utils.data.random_split(data_set, [train_len, len(data_set) - train_len])
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_set, shuffle=True, batch_size=batch_size)
    return train_dataloader, val_dataloader

def add_padding(data, max_length, index_to_pad, index_to_eos):
    '''
    增加padding

    :param data:
    :param max_length: 每行最大单词数
    :param index_to_pad: pad对应的索引号
    :param index_to_eos: 行结束符的索引号
    :return:
    '''
    data = data[:max_length -1] #减1是为了在最后添加eos这个token索引
    padding_len = (max_length - 1) - len(data)
    assert padding_len >= 0, 'padding_len = max_length - len(data)'
    data.extend([index_to_pad] * padding_len)
    data.append(index_to_eos)
    return data

def get_dataset_to_model(batch, hidden_dim, device):
    encoder_input, encoder_input_with_oov, decoder_input, decoder_input_with_oov, src_oov_lens = batch
    batch_size = encoder_input.shape[0]
    # 此batch中包括oov长度最长的那个长度
    max_src_oov_len = src_oov_lens.max().item()
    '''
    这里注意一下oov_zeros和init_coverage两个张量
    oov_zeros：后面在计算loss之前会和vocab词典拼接计算loss
    init_coverage：这是指针网络中使用覆盖机制，缓解生成重复的词。这个覆盖机制是累加各个时间步的attention权重
    '''
    oov_zeros = None
    if max_src_oov_len > 0: #使用指针时，并且在这个batch中存在oov词汇，oov_zeros才不是None
        oov_zeros = torch.zeros((batch_size, max_src_oov_len), dtype=torch.float32)
    init_coverage = torch.zeros(encoder_input.size(), dtype=torch.float32)
    #注意力上下文
    init_context_vec = torch.zeros((batch_size, hidden_dim), dtype=torch.float32)
    data_input = [encoder_input, encoder_input_with_oov, decoder_input, decoder_input_with_oov, oov_zeros, init_coverage, init_context_vec]
    data_input = [dt.to(device) if dt is not None else None for dt in data_input]
    return data_input
