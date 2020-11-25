from torchtext import data,datasets
import os
import random
from configparser import ConfigParser
import pickle
import torch.nn as nn

'''
1.定义Field：声明如何处理数据，主要包含以下数据预处理的配置信息，比如指定分词方法，是否转成小写，起始字符，结束字符，补全字符以及词典等等
2.定义Dataset：用于得到数据集，继承自pytorch的Dataset。此时数据集里每一个样本是一个 经过 Field声明的预处理 预处理后的 wordlist
3.建立vocab：在这一步建立词汇表，词向量(word embeddings)
4.构造迭代器Iterator：: 主要是数据输出的模型的迭代器。构造迭代器，支持batch定制用来分批次训练模型。
'''

def build_field_dataset_vocab(data_directory, src_name, trg_name, vocab):

    tokenize = lambda x: x.split()

    # 定义field，这里source与target共用vocab字典
    source = data.Field(sequential=True, tokenize=tokenize,
                        lower=True, use_vocab=True,
                        init_token='<sos>', eos_token='<eos>',
                        pad_token='<pad>', unk_token='<unk>',
                        batch_first=True, fix_length=50,
                        include_lengths=True) #include_lengths=True为方便之后使用torch的pack_padded_sequence

    # 定义数据集
    train_data = datasets.TranslationDataset(path=data_directory,
                                             exts=(src_name, trg_name),
                                             fields=(source, source))  # source与target共用vocab可使用同一个Fields
    # 创建词汇表
    if vocab == None:
        source.build_vocab(train_data, min_freq=2)
    else:
        source.vocab = vocab

    # 划分训练与验证集，一个问题，利用random_split进行数据集划分后，会丢失fields属性
    train_set, val_set = train_data.split(split_ratio=0.95, random_state=random.seed(1))

    BATCH_SIZE = 256
    # 生成训练与验证集的迭代器
    train_iterator, val_iterator = data.BucketIterator.splits(
        (train_set, val_set),
        batch_size=BATCH_SIZE,
        # shuffle=True,
        # device=device,
        sort_within_batch=True, #为true则一个batch内的数据会按sort_key规则降序排序
        sort_key=lambda x: len(x.src)
        # repeat=False
    )
    return source, train_iterator, val_iterator

#计算model中可训练的参数个数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#对所有模块和子模块进行权重初始化
def init_weights(model):
    for name,param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

#保存词典
def save_vocab(vocab, path):
    with open(path, 'wb') as f_write:
        pickle.dump(vocab, f_write)

#加载词典
def load_vocab(path):
    with open(path, 'rb') as f_read:
        vocab = pickle.load(f_read)
    return vocab

#每个epoch所花时间
def epoch_time(start_time, end_time):
    run_tim = end_time - start_time
    run_mins = int(run_tim / 60)
    run_secs = int(run_tim-(run_mins * 60))
    return run_mins,run_secs

if __name__ == '__main__':
    parent_directory = os.path.dirname(os.path.abspath("."))
    data_directory = os.path.join(parent_directory, 'data') + '\\'

    config = ConfigParser()
    config.read(os.path.join(parent_directory, 'resource')+'/config.cfg')
    section = config.sections()[0]
    source, train_iterator, val_iterator = build_field_dataset_vocab(data_directory, config.get(section, 'chat_source_name'), config.get(section, 'chat_target_name'))
    print(len(source.vocab))
    print(len(train_iterator))
    print(len(val_iterator))

