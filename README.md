# Chinese chatbot with seq2seq
Chinese chatbot for neural machine translation in PyTorch.

- 利用seq2seq系列的神经网络模型构建中文chatbot。数据来自于[小黄鸡](https://github.com/aceimnorstuvwxz/dgk_lost_conv/tree/master/results).
- 每行数据被处理成字形式，这里没有分词。数据集、字典等的生成使用torchtext处理。
- 利用apex进行混合精度训练。


## Model
### 1.seq2seq
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
* Encoder: LSTM
* Decoder: LSTM
* ![image](https://raw.githubusercontent.com/jiangnanboy/chatbot_chinese/master/img/seq2seq1.png)
* ![image](https://raw.githubusercontent.com/jiangnanboy/chatbot_chinese/master/img/seq2seq2.png)

### 2.seq2seq_attention
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* Encoder: GRU (packed padded sequences)
* Decoder: GRU
* ![image](https://raw.githubusercontent.com/jiangnanboy/chatbot_chinese/master/img/seq2seq_attention1.png)
* ![image](https://raw.githubusercontent.com/jiangnanboy/chatbot_chinese/master/img/seq2seq_attention2.png)

### 3.seq2seq_attention with pointer generator
* [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
* Encoder: GRU (packed padded sequences)
* Decoder: GRU (pointer network and coverage)
* ![image](https://raw.githubusercontent.com/jiangnanboy/chatbot_chinese/master/img/seq2seq_pointernet1.png)

### 4.seq2seq with Convolutional Neural Network
* [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)
* Encoder：cnn
* Encoder：cnn
* ![image](https://raw.githubusercontent.com/jiangnanboy/chatbot_chinese/master/img/seq2seq_convolution1.png)
* ![image](https://raw.githubusercontent.com/jiangnanboy/chatbot_chinese/master/img/seq2seq_convolution2.png)

### 5.seq2seq with transformer
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* Encoder：multi-head-self-attention -> feedforward
* Encoder：multi-head-self-attention -> multi-head-encoder-attention -> feedforward
* ![image](https://raw.githubusercontent.com/jiangnanboy/chatbot_chinese/master/img/seq2seq_transformer.png)

## Use
- parameters setting
```
resource/config.cfg
```
- train data
```
data/chat_source.src
data/chat_source.trg
```
- model save path
```
model/
```
- vocabulary dictionary
```
vocab/vocab.pk
```
- [seq2seq](https://github.com/jiangnanboy/chatbot_chinese/blob/master/src/seq2seq.py)
```
python seq2seq.py -type train
python seq2seq.py -type predict
```
- [seq2seq_attention](https://github.com/jiangnanboy/chatbot_chinese/blob/master/src/seq2seq_attention.py)
```
python seq2seq_attention.py -type train
python seq2seq_attention.py -type predict
```
- [seq2seq_attention_with_pointer_network](https://github.com/jiangnanboy/chatbot_chinese/blob/master/src/pointer_generator/seq2seq_pointernet.py)
```
python seq2seq_pointernet.py -type train
python seq2seq_pointernet.py -type predict
```
- [seq2seq_cnn](https://github.com/jiangnanboy/chatbot_chinese/blob/master/src/seq2seq_convolution.py)
```
python seq2seq_convolution.py -type train
python seq2seq_convolution.py -type predict
```
- [seq2seq_transformer](https://github.com/jiangnanboy/chatbot_chinese/blob/master/src/seq2seq_transformer.py)
```
python seq2seq_transformer.py -type train
python seq2seq_transformer.py -type predict
```
## Note

使用Apex导致的问题：
```
Loss整体变大，而且很不稳定。效果变差。会遇到梯度溢出。
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 32768.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 16384.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 8192.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 4096.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 2048.0
...
ZeroDivisionError: float division by zero

解决办法如下来防止出现梯度溢出：

1、apex中amp.initialize(model, optimizer, opt_level='O0')的opt_level由O2换成O1，再不行换成O0(欧零)
2、把batchsize从32调整为16会显著解决这个问题，另外在换成O0(欧0)的时候会出现内存不足的情况，减小batchsize也是有帮助的
3、减少学习率
4、增加Relu会有效保存梯度，防止梯度消失
```

## Requirements

* GPU & CUDA
* Python3.6.5
* PyTorch1.5
* torchtext0.6
* apex0.1

## References

Based on the following implementations

* http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
* https://github.com/bentrevett
* https://gitee.com/dogecheng/python/blob/master/pytorch/Seq2SeqForTranslation.ipynb
* https://github.com/atulkum/pointer_summarizer/blob/master/training_ptr_gen/train.py
* https://github.com/abisee/pointer-generator