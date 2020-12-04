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

### 2.seq2seq_attention
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* Encoder: GRU (packed padded sequences)
* Decoder: GRU

### 3.seq2seq_attention with pointer generator
* [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
* Encoder: GRU (packed padded sequences)
* Decoder: GRU (pointer network and coverage)

## Use
- parameters setting
```
resource/config.cfg
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