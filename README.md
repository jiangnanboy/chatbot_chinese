# Chinese chatbot with seq2seq
Chinese chatbot for neural machine translation in PyTorch.

- 利用seq2seq系列的神经网络模型构建中文chatbot。数据来自于[小黄鸡](https://github.com/aceimnorstuvwxz/dgk_lost_conv/tree/master/results).
- 每行数据被处理成字形式，这里没有分词。数据集、字典等的生成使用torchtext处理。
- 利apex进行混合精度训练。


## Model
### 1.seq2seq
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
* Encoder: LSTM
* Decoder: LSTM

### 2.seq2seq_attention
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* Encoder: GRU
* Decoder: GRU

## Use
- seq2seq
```
python seq2seq.py -type train
python seq2seq.py -type predict
```
- seq2seq_attention
```
python seq2seq_attention.py -type train
python seq2seq_attention.py -type predict
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