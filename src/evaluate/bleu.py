# -*- coding: utf-8 -*-

import torch

from torchtext.data.metrics import bleu_score

# 利用bleu评估预测序列和真实序列的值
def bleu_score(data, model, device=None):
    '''
    传入的model以预测模式
    :param data:
    :param model:
    :param device:
    :return:
    '''
    model.eval()
    trgs = []
    predict_trgs = []
    with torch.no_grad():
        for batch in data:
            src = batch.src
            trg = batch.trg
            output, _ = model(src, trg[:, :-1])  # 这里忽略了trg的<eos>，output=[batch_size, trg_len-1, output_dim]
            predict_trgs.append(output)
            trgs.append([trg])
    score = bleu_score(predict_trgs, trgs)
    return score
