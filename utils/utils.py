import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from typing import Tuple, List


def get_blood_col():
    origin_con_name = ['PDW(fL)', '[IG%(%)]', 'NRBC%(%)', '[HFLC%(%)]', '[NE-SSC(ch)]', '[NE-SFL(ch)]',
                    '[NE-FSC(ch)]', '[LY-X(ch)]', '[LY-Y(ch)]', '[LY-Z(ch)]', '[MO-X(ch)]',
                    '[MO-Y(ch)]', '[MO-Z(ch)]', '[NE-WX]', '[NE-WY]', '[NE-WZ]', '[LY-WX]',
                    '[LY-WY]', '[LY-WZ]', '[MO-WX]', '[MO-WY]', '[MO-WZ]']
    origin_cat_name = ['WBC(10^9/L)', 'RBC(10^12/L)', 'HGB(g/L)', 'PLT(10^9/L)',
                        'MCV(fL)', 'RDW-CV(%)', 'MPV(fL)', 'P-LCR(%)', 'LYMPH#(10^9/L)',
                        'LYMPH%(%)', 'MONO#(10^9/L)', 'MONO%(%)', 'NEUT%(%)', 'EO%(%)',
                        'BASO%(%)', 'Q-Flag(Blasts/Abn Lympho?)', 'Q-Flag(Blasts?)',
                        'Q-Flag(Abn Lympho?)', 'Q-Flag(Atypical Lympho?)']
    columns = origin_cat_name + origin_con_name  
    return columns




def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)  # batch_size
        _, pred = output.topk(maxk, 1, True, True)  # pred: (batch_size, maxk)
        pred = pred.t()  # pred: (maxk, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results



def f1_c(y_true, y_pred, num_cls):
    confusion_matrix = np.zeros((num_cls, num_cls))
    # 填充混淆矩阵
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    precision = np.diag(confusion_matrix) / np.maximum(confusion_matrix.sum(axis=0), 1e-10)  # 防止除以零
    recall = np.diag(confusion_matrix) / np.maximum(confusion_matrix.sum(axis=1), 1e-10)     # 防止除以零

    f1_per_class = np.zeros_like(precision)

    numerator = 2 * (precision * recall)  
    denominator = precision + recall  

    # 对于分母不为零的情况，计算 F1 分数
    f1_per_class[denominator != 0] = numerator[denominator != 0] / denominator[denominator != 0]

    support = confusion_matrix.sum(axis=1)
    
    weighted_f1 = np.sum(support * f1_per_class) / np.sum(support) 

    return weighted_f1



class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


