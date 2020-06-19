#!user\bin\python3 transformer_single_extra\ validation
# -*- coding: utf-8 -*-
# @Time  : 2019/12/15 21:58
# @user  : miss
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

class validation():
    def __init__(self,label=None,prediction=None):
        self.label = label
        self.pre = prediction

    def acc(self, label = None, prediction = None):
        if label is not None:
            self.label = label
        if prediction is not None:
            self.pre = prediction
        if self.label is None or self.pre is None:
            raise Exception('label or prediction is None!')
        # 准确率
        acc_value = accuracy_score(y_true=self.label, y_pred=self.pre)
        return acc_value

    def f1(self, label = None, prediction = None):
        if label is not None:
            self.label = label
        if prediction is not None:
            self.pre = prediction
        if self.label is None or self.pre is None:
            raise Exception('label or prediction is None!')
        # 宏平均F1
        f1_value = f1_score(y_true=self.label, y_pred=self.pre, average='macro')
        return f1_value

    def MP(self, label = None, prediction = None):
        # 宏平均准确率
        if label is not None:
            self.label = label
        if prediction is not None:
            self.pre = prediction
        if self.label is None or self.pre is None:
            raise Exception('label or prediction is None!')
        # 宏平均准确率
        MP_value = precision_score(y_true=self.label, y_pred=self.pre, average='macro')
        return MP_value

    def MR(self, label=None, prediction=None):
        # 宏平均召回率
        if label is not None:
            self.label = label
        if prediction is not None:
            self.pre = prediction
        if self.label is None or self.pre is None:
            raise Exception('label or prediction is None!')
        # 宏平均召回率
        MR_value = recall_score(y_true=self.label, y_pred=self.pre, average='macro')
        return MR_value

    def smooth(self,List):
        value = 0
        for val in List:
            value+=val
        value = value/len(List)
        return value


