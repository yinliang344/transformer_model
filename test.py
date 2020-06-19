#!user\bin\python3 fypj_project\ test
# -*- coding: utf-8 -*-
# @Time  : 2019/11/27 18:06
# @user  : miss

import re
from tqdm import tqdm

# patten1 = r'(?s)本院认为.*?([减从]轻处罚|从[重宽]处罚)'
# patten2 = r'(?s)本院认为.*?(数额特别巨大|数额[巨较]大)'
# string = "本院认为，，数额巨大，情节严重，从宽处罚，回家为将回归六"
# pattern1 = re.compile(patten1)
# pattern2 = re.compile(patten2)
# xxx = pattern1.findall(string)
# xxxx = pattern2.findall(string)
# print(xxx)
# print(xxxx)

def xq_transfor(xq):
    xq_c = 0
    if xq > 0 and xq <= 6:#六个月
        xq_c = 1
    elif xq > 6 and xq <= 7:#一个月
        xq_c = 2
    elif xq > 7 and xq <= 9:#两个月
        xq_c = 3
    elif xq > 9 and xq <= 12:#三个月
        xq_c = 4
    elif xq > 12 and xq <= 18:#六个月
        xq_c = 5
    elif xq > 18 and xq <= 24:#六个月
        xq_c = 6
    elif xq > 24 and xq <= 42:#18个月
        xq_c = 7
    elif xq > 42 and xq <= 60:
        xq_c = 8
    elif xq > 60:
        xq_c = 9
    return xq_c
with open('./data/xq_data.txt', 'r',encoding='utf-8') as file,open('./data/train_index.txt','r',encoding='utf-8') as file_data:
    print(len(file_data.readlines()))
    data_lines = file.readlines()
    data = []
    for line in tqdm(data_lines):
        x = True
        temp = list(map(int,line.split()))
        temp1 = xq_transfor(temp[0])
        temp[0] = temp1
        for i in range(len(data)):
            if temp1 == data[i][0]:
                data[i][1]+=temp[1]
                x=False
        if x==True:
            data.append(temp)
    data = sorted(data,key=lambda x:x[0])
    print(data)