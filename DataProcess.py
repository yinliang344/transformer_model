#!user\bin\python3 fypj_project\ DataProcess
# -*- coding: utf-8 -*-
# @Time  : 2019/11/27 18:06
# @user  : miss
# import psycopg2
import re
from tqdm import trange,tqdm
from config import *
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from rich.progress import track

class DataProcess():
    def __init__(self):
        self.xxx = 123

    def input_password(self, database, user, password, host, port):
        '''
        :param database:
        :param user:
        :param password:
        :param host:
        :param port:
        '''
        self.data_base = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def data_base_connection(self):
        '''
        连接数据库函数
        :return: None
        '''
        self.__redis_hash = 'JudgeSentence'
        self.conn = psycopg2.connect(database=self.data_base, user=self.user, password=self.password, host=self.host,
                                     port=self.port)  # conn相当于sess
        self.cur = self.conn.cursor()  # cur是cursor的简写, 代表"光标,指示器"

    def _getSaveOriginalDataFromDataBase(self):
        '''
        从数据库中读取数据，并保存到本地
        :return: list格式的数据
        '''
        # (正文，犯罪事实，被告人姓名，被告人性别，被告人教育程度，前科数量，前科罪名，前科刑期，刑罚类别,前科罚金，涉案金额，罚金，刑期，缓刑，法条，罪名，MD5码)
        sqlStr = "select zw,fzss,bgrxm,bgrxb,bgrjycd,qksl,qkzm_id,qkxq,xflb,qkfj,saje,fj,xq,hx,fltw,zm,wsmd5 from bgrspb_xx_test3 " \
                 "where fltw!='' and xq!='' and fzss!='' and zm!='' and xflb!=0 limit 1200000"
        self.cur.execute(sqlStr)
        orig = self.cur.fetchall()  # type: list
        patten1 = r'(?s)本院认为.*?(减轻处罚|从[轻重宽严]处罚).*?如不服本判决'
        patten2 = r'(?s)本院认为.*?(数额特别巨大|数量特别巨大|数额[巨较]大|数量[巨较]大).*?如不服本判决'
        patten3 = r'(?s)本院认为.*?(情节特别严重|情节严重|情节特别恶劣|情节恶劣|情节较轻|情节轻微).*?如不服本判决'
        patten4 = r'(?s)本院认为.*?(造成特别严重后果|造成特别重大损失|造成严重后果|造成重大损失).*?如不服本判决'
        patten5 = r'(?s)本院认为.*?(自首|主动承认|认罪态度很好|认罪态度好).*?如不服本判决'
        pat1 = re.compile(patten1)
        pat2 = re.compile(patten2)
        pat3 = re.compile(patten3)
        pat4 = re.compile(patten4)
        pat5 = re.compile(patten5)
        MD5_list = []
        MD5_remove = []
        for i in trange(len(orig),):
            ori = [str(k).replace('\n', '') for k in orig[i]]
            if ori[-1] in MD5_list:
                MD5_list.remove(ori[-1])
                MD5_remove.append(ori[-1])
            elif ori[-1] in MD5_remove:
                continue
            else:
                MD5_list.append(ori[-1])
        print(len(MD5_list))
        with open('.\data\data_multi_label.txt','w+',encoding='utf-8') as data_all:
            for i in trange(len(orig)):
                ori = [str(k).replace('\n','') for k in orig[i]]
                if ori[2] == '': #如果犯罪嫌疑人姓名为空
                    ori[2] = 'xxx'
                if ori[3] == '': #如果被告人性别为空
                    ori[3] = '1'
                if ori[4] == '': #如果被告人教育程度为空
                    ori[4] = '4'
                if ori[5] == '': #如果前科数量为空
                    ori[5] = '0'
                if ori[6] == '': #如果前科罪名为空
                    ori[6] = '0'
                if ori[7] == '': #如果前科刑期为空
                    ori[7] = '0'
                if ori[8] == '': #如果刑法类别为空
                    ori[8] = '0'
                if ori[9] == '': #如果前科罚金为空
                    ori[9] = '0'
                if len(ori[15].strip().split(',')) > 1:
                    continue
                # if originalData[i][]
                if ori[-1]  not in MD5_list:
                    continue
                jielun = pat1.findall(ori[0])
                jine = pat2.findall(ori[0])
                qingjie = pat3.findall(ori[0])
                houguo = pat4.findall(ori[0])
                taidu = pat5.findall(ori[0])
                if jielun==[]:
                    jielun.append('无')
                if jine==[]:
                    jine.append('无')
                if qingjie==[]:
                    qingjie.append('无')
                if houguo==[]:
                    houguo.append('无')
                if taidu==[]:
                    taidu.append('无')
                # file.write(ori[0]+'\n\n')

                origi = ori[1:]+jielun+jine+qingjie+houguo+taidu
                data_all.write('##'.join(origi)+'\n')

    def __GetDict__(self,filepath):
        '''
        从原始数据中得到字典信息，比如将字符转换为索引，得到词典，将一些属性转换，
        最后写入待使用的文件，分为训练集和测试集
        :param filepath: 训练集和测试集所在的绝对或者相对路径
        :return: None
        '''
        with open(filepath + '/train_data.txt', 'r', encoding='utf-8') as train_data, \
                open(filepath+'/test_data.txt','r',encoding='utf-8') as test_data,\
                open(filepath+'/test_index.txt','w+',encoding='utf-8') as test_index,\
                open(filepath + '/word2index.txt', 'w+', encoding='utf-8') as word2index, \
                open(filepath + '/train_index.txt', 'w+', encoding='utf-8') as train_index, \
                open(filepath + '/qkzm_index.txt', 'w+', encoding='utf-8') as qkzm2index:
            train_data_lines = train_data.readlines()
            test_data_lines = test_data.readlines()
            word_index = {}
            word2dict = {}
            qkzm_dict = {}
            jl_dict = {"从轻处罚":0,"从重处罚":4,"从宽处罚":1,"减轻处罚":2,"从严处罚":3}
            je_dict = {"数量较大":0,"数额较大":1,"数量巨大":2,"数额巨大":3,"数量特别巨大":4,"数额特别巨大":5}
            qj_dict = {"情节轻微":0,"情节较轻":1,"情节严重":2,"情节恶劣":3,"情节特别严重":4,"情节特别恶劣":5}
            hg_dict = {"造成重大损失":0,"造成严重后果":1,"造成特别严重损失":2,"造成特别严重后果":3}
            rztd_dict = {"自首":0,"主动承认":1,"认罪态度好":2,"认罪态度很好":3}
            word2dict['PAD'] = 0
            word2dict['criminal'] = 1
            word2dict['unk'] = 2
            ss_len = 0
            qksl_x = 0
            qksl_y = 0
            qkxq_x = 0
            qkxq_y = 0
            qkfj_x = 0
            qkfj_y = 0
            saje_x = 0
            saje_y = 0
            number = 0
            for line in tqdm(train_data_lines,desc="run get mean value"):
                data = line.strip().split("##")
                s = data[0].strip()
                xq = self.xq_trans(data[11])
                qksl = int(data[4])
                qksl_x+=qksl
                qkxq = self.xq_trans(data[6])
                qkxq_x+=qkxq
                qkfj = int(data[8])
                qkfj_x+=qkfj
                saje = int(data[9])
                saje_x+=saje
                number+=1
                for word in s:
                    if word not in stop_word:
                        if word not in word_index:
                            word_index[word] = 1
                        else:
                            word_index[word] += 1
            qksl_x = qksl_x/number
            qkxq_x = qkxq_x/number
            qkfj_x = qkfj_x/number
            saje_x = saje_x/number

            for line in tqdm(train_data_lines,desc="run get variance"):
                data = line.strip().split("##")
                qkzm = data[5]
                if qkzm in qkzm_dict.keys():
                    qkzm_dict[qkzm] += 1
                else:
                    qkzm_dict[qkzm] = 1
                xq = self.xq_trans(data[11])
                qksl = int(data[4])
                qksl_y += (qksl-qksl_x)**2
                qkxq = self.xq_trans(data[6])
                qkxq_y += (qkxq-qkxq_x)**2
                qkfj = int(data[8])
                qkfj_y += (qkfj-qkfj_x)**2
                saje = int(data[9])
                saje_y += (saje-saje_x)**2
            qksl_y = (qksl_y/number)**0.5
            qkxq_y = (qkxq_y/number)**0.5
            qkfj_y = (qkfj_y/number)**0.5
            saje_y = (saje_y/number)**0.5

            qkzm_rank = sorted(qkzm_dict.items(), key= lambda x:x[1],reverse=True)
            qkzm_dict = {}
            for item in qkzm_rank:
                qkzm_dict[item[0]] = len(qkzm_dict)
                qkzm2index.write(str(item[0])+' '+str(item[1])+'\n')
            for item in word_index.items():
                word2dict[item[0]] = len(word2dict)
            for item in word2dict.items():
                word2index.write(str(item[0]) + ' ' + str(item[1]) + '\n')
            number = 0
            #write train file
            for line in tqdm(train_data_lines, desc="write train file"):
                data = line.strip().split("##")
                s = data[0].strip()
                xm = str(data[1])
                # (犯罪事实，被告人姓名，被告人性别，被告人教育程度，前科数量，前科罪名，前科刑期，刑罚类别,前科罚金，涉案金额，罚金，刑期，缓刑，MD5码)
                xb = str(data[2])
                jycd = str(data[3])
                qksl = int(data[4])
                qksl = (qksl-qksl_x)/qksl_y
                try:
                    qkzm = str(qkzm_dict[data[5]])
                    qkxq = self.xq_trans(data[6])
                    qkxq = (qkxq-qkxq_x)/qkxq_y
                    xflb = str(data[7])
                    qkfj = int(data[8])
                    qkfj = (qkfj-qkfj_x)/qkfj_y
                    saje = int(data[9])
                    saje = (saje-saje_x)/saje_y
                    fj = int(data[10])
                    fj = str(self.__je_trainfor__(fj))
                except:
                    continue

                try:
                    xq = self.xq_trans(data[11])
                    hx = self.xq_trans(data[12])
                    xq = str(self.__xq_transfor__(xq))
                    hx = str(self.__xq_transfor__(hx))
                except:
                    continue

                try:
                    jl = str(jl_dict[data[14]])
                    je = str(je_dict[data[15]])
                    qj = str(qj_dict[data[16]])
                    hg = str(hg_dict[data[17]])
                    rztd = str(rztd_dict[data[18]])
                except:
                    continue
                ss_len += len(s)
                number+=1
                s_index = ['0' for i in range(truncature_len)]
                for i in range(truncature_len):
                    if i < len(s):
                        if s[i]==xm:
                            s_index[i] = str(word2dict['criminal'])
                        elif s[i] in word2dict:
                            s_index[i] = str(word2dict[s[i]])
                        else:
                            s_index[i] = str(word2dict['unk'])
                data_line = [" ".join(s_index)]
                # 犯罪事实描述，姓名，性别，教育程度，前科数量，前科罪名，前科刑期，刑罚类别，涉案金额，罚金，刑期，缓刑，结论，金额，情节，后果，认罪态度
                data_line = data_line+[xm,xb,jycd,qksl,qkzm,qkxq,xflb,qkfj,saje,fj,xq,hx,jl,je,qj,hg,rztd]
                train_index.write("##".join(data_line)+'\n')
            #write test file
            for line in tqdm(test_data_lines, desc="write test file"):
                data = line.strip().split("##")
                s = data[0].strip()
                xm = str(data[1])
                # (犯罪事实，被告人姓名，被告人性别，被告人教育程度，前科数量，前科罪名，前科刑期，刑罚类别,前科罚金，涉案金额，罚金，刑期，缓刑，MD5码)
                xb = str(data[2])
                jycd = str(data[3])
                qksl = int(data[4])
                qksl = (qksl - qksl_x) / qksl_y
                try:
                    qkzm = str(qkzm_dict[data[5]])
                    qkxq = self.xq_trans(data[6])
                    qkxq = (qkxq - qkxq_x) / qkxq_y
                    xflb = str(data[7])
                    qkfj = int(data[8])
                    qkfj = (qkfj - qkfj_x) / qkfj_y
                    saje = int(data[9])
                    saje = (saje - saje_x) / saje_y
                    fj = int(data[10])
                    fj = str(self.__je_trainfor__(fj))
                except:
                    continue

                xq = self.xq_trans(data[11])
                hx = self.xq_trans(data[12])
                xq = str(self.__xq_transfor__(xq))
                hx = str(self.__xq_transfor__(hx))
                try:
                    jl = str(jl_dict[data[14]])
                    je = str(je_dict[data[15]])
                    qj = str(qj_dict[data[16]])
                    hg = str(hg_dict[data[17]])
                    rztd = str(rztd_dict[data[18]])
                except:
                    continue

                s_index = ['0' for i in range(truncature_len)]
                for i in range(truncature_len):
                    if i < len(s):
                        if s[i]==xm:
                            s_index[i] = str(word2dict['criminal'])
                        elif s[i] in word2dict:
                            s_index[i] = str(word2dict[s[i]])
                        else:
                            s_index[i] = str(word2dict['unk'])
                data_line = [" ".join(s_index)]
                #犯罪事实描述，姓名，性别，教育程度，前科数量，前科罪名，前科刑期，刑罚类别，涉案金额，罚金，刑期，缓刑，结论，金额，情节，后果，认罪态度
                data_line = data_line+[xm,xb,jycd,qksl,qkzm,qkxq,xflb,qkfj,saje,fj,xq,hx,jl,je,qj,hg,rztd]
                test_index.write("##".join(data_line)+'\n')
            print(ss_len/number)
            print(number)
            print(len(word_index))
            print(len(word2dict))

    def get_dict_multi_lablel(self,filepath):
        '''
        从原始数据中得到字典信息，比如将字符转换为索引，得到词典，将一些属性转换，
        最后写入待使用的文件，分为训练集和测试集
        :param filepath: 训练集和测试集所在的绝对或者相对路径
        :return: None
        '''
        with open(filepath + '/train_data_multi.txt', 'r', encoding='utf-8') as train_data, \
                open(filepath + '/test_data_multi.txt', 'r', encoding='utf-8') as test_data, \
                open(filepath + '/test_index.txt', 'w+', encoding='utf-8') as test_index, \
                open(filepath + '/word2index.txt', 'w+', encoding='utf-8') as word2index, \
                open(filepath + '/train_index.txt', 'w+', encoding='utf-8') as train_index, \
                open(filepath + '/zm_dict.txt', 'r', encoding='utf-8') as zm_index:
            train_data_lines = train_data.readlines()
            test_data_lines = test_data.readlines()
            zm_index_lines = zm_index.readlines()
            zm_dict = {}
            word_index = {}
            word2dict = {}
            jl_dict = {"无":0, "从轻处罚": 1,  "从宽处罚": 2, "减轻处罚": 3, "从严处罚": 4,"从重处罚": 5}
            je_dict = {"无":0, "数量较大": 1, "数额较大": 2, "数量巨大": 3, "数额巨大": 4, "数量特别巨大": 5, "数额特别巨大": 6}
            qj_dict = {"无":0, "情节轻微": 1, "情节较轻": 2, "情节严重": 3, "情节恶劣": 4, "情节特别严重": 5, "情节特别恶劣": 6}
            hg_dict = {"无":0, "造成重大损失": 1, "造成严重后果": 2, "造成特别严重损失": 3, "造成特别严重后果": 4}
            rztd_dict = {"无":0, "自首": 1, "主动承认": 2, "认罪态度好": 3, "认罪态度很好": 4}
            word2dict['PAD'] = 0
            word2dict['criminal'] = 1
            word2dict['unk'] = 2
            ss_len = 0
            qksl_x = 0
            qksl_y = 0
            qkxq_x = 0
            qkxq_y = 0
            qkfj_x = 0
            qkfj_y = 0
            saje_x = 0
            saje_y = 0
            number = 0
            for line in zm_index_lines:
                item = line.strip().split()
                zm_dict[item[0]] = item[1]
            for line in tqdm(train_data_lines, desc="run get mean value"):
                data = line.strip().split("##")
                s = data[0].strip()
                xq = self.xq_trans(data[11])
                qksl = int(data[4])
                qksl_x += qksl
                qkxq = self.xq_trans(data[6])
                qkxq_x += qkxq
                qkfj = int(data[8])
                qkfj_x += qkfj
                saje = int(data[9])
                saje_x += saje
                number += 1
                for word in s:
                    if word not in stop_word:
                        if word not in word_index:
                            word_index[word] = 1
                        else:
                            word_index[word] += 1
            qksl_x = qksl_x / number
            qkxq_x = qkxq_x / number
            qkfj_x = qkfj_x / number
            saje_x = saje_x / number

            for line in tqdm(train_data_lines, desc="run get variance"):
                data = line.strip().split("##")
                xq = self.xq_trans(data[11])
                qksl = int(data[4])
                qksl_y += (qksl - qksl_x) ** 2
                qkxq = self.xq_trans(data[6])
                qkxq_y += (qkxq - qkxq_x) ** 2
                qkfj = int(data[8])
                qkfj_y += (qkfj - qkfj_x) ** 2
                saje = int(data[9])
                saje_y += (saje - saje_x) ** 2
            qksl_y = (qksl_y / number) ** 0.5
            qkxq_y = (qkxq_y / number) ** 0.5
            qkfj_y = (qkfj_y / number) ** 0.5
            saje_y = (saje_y / number) ** 0.5

            for item in word_index.items():
                word2dict[item[0]] = len(word2dict)
            for item in word2dict.items():
                word2index.write(str(item[0]) + ' ' + str(item[1]) + '\n')
            number = 0
            # write train file
            xq_dict = {}
            for line in tqdm(train_data_lines, desc="write train file"):
                data = line.strip().split("##")
                s = data[0].strip()
                if len(s)<=150:
                    continue
                xm = str(data[1])
                # (犯罪事实，被告人姓名，被告人性别，被告人教育程度，前科数量，前科罪名，前科刑期，刑罚类别,前科罚金，涉案金额，罚金，刑期，缓刑，MD5码)
                xb = str(data[2])
                jycd = str(data[3])
                qksl = int(data[4])
                qksl = str((qksl - qksl_x) / qksl_y)
                try:
                    qkzm = str(zm_dict[data[5]])
                    qkxq = self.xq_trans(data[6])
                    qkxq = str((qkxq - qkxq_x) / qkxq_y)
                    xflb = str(data[7])
                    qkfj = int(data[8])
                    qkfj = str((qkfj - qkfj_x) / qkfj_y)
                    saje = int(data[9])
                    saje = str((saje - saje_x) / saje_y)
                    fj = int(data[10])
                    fj = str(self.__je_trainfor__(fj))
                except:
                    continue

                try:
                    xq = str(data[11])
                    hx = str(data[12])
                    ft = str(data[13])
                    zm = str(data[14])
                except:
                    continue

                try:
                    jl = str(jl_dict[data[15]])
                    je = str(je_dict[data[16]])
                    qj = str(qj_dict[data[17]])
                    hg = str(hg_dict[data[18]])
                    rztd = str(rztd_dict[data[19]])
                except:
                    continue

                if xq not in xq_dict.keys():
                    xq_dict[xq] = 1
                else:
                    xq_dict[xq] += 1
                ss_len += len(s)
                number += 1
                s_index = ['0' for i in range(truncature_len_multi)]
                for i in range(truncature_len_multi):
                    if i < len(s):
                        if s[i] == xm:
                            s_index[i] = str(word2dict['criminal'])
                        elif s[i] in word2dict:
                            s_index[i] = str(word2dict[s[i]])
                        else:
                            s_index[i] = str(word2dict['unk'])
                data_line = [" ".join(s_index)]
                # 犯罪事实描述，姓名，性别，教育程度，前科数量，前科罪名，前科刑期，刑罚类别，涉案金额，罚金，刑期，缓刑，法条，罪名，结论，金额，情节，后果，认罪态度
                data_line = data_line + [xm, xb, jycd, qksl,
                                         qkzm, qkxq, xflb, qkfj,
                                         saje, fj, xq, hx,ft,zm,
                                         jl, je, qj, hg,rztd]
                train_index.write("##".join(data_line) + '\n')
            # write test file
            for line in tqdm(test_data_lines, desc="write test file"):
                data = line.strip().split("##")
                s = data[0].strip()
                if len(s)<=150:
                    continue
                xm = str(data[1])
                # (犯罪事实，被告人姓名，被告人性别，被告人教育程度，前科数量，前科罪名，前科刑期，刑罚类别,前科罚金，涉案金额，罚金，刑期，缓刑，MD5码)
                xb = str(data[2])
                jycd = str(data[3])
                qksl = int(data[4])
                qksl = str((qksl - qksl_x) / qksl_y)
                try:
                    qkzm = str(zm_dict[data[5]])
                    qkxq = self.xq_trans(data[6])
                    qkxq = str((qkxq - qkxq_x) / qkxq_y)
                    xflb = str(data[7])
                    qkfj = int(data[8])
                    qkfj = str((qkfj - qkfj_x) / qkfj_y)
                    saje = int(data[9])
                    saje = str((saje - saje_x) / saje_y)
                    fj = int(data[10])
                    fj = str(self.__je_trainfor__(fj))
                except:
                    continue
                try:
                    xq = str(data[11])
                    hx = str(data[12])
                    ft = str(data[13])
                    zm = str(data[14])
                except:
                    continue
                try:
                    jl = str(jl_dict[data[15]])
                    je = str(je_dict[data[16]])
                    qj = str(qj_dict[data[17]])
                    hg = str(hg_dict[data[18]])
                    rztd = str(rztd_dict[data[19]])
                except:
                    continue

                s_index = ['0' for i in range(truncature_len_multi)]
                for i in range(truncature_len_multi):
                    if i < len(s):
                        if s[i] == xm:
                            s_index[i] = str(word2dict['criminal'])
                        elif s[i] in word2dict:
                            s_index[i] = str(word2dict[s[i]])
                        else:
                            s_index[i] = str(word2dict['unk'])
                data_line = [" ".join(s_index)]
                # 犯罪事实描述，姓名，性别，教育程度，前科数量，前科罪名，前科刑期，刑罚类别，涉案金额，罚金，刑期，缓刑，法条，罪名，结论，金额，情节，后果，认罪态度
                data_line = data_line + [xm, xb, jycd, qksl,
                                         qkzm, qkxq, xflb, qkfj,
                                         saje, fj, xq, hx,ft,zm,
                                         jl, je, qj, hg,rztd]
                test_index.write("##".join(data_line) + '\n')
            print(ss_len / number)
            print(number)
            print(len(word_index))
            print(len(word2dict))
            print(xq_dict)

    def get_dict(self,filepath):
        """
        :param filepath: the path of file
        :return: None
        """
        self.__GetDict__(filepath=filepath)

    def __xq_trans__(self,xq):
        '''
        :param xq: str型
        :return:
        '''
        xq_trans = 0
        if ',' in xq:
            xq_s = xq.split(',')
            xq_trans = int(xq_s[0])*12+int(xq_s[1])
        else:
            xq_trans = int(xq)*12
        return xq_trans

    def xq_trans(self,xq):
        '''
        :param xq:
        :return:
        '''
        return self.__xq_trans__(xq=xq)

    def batch_generator(self,all_data, all_label,all_input_float,all_input_int, batch_size, shuffle=True):
        '''
        batch生成器，生成一个可以随机取batch的生成器
        :param all_data: 输入数据，np.array数据
        :param all_label:标签数据，np.array数据
        :all_input:属性数据，np.array数据
        :param batch_size: 每个batch的大小，int型
        :param shuffle: 是否随机打乱数据,bool型数据
        :return: 一个batch生成器
        '''
        len_data = len(all_data)
        batch_len = int(len_data // batch_size)
        while True:
            batch_number = 0
            init_index = [i for i in range(len(all_data))]
            if shuffle == True:
                random.shuffle(init_index)
            batch_data = []
            batch_label = []
            batch_input_float = []
            batch_input_int = []
            for i in init_index:
                batch_data.append(all_data[i])
                batch_label.append(all_label[i])
                batch_input_float.append(all_input_float[i])
                batch_input_int.append(all_input_int[i])
                if len(batch_label) == batch_size:
                    batch_number += 1
                    yield batch_data,batch_input_float,batch_input_int, batch_label
                    batch_data = []
                    batch_label = []
                    batch_input_float = []
                    batch_input_int = []
                if batch_number == batch_len:
                    break

    def __read_file__(self,filepath):
        '''
        读取处理好的文件，准备载入模型
        :param filepath: 文件地址，绝对地址或者相对地址
        :return: train_x_all, train_y_input_all, train_y_output_all,test_x_all, test_y_input_all, test_y_output_all
        '''
        with open(filepath+'/train_index.txt','r',encoding='utf-8') as data_train, \
                open(filepath+'/test_index.txt','r',encoding='utf-8') as data_test:
            data_train_lines = data_train.readlines()
            data_test_lines = data_test.readlines()
            train_x_all = []
            train_y_input_float = []
            train_y_input_int = []
            train_y_output_all = []
            test_x_all = []
            test_y_input_float = []
            test_y_input_int = []
            test_y_output_all = []
            for line in tqdm(data_train_lines, desc='train file reading'):
                data_line = line.strip().split('##')
                x = list(map(int, data_line[0].strip().split(' ')))
                # x_all.append(x)
                try:
                    xb = float(data_line[2])
                except:
                    continue
                train_x_all.append(x)
                jycd = int(data_line[3])
                qksl = float(data_line[4])
                qkzm = int(data_line[5])
                qkxq = float(data_line[6])
                xflb = int(data_line[7])
                qkfj = float(data_line[8])
                saje = float(data_line[9])
                xq = int(data_line[11])
                jl = int(data_line[15])
                je = int(data_line[16])
                qj = int(data_line[17])
                hg = int(data_line[18])
                rztd = int(data_line[19])
                train_y_input_float.append([qksl, qkxq, qkfj, saje])
                train_y_input_int.append([xb, jycd, qkzm, jl, je, qj, hg, rztd])
                train_y_output_all.append(xq)
            for line in tqdm(data_test_lines, desc='test file reading'):
                data_line = line.strip().split('##')
                x = list(map(int, data_line[0].strip().split(' ')))
                # x_all.append(x)
                try:
                    xb = float(data_line[2])
                except:
                    continue
                test_x_all.append(x)
                jycd = float(data_line[3])
                qksl = float(data_line[4])
                qkzm = float(data_line[5])
                qkxq = float(data_line[6])
                xflb = float(data_line[7])
                qkfj = float(data_line[8])
                saje = float(data_line[9])
                xq = int(data_line[11])
                jl = float(data_line[15])
                je = float(data_line[16])
                qj = float(data_line[17])
                hg = float(data_line[18])
                rztd = float(data_line[19])
                test_y_input_float.append([qksl, qkxq, qkfj, saje])
                test_y_input_int.append([xb, jycd, qkzm, jl, je, qj, hg, rztd])
                test_y_output_all.append(xq)
            train_x_all = np.array(train_x_all)
            train_y_input_float = np.array(train_y_input_float)
            train_y_input_int = np.array(train_y_input_int)
            train_y_output_all = np.array(train_y_output_all)
            test_x_all = np.array(test_x_all)
            test_y_input_float = np.array(test_y_input_float)
            test_y_input_int = np.array(test_y_input_int)
            test_y_output_all = np.array(test_y_output_all)

            return train_x_all, train_y_input_float, train_y_input_int, train_y_output_all, test_x_all, test_y_input_float,test_y_input_int, test_y_output_all

    def read_file(self,filepath):
        '''
        :param filepath:
        :return:
        '''
        return self.__read_file__(filepath=filepath)

    def read_file_multi(self,filepath):
        '''
        读取处理好的文件，准备载入模型
        :param
        filepath: 文件地址，绝对地址或者相对地址
        :return: train_x_all, train_y_input_all, train_y_output_all, test_x_all, test_y_input_all, test_y_output_all
        '''
        with open(filepath+'/train_index.txt','r',encoding='utf-8') as data_train, \
                open(filepath+'/test_index.txt','r',encoding='utf-8') as data_test:
            data_train_lines = data_train.readlines()
            data_test_lines = data_test.readlines()
            train_x_all = []
            train_y_input_float = []
            train_y_input_int = []
            train_y_output_all = []
            test_x_all = []
            test_y_input_float = []
            test_y_input_int = []
            test_y_output_all = []
            for line in tqdm(data_train_lines, desc='train file reading'):
                data_line = line.strip().split('##')
                x = list(map(int, data_line[0].strip().split(' ')))
                # x_all.append(x)
                try:
                    xb = float(data_line[2])
                except:
                    continue
                train_x_all.append(x)
                jycd = int(data_line[3])
                qksl = float(data_line[4])
                qkzm = int(data_line[5])
                qkxq = float(data_line[6])
                qkfj = float(data_line[8])
                saje = float(data_line[9])
                xq = int(data_line[11])
                ft = int(data_line[13])
                zm = int(data_line[14])
                jl = int(data_line[15])
                je = int(data_line[16])
                qj = int(data_line[17])
                hg = int(data_line[18])
                rztd = int(data_line[19])
                train_y_input_float.append([qksl, qkxq, qkfj, saje])
                train_y_input_int.append([xb, jycd, qkzm, jl, je, qj, hg, rztd])
                train_y_output_all.append([ft,xq,zm])
            for line in tqdm(data_test_lines, desc='test file reading'):
                data_line = line.strip().split('##')
                x = list(map(int, data_line[0].strip().split(' ')))
                # x_all.append(x)
                try:
                    xb = float(data_line[2])
                except:
                    continue
                test_x_all.append(x)
                jycd = float(data_line[3])
                qksl = float(data_line[4])
                qkzm = float(data_line[5])
                qkxq = float(data_line[6])
                qkfj = float(data_line[8])
                saje = float(data_line[9])
                xq = int(data_line[11])
                ft = int(data_line[13])
                zm = int(data_line[14])
                jl = float(data_line[15])
                je = float(data_line[16])
                qj = float(data_line[17])
                hg = float(data_line[18])
                rztd = float(data_line[19])
                test_y_input_float.append([qksl, qkxq, qkfj, saje])
                test_y_input_int.append([xb, jycd, qkzm, jl, je, qj, hg, rztd])
                test_y_output_all.append([ft,xq,zm])
            train_x_all = np.array(train_x_all)
            train_y_input_float = np.array(train_y_input_float)
            train_y_input_int = np.array(train_y_input_int)
            train_y_output_all = np.array(train_y_output_all)
            test_x_all = np.array(test_x_all)
            test_y_input_float = np.array(test_y_input_float)
            test_y_input_int = np.array(test_y_input_int)
            test_y_output_all = np.array(test_y_output_all)

            return train_x_all, train_y_input_float, train_y_input_int, train_y_output_all, test_x_all, test_y_input_float, test_y_input_int, test_y_output_all

    def read_file_test(self,filepath):
        with open(filepath+'/test_index.txt','r',encoding='utf-8') as data_test:
            data_test_lines = data_test.readlines()
            test_x_all = []
            test_y_input_float = []
            test_y_input_int = []
            test_y_output_all = []

            for line in tqdm(data_test_lines, desc='test file reading'):
                data_line = line.strip().split('##')
                x = list(map(int, data_line[0].strip().split(' ')))
                # x_all.append(x)
                try:
                    xb = float(data_line[2])
                except:
                    continue
                test_x_all.append(x)
                jycd = float(data_line[3])
                qksl = float(data_line[4])
                qkzm = float(data_line[5])
                qkxq = float(data_line[6])
                qkfj = float(data_line[8])
                saje = float(data_line[9])
                xq = int(data_line[11])
                ft = int(data_line[13])
                zm = int(data_line[14])
                jl = float(data_line[15])
                je = float(data_line[16])
                qj = float(data_line[17])
                hg = float(data_line[18])
                rztd = float(data_line[19])
                test_y_input_float.append([qksl, qkxq, qkfj, saje])
                test_y_input_int.append([xb, jycd, qkzm, jl, je, qj, hg, rztd])
                test_y_output_all.append([ft, xq, zm])
            test_x_all = np.array(test_x_all)
            test_y_input_float = np.array(test_y_input_float)
            test_y_input_int = np.array(test_y_input_int)
            test_y_output_all = np.array(test_y_output_all)

            return test_x_all, test_y_input_float, test_y_input_int, test_y_output_all

    def __xq_transfor__(self,xq):
        '''
        将整数刑期划分为十个区间，分为十类
        :param xq: int型，区间为[0,180]
        :return: 划分后的类别索引
        '''
        xq_c = 0
        if xq == 0:
            xq_c = 0
        elif xq > 0 and xq <= 6:
            xq_c = 1
        elif xq > 6 and xq <= 9:
            xq_c = 2
        elif xq > 9 and xq <= 12:
            xq_c = 3
        elif xq > 12 and xq <= 24:
            xq_c = 4
        elif xq > 24 and xq <= 36:
            xq_c = 5
        elif xq > 36 and xq <= 60:
            xq_c = 6
        elif xq > 60 and xq <= 84:
            xq_c = 7
        elif xq > 84 and xq <= 120:
            xq_c = 8
        elif xq > 120 and xq <= 300:
            xq_c = 9
        elif xq > 300:
            xq_c = 10
        return xq_c

    def __je_trainfor__(self,fj):
        '''
        :param fj: 罚金金额，int型或者float型
        :return: 将罚金等比例缩小10000倍，返回，float型
        '''
        return fj/10000

    def data_split(self,file_path):
        '''
        分割原始数据集为训练集和测试集，分割比例为8：2，
        并去除刑期大于有期徒刑15年的、数据读取错误的、以及限制刑期为0的条数为30000条
        :param file_path: the path of dataset
        :return: None
        '''
        with open(file_path+'/data_all.txt','r',encoding='utf-8') as data_all,\
                open(file_path+'/train_data.txt','w+',encoding='utf-8') as train_data,\
                open(file_path+'/test_data.txt','w+',encoding='utf-8') as test_data:
            data_all_lines = data_all.readlines()
            xflb_dict = ['4', '5', '7']
            xq_dict = {}
            xq_l = []
            xq_dict['0'] = 0
            data_l = []
            for line in tqdm(data_all_lines, total=len(data_all_lines),desc="run get xq"):
                data = line.strip().split("##")
                xq = self.xq_trans(data[11])
                if xq < 0 or xq > 180 or (data[7] in xflb_dict):
                    continue
                if xq_dict['0'] > 30000 and xq == 0:
                    continue
                if str(xq) in xq_dict.keys():
                    xq_dict[str(xq)] += 1
                else:
                    xq_dict[str(xq)] = 1
                try:
                    qkxq = self.xq_trans(data[6])
                    qkxq = str(self.__xq_transfor__(qkxq))
                    xflb = str(data[7])
                    qkfj = int(data[8])
                    qkfj = str(self.__je_trainfor__(qkfj))
                    saje = int(data[9])
                    saje = str(self.__je_trainfor__(saje))
                    fj = int(data[10])
                    xq = self.__xq_transfor__(xq)
                except:
                    continue
                xq_l.append(xq)
                data_l.append(data)
            ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=5)
            x = ss.split(data_l, xq_l)
            train_index, test_index = next(x)
            train_data_l,test_data_l = np.array(data_l)[train_index],np.array(data_l)[test_index]
            for data in tqdm(train_data_l,desc='write train file'):
                train_data.write('##'.join(data)+'\n')
            for data in tqdm(test_data_l,desc='write test file'):
                test_data.write('##'.join(data)+'\n')

    def data_split_multi(self,file_path):
        '''
        分割原始数据集为训练集和测试集，分割比例为8：2，
        并去除刑期大于有期徒刑15年的、数据读取错误的、以及限制刑期为0的条数为30000条
        :param
        file_path: the
        path
        of
        dataset
        :return: None
        '''
        with open(file_path+'/data_multi_label.txt','r',encoding='utf-8') as data_all,\
                open(file_path+'/train_data_multi.txt','w+',encoding='utf-8') as train_data,\
                open(file_path+'/test_data_multi.txt','w+',encoding='utf-8') as test_data, \
                open(file_path+'/ft_dict.txt','w+',encoding='utf-8') as ft_count, \
                open(file_path+'/zm_dict.txt','w+',encoding='utf-8') as zm_count, \
                open(file_path+'/xq_dict.txt','w+',encoding='utf-8') as xq_count:
            data_all_lines = data_all.readlines()
            xflb_dict = ['4', '5', '7']
            xq_dict = {}
            zm_dict = {}
            ft_dict = {}
            xq_l = []
            xq_dict['0'] = 0
            data_l = []
            for line in tqdm(data_all_lines,desc='run get ft and zm'):
                data = line.strip().split("##")
                zm = data[14]
                if zm in zm_dict.keys():
                    zm_dict[zm] += 1
                else:
                    zm_dict[zm] = 1
                ft = data[13].strip().split(',')
                ft = ft[0]
                if ft in ft_dict.keys():
                    ft_dict[ft] += 1
                else:
                    ft_dict[ft] = 1

            zm_dic = sorted(zm_dict.items(), key= lambda x:x[1],reverse=True)
            zm_dict = {}
            for i in range(100):
                zm_dict[zm_dic[i][0]] = zm_dic[i][1]
            zm_dict = {}
            for i in range(100):
                zm_dict[zm_dic[i][0]] = len(zm_dict)
            for item in zm_dict.items():
                zm_count.write(str(item[0]) + ' ' + str(item[1]) + '\n')

            ft_dic = sorted(ft_dict.items(), key=lambda x: x[1], reverse=True)
            ft_dict = {}
            for i in range(100):
                ft_dict[ft_dic[i][0]] = ft_dic[i][1]
            ft_dict = {}
            for i in range(100):
                ft_dict[ft_dic[i][0]] = len(ft_dict)
            for item in ft_dict.items():
                ft_count.write(str(item[0]) + ' ' + str(item[1]) + '\n')

            for line in tqdm(data_all_lines, total=len(data_all_lines),desc="run get xq"):
                data = line.strip().split("##")
                zm = data[14]
                ft = data[13].strip().split(',')
                ft = ft[0]
                if zm not in zm_dict.keys():
                    continue
                if ft not in ft_dict.keys():
                    continue

                zm = str(zm_dict[zm])
                ft = str(ft_dict[ft])

                xq = self.xq_trans(data[11])
                if (data[7]=='1' or data[7]=='2') and xq!=0:
                    continue
                if data[7] == '6':
                    xq = 0
                if data[7] == '4' or data[7] == '5' or data[7] == '7':
                    xq = 400
                try:
                    qkxq = self.xq_trans(data[6])
                    qkxq = str(self.__xq_transfor__(qkxq))
                    xflb = str(data[7])
                    qkfj = int(data[8])
                    qkfj = str(self.__je_trainfor__(qkfj))
                    saje = int(data[9])
                    saje = str(self.__je_trainfor__(saje))
                    fj = int(data[10])
                    xq = self.__xq_transfor__(xq)
                    hx = self.xq_trans(data[12])
                    hx = self.__xq_transfor__(hx)
                except:
                    continue
                if str(xq) in xq_dict.keys():
                    xq_dict[str(xq)] += 1
                else:
                    xq_dict[str(xq)] = 1
                xq_l.append(xq)
                data_l.append(data[:11]+[str(xq),str(hx),ft,zm]+data[16:])
            for item in xq_dict.items():
                xq_count.write(str(item[0]) + ' ' + str(item[1]) + '\n')
            ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=5)
            x = ss.split(data_l, xq_l)
            train_index, test_index = next(x)
            print(len(train_index))
            print(len(test_index))
            train_data_l,test_data_l = np.array(data_l)[train_index],np.array(data_l)[test_index]
            for data in tqdm(train_data_l,desc='write train file'):
                train_data.write('##'.join(data)+'\n')
            for data in tqdm(test_data_l,desc='write test file'):
                test_data.write('##'.join(data)+'\n')
            name_list = []
            number_list = []
            xq_dic = sorted(xq_dict.items(), key=lambda x: int(x[0]))
            for item in xq_dic:
                xq_count.write(str(item[0]) + ' ' + str(item[1]) + '\n')
                name_list.append(item[0])
                number_list.append(item[1])
            self.autolabel(plt.bar(range(len(number_list)), number_list, color='rgb', tick_label=name_list))
            plt.show()

    def autolabel(self, rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % int(height))

    def test(self):
        for i in track(range(100000),description='doing'):
            for j in range(10000):
                s = 1


if __name__ == '__main__':
    DP = DataProcess()
    # DP.data_base_connection()
    # DP._getSaveOriginalDataFromDataBase()
    # DP.__datasplit__('./data')
    # DP.data_split_multi(file_path='./data_multi')
    # DP.get_dict_multi_lablel(filepath='./data_multi')
    # DP.get_dict('./data')
