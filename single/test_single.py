#!user\bin\python3 fypj_project\ test2
# -*- coding: utf-8 -*-
# @Time  : 2019/12/9 19:22
# @user  : miss
from single.tranformer_HAN_single import HAN_model
from DataProcess import DataProcess
import numpy as np
import config as cf
import tensorflow as tf
from tqdm import trange
from validation import validation

DP = DataProcess()
train_x_all, train_y_input_float, train_y_input_int, train_y_output_all, test_x_all, test_y_input_float, test_y_input_int, test_y_output_all= DP.read_file('./data_multi')
print('1、load data完成')
Model = HAN_model()
print('2、构造模型完成')
# # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
# opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Model.all_loss)
saver = tf.train.Saver(max_to_keep=10)
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    # merged = tf.summary.merge_all()
    # writer_train = tf.summary.FileWriter("logs_single_extra/train", sess.graph)
    # writer_test = tf.summary.FileWriter("logs_single_extra/test")
    # sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state('./single/ckpt_single/')
    saver.restore(sess, save_path=ckpt.model_checkpoint_path)
    print('3、初始化完成')
    print('4、开始测试')
    rule_all_acc = 0
    zm_all_acc = 0
    xq_all_acc = 0
    xq_pre_all = []
    xq_label_true = []
    batch_number = int(len(test_y_output_all)//cf.batch_size)
    batch_test= DP.batch_generator(all_data=test_x_all,
                                   all_label=test_y_output_all,
                                   all_input_float=test_y_input_float,
                                   all_input_int=test_y_input_int,
                                   batch_size=cf.batch_size,
                                   shuffle=True)
    for k in trange(batch_number):
        batch_data_test,batch_input_test_float,batch_input_test_int, batch_label_test = next(batch_test)
        feed_dic_test = {Model.input_data: batch_data_test,
                         Model.input_shuxing_float:batch_input_test_float,
                         Model.input_shuxing_int:batch_input_test_int,
                         Model.input_label_xq: batch_label_test,
                         Model.keep_rate: 1.0,
                         Model.is_trainning: False}
        xq_pre = sess.run(Model.xq_max_index,feed_dict=feed_dic_test)
        xq_pre_all += list(xq_pre)
        xq_label_true += list(batch_label_test)
    Valid = validation(label=xq_label_true,prediction=xq_pre_all)
    print('acc: ',Valid.acc())
    print("f1: ",Valid.f1())
    print("MP: ",Valid.MP())
    print("MR: ",Valid.MR())
