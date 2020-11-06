#!user\bin\python3 transformer_single_extra\ tranformer_HAN_single
# -*- coding: utf-8 -*-
# @Time  : 2019/12/15 15:47
# @user  : miss

import tensorflow as tf
from config import *
import numpy as np
import Multi_Head_Attention as MHA


class HAN_model():
    def __init__(self):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size_multi
        self.truncature_len = truncature_len_multi
        self.hidden_size = hidden_size
        self.len_sq = len_of_sq
        self.len_word = len_of_word
        self.shuxing_number_float = shuxing_number_float
        self.shuxing_number_int = shuxing_number_int
        self.xavier_initialzer = tf.contrib.layers.xavier_initializer()
        self.He_initialzer = tf.contrib.layers.variance_scaling_initializer()

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.int32, shape=(None, self.truncature_len),name='input_data')
            self.input_shuxing_float = tf.placeholder(dtype=tf.float32,shape=(None,self.shuxing_number_float),name='input_shuxing_float')
            self.input_shuxing_int = tf.placeholder(dtype=tf.int32,shape=(None,self.shuxing_number_int),name='input_shuxing_int')
            self.input_label_xq = tf.placeholder(dtype=tf.int64, shape=(None,), name='input_label_xq')
            self.keep_rate = tf.placeholder(dtype=tf.float32, shape=(None), name='keep')
            self.is_trainning = tf.placeholder(dtype=tf.bool, shape=(None), name='trainning')

        with tf.name_scope('embedding'):
            embedding_table = tf.get_variable(name='embedding_table',
                                              shape=(self.vocab_size, self.embedding_size),
                                              dtype=tf.float32,
                                              initializer=self.xavier_initialzer)
            data_embedding = tf.nn.embedding_lookup(embedding_table, self.input_data)
            data_embedding = MHA.layer_norm(data_embedding,scope='embedding_ln_norm')
            # self.xb_embedding = self.embedding(name='xq_embedding',
            #                                    shape=(xb_number,32),
            #                                    input=tf.slice(self.input_shuxing_int,[0,0],[-1,1]))
            # # print(self.xb_embedding)
            # self.jycd_embedding = self.embedding(name='jycd_embedding',
            #                                    shape=(jycd_number, 32),
            #                                    input=tf.slice(self.input_shuxing_int, [0, 1], [-1, 1]))
            # # print(self.jycd_embedding)
            # self.qkzm_embedding = self.embedding(name='qkzm_embedding',
            #                                      shape=(zm_number, 64),
            #                                      input=tf.slice(self.input_shuxing_int, [0, 2], [-1, 1]))
            # self.xflb_embedding = self.embedding(name='xflb_embedding',
            #                                      shape=(xflb_number, 32),
            #                                      input=tf.slice(self.input_shuxing_int, [0, 3], [-1, 1]))
            # self.jl_embedding = self.embedding(name='jl_embedding',
            #                                      shape=(jl_number, 32),
            #                                      input=tf.slice(self.input_shuxing_int, [0, 4], [-1, 1]))
            # self.je_embedding = self.embedding(name='je_embedding',
            #                                      shape=(je_number, 32),
            #                                      input=tf.slice(self.input_shuxing_int, [0, 5], [-1, 1]))
            # self.qj_embedding = self.embedding(name='qj_embedding',
            #                                      shape=(qj_number, 32),
            #                                      input=tf.slice(self.input_shuxing_int, [0, 6], [-1, 1]))
            # self.hg_embedding = self.embedding(name='hg_embedding',
            #                                      shape=(hg_number, 32),
            #                                      input=tf.slice(self.input_shuxing_int, [0, 7], [-1, 1]))
            # self.rztd_embedding = self.embedding(name='rztd_embedding',
            #                                      shape=(rztd_number, 32),
            #                                      input=tf.slice(self.input_shuxing_int, [0, 8], [-1, 1]))

        with tf.name_scope('encoder'):
            self.data_cov_1 = self.conv1D(inputs=data_embedding,
                                        kernel_shape=(3,self.embedding_size,self.embedding_size),
                                        strides=1,
                                        kernel_name='conv1d_1',
                                        initializer=self.He_initialzer,
                                        padding='SAME',
                                        activation='relu',
                                        dropuot_rate=self.keep_rate)
            self.data_cov_2 = self.conv1D(inputs=self.data_cov_1,
                                          kernel_shape=(3, self.embedding_size, self.embedding_size),
                                          strides=1,
                                          kernel_name='conv1d_2',
                                          initializer=self.He_initialzer,
                                          padding='SAME',
                                          activation='relu',
                                          dropuot_rate=self.keep_rate)
            # shape=(batch*len_sq/len_word, len_word, embedding_size)
            self.data_word_lavel = tf.reshape(self.data_cov_2, shape=(-1, self.len_word, self.embedding_size))
            # print(self.data_word_lavel)
            # shape=(3200, 15, 600)
            self.word_encoder = MHA.encoder(name='word_encoder',
                                            inputs=self.data_word_lavel,
                                            embedding_size=self.embedding_size,
                                            nb_layers=2,
                                            nb_head=4,
                                            size_per_head=64,
                                            initializer=self.He_initialzer,
                                            training=self.is_trainning,
                                            keep_rate=self.keep_rate,
                                            activition='relu')
            # print(self.word_encoder)
            # shape=(3200, 600)
            self.word_atten = self.attention(inputs=self.word_encoder,
                                             name='word',
                                             initializer=self.He_initialzer,
                                             keep_rate=self.keep_rate,
                                             is_trainning=self.is_trainning,
                                             activation='relu')
            # print(self.word_atten)
            # shape=(128, 25, 600)
            self.data_sq_lavel = tf.reshape(self.word_atten, shape=(-1, self.len_sq, np.shape(self.word_atten)[-1]))
            # shape=(128, 25, 600)
            self.sq_encoder = MHA.encoder(name='sq_encoder',
                                            inputs=self.data_sq_lavel,
                                            embedding_size=self.embedding_size,
                                            nb_layers=2,
                                            nb_head=4,
                                            size_per_head=64,
                                            initializer=self.He_initialzer,
                                            training=self.is_trainning,
                                            keep_rate=self.keep_rate,
                                            activition='relu')
            # print(self.sq_encoder)
            self.sq_atten = self.attention(inputs=self.sq_encoder,
                                           name='sq',
                                           initializer=self.He_initialzer,
                                           keep_rate=self.keep_rate,
                                           is_trainning=self.is_trainning,
                                           activation='relu')
            # print(self.sq_atten)
            # print(self.sq_atten)

        # with tf.name_scope('sx_encoder'):
        #     self.sx_float = self.sx_dense_layer(name='sx_float',inputs=self.input_shuxing_float,output_size=32)
        #     self.xb_int = self.sx_dense_layer(name='xb_int',inputs=self.xb_embedding,output_size=32)
        #     self.jycd_int = self.sx_dense_layer(name='jycd_int',inputs=self.jycd_embedding,output_size=32)
        #     self.qkzm_int = self.sx_dense_layer(name='qkzm_int',inputs=self.qkzm_embedding,output_size=64)
        #     self.xflb_int = self.sx_dense_layer(name='xflb_int',inputs=self.xflb_embedding,output_size=32)
        #     self.jl_int = self.sx_dense_layer(name='jl_int',inputs=self.jl_embedding,output_size=32)
        #     self.je_int = self.sx_dense_layer(name='je_int',inputs=self.je_embedding,output_size=32)
        #     self.qj_int = self.sx_dense_layer(name='qj_int',inputs=self.qj_embedding,output_size=32)
        #     self.hg_int = self.sx_dense_layer(name='hg_int',inputs=self.hg_embedding,output_size=32)
        #     self.rztd_int = self.sx_dense_layer(name='rztd_int',inputs=self.rztd_embedding,output_size=32)


        with tf.name_scope('classifier'):
            self.all_weight = self.sq_atten
            # print(self.all_weight)

            # self.xq_dense_1 = self.fully_conacation(name='xq_1',
            #                                         input=self.all_weight,
            #                                         haddin_size=xq_number * 256,
            #                                         training=self.is_trainning,
            #                                         keep_rate=self.keep_rate,
            #                                         activation='relu')

            self.xq_dense_2 = self.fully_conacation(name='xq_2',
                                                    input=self.all_weight,
                                                    haddin_size=xq_number*6,
                                                    training=self.is_trainning,
                                                    keep_rate=self.keep_rate,
                                                    activation='relu')
            self.xq_dense_3 = self.fully_conacation(name='xq_3',
                                                    input=self.xq_dense_2,
                                                    haddin_size=xq_number,
                                                    training=self.is_trainning,
                                                    keep_rate=self.keep_rate,
                                                    activation='relu')

        with tf.name_scope('classifier_loss'):
            # self.label_rule = tf.reduce_sum(tf.slice(self.input_label,[0,0],[-1,1],name='rule_slice'),axis=-1)
            # self.label_zm = tf.reduce_sum(tf.slice(self.input_label,[0,1],[-1,1],name='zm_slice'),axis=-1)
            # self.label_xq = tf.reduce_sum(tf.slice(self.input_label,[0,2],[-1,1],name='xq_slice'),axis=-1)

            # self.label_xq_float = tf.cast(self.input_label_xq,dtype=tf.float32)
            # self.fj_loss = tf.losses.mean_squared_error(labels=self.input_label_fj,predictions=self.fj_dense_3)
            # tf.add_to_collection('loss', self.fj_loss)
            # tf.summary.scalar('fj_loss', self.fj_loss)
            # print(self.rule_loss)
            # self.hx_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label_hx,
            #                                                                              logits=self.hx_dense_3),axis=-1)
            # tf.add_to_collection('loss', self.hx_loss)
            # tf.summary.scalar('hx_loss', self.hx_loss)
            # print(self.zm_loss)
            self.xq_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label_xq,
                                                                                         logits=self.xq_dense_3),axis=-1)
            tf.add_to_collection('loss', self.xq_loss)
            tf.summary.scalar('xq_loss', self.xq_loss)
            # print(self.xq_loss)
            self.all_loss = tf.add_n(tf.get_collection('loss'))
            tf.summary.scalar('all_loss', self.all_loss)

        with tf.name_scope('accuracy'):
            # self.rule_max_index = tf.argmax(self.rule_dense_2, axis=1)
            # print(self.rule_max_index)
            # self.hx_max_index = tf.argmax(self.hx_dense_2, axis=1)
            # print(self.zm_max_index)
            self.xq_max_index = tf.argmax(self.xq_dense_3, axis=1)
            # print(self.xq_max_index)
            # self.rule_acc = tf.reduce_mean(
            #     tf.cast(tf.equal(self.rule_max_index, self.input_label_rule), dtype=tf.float32), axis=-1)
            # tf.summary.scalar('rule_acc', self.rule_acc)
            # self.hx_acc = tf.reduce_mean(tf.cast(tf.equal(self.hx_max_index, self.input_label_hx), dtype=tf.float32),
            #                              axis=-1)
            # tf.summary.scalar('hx_acc', self.hx_acc)
            self.xq_acc = tf.reduce_mean(tf.cast(tf.equal(self.xq_max_index, self.input_label_xq), dtype=tf.float32),
                                         axis=-1)
            tf.summary.scalar('xq_acc', self.xq_acc)
            self.all_acc = [self.xq_loss]
            # print(self.all_acc)
            update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.all_loss)
            self.train_op = tf.group(self.opt_op, update_ops)

    def Dynamic_LSTM(self, input, keep_rate, training, name,init_hadden_state_fw=None,init_hadden_state_bw=None):
        with tf.variable_scope("lst_" + str(name) + "_1"):
            cell_f_1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            cell_f_1 = tf.nn.rnn_cell.DropoutWrapper(cell_f_1, output_keep_prob=keep_rate)
            cell_b_1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            cell_b_1 = tf.nn.rnn_cell.DropoutWrapper(cell_b_1, output_keep_prob=keep_rate)
            lstm_output_s1, hadden_state = tf.nn.bidirectional_dynamic_rnn(cell_f_1, cell_b_1,
                                                                initial_state_bw=init_hadden_state_bw,
                                                                initial_state_fw=init_hadden_state_fw,
                                                                inputs=input,dtype=tf.float32)
            # state_fw,state_bw = hadden_state
            # print(state_fw)
            lstm_output = tf.concat(lstm_output_s1, axis=-1)
            lstm_output = MHA.layer_norm(lstm_output,scope=name+'_ln_norm')

            return lstm_output

    def fully_conacation(self,name, input, haddin_size,initializer=None, training=True, keep_rate=1.0, activation='relu'):
        with tf.variable_scope(name):
            dense_out = tf.layers.dense(inputs=input, units=haddin_size,kernel_initializer=initializer)
            dense_out = MHA.layer_norm(dense_out,scope=name+'_ln')
            if activation == 'relu':
                dense_relu = tf.nn.relu(dense_out)
                dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
                return dense_relu
            elif activation == 'leaky_relu':
                dense_relu = tf.nn.leaky_relu(dense_out)
                dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
                return dense_relu
            elif activation == 'sigmoid':
                dense_relu = tf.nn.sigmoid(dense_out)
                dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
                return dense_relu
            elif activation == 'tanh':
                dense_relu = tf.nn.tanh(dense_out)
                dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
                return dense_relu
            elif activation == 'None':
                dense_relu = tf.nn.dropout(dense_out, keep_prob=keep_rate)
                return dense_relu

    def assert_regular(self, tensor1, tensor2):
        assert_regular_dot = tf.reduce_sum(tf.multiply(tensor1, tensor2), axis=-1)
        u = tf.sqrt(tf.reduce_sum(tf.square(tensor1), axis=-1))
        v = tf.sqrt(tf.reduce_sum(tf.square(tensor2), axis=-1))
        loss_cos = tf.reduce_mean(assert_regular_dot / tf.multiply(u, v), axis=-1)
        loss_sin = tf.sqrt(1 - tf.square(loss_cos))
        return loss_sin

    def attention(self, inputs, name,initializer=None,keep_rate=None,is_trainning=True,activation='relu'):
        with tf.variable_scope(name + 'att'):
            transfor_data = MHA.Dense(inputs=inputs,
                                      output_size=np.shape(inputs)[-1],
                                      initializer=initializer,
                                      keep_rate=keep_rate,
                                      is_trainning=is_trainning,
                                      activition=activation)
            u_atten = tf.get_variable(name=name + 'att_vocter', shape=(1, np.shape(inputs)[-1]), dtype=tf.float32,initializer=initializer)
            att_weght = tf.reduce_sum(tf.multiply(transfor_data, u_atten), keep_dims=True, axis=2)
            att_weght = tf.nn.softmax(att_weght)
            att_sum = tf.reduce_sum(tf.multiply(inputs, att_weght), axis=1)
        return att_sum

    def conv1D(self, inputs, kernel_shape,strides,kernel_name,initializer=None, padding='VALID',activation='relu', dropuot_rate=1.0):
        '''
        1D卷积，应用于序列数据
        :param inputs: Given an input tensor of shape [batch, in_width, in_channels]
        :param kernel_shape: a filter / kernel tensor of shape [filter_width, in_channels, out_channels]
        :param strides: An `integer`.  The number of entries by which the filter is moved right at each step.
        :param kernel_name: A name for the operation (optional).
        :param initializer:
        :param padding: "SAME" or "VALID"
        :param activation:
        :param dropuot_rate: A "float". the keep rate for layer.
        :return: A `Tensor`.  Has the same type as input.
        '''
        with tf.name_scope('conv1d_'+kernel_name):
            kernel = tf.get_variable(dtype=tf.float32, shape=kernel_shape, name=kernel_name,initializer=initializer)
            conv_output = tf.nn.conv1d(value=inputs, filters=kernel, stride=strides, padding=padding)
            conv_output = MHA.layer_norm(conv_output,scope=kernel_name+'_ln')
            if activation is 'relu':
                conv_output = tf.nn.relu(conv_output)
            elif activation is 'leaky_relu':
                conv_output = tf.nn.leaky_relu(conv_output)
            elif activation is 'sigmoid':
                conv_output = tf.nn.sigmoid(conv_output)
            elif activation is 'tanh':
                conv_output = tf.nn.tanh(conv_output)
            if dropuot_rate is not None:
                conv_output = tf.nn.dropout(conv_output, keep_prob=dropuot_rate)
            return conv_output

    def embedding(self,name,shape,input):
        '''
        :param name:
        :param shape:
        :param input:
        :return:
        '''
        with tf.variable_scope(name+'_scope'):
            embedding_table = tf.get_variable(name=name,
                                                shape=shape,
                                                dtype=tf.float32,
                                                initializer=self.xavier_initialzer)
            input_embedding = tf.nn.embedding_lookup(embedding_table,input)
            output_embedding = tf.reduce_sum(input_embedding,axis=1)

        return output_embedding

    def sx_dense_layer(self,name,inputs,output_size):
        '''
        :param name:
        :param inputs:
        :param output_size:
        :return:
        '''
        with tf.variable_scope(name+'_xs'):
            dense_1 = self.fully_conacation(name=name+'_0',
                                            input=inputs,
                                            haddin_size=output_size*2,
                                            initializer=self.He_initialzer,
                                            training=self.is_trainning,
                                            keep_rate=self.keep_rate,
                                            activation='relu')
            dense_2 = self.fully_conacation(name=name + '_1',
                                            input=dense_1,
                                            haddin_size=output_size,
                                            initializer=self.He_initialzer,
                                            training=self.is_trainning,
                                            keep_rate=self.keep_rate,
                                            activation='relu')
            return dense_2

# if __name__=='__main__':
#     Model = HAN_model()
#     int_op = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(int_op)
#         sess.run(Model.sq_encoder)