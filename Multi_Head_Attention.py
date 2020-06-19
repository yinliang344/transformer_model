#! -*- coding: utf-8 -*-

import tensorflow as tf
from config import *
import numpy as np
'''
inputs是一个形如(batch_size, seq_len, word_size)的张量；
函数返回一个形如(batch_size, seq_len, position_size)的位置张量。
'''


def Position_Embedding(inputs, position_size):
    batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
    position_j = 1. / tf.pow(10000.,
                             2 * tf.range(position_size / 2,
                                          dtype=tf.float32) / position_size)
    position_j = tf.expand_dims(position_j, 0)
    position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
    position_i = tf.expand_dims(position_i, 1)
    position_ij = tf.matmul(position_i, position_j)
    position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
    position_embedding = tf.expand_dims(
        position_ij, 0) + tf.zeros((batch_size, seq_len, position_size))
    return position_embedding


'''
inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
'''


def Mask(inputs, seq_len, mode='mul'):
    if seq_len is None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len, truncature_len), tf.float32)
        for _ in range(len(inputs.shape) - 2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12


'''
普通的全连接
inputs是一个二阶或二阶以上的张量，即形如(batch_size,...,input_size)。
只对最后一个维度做矩阵乘法，即输出一个形如(batch_size,...,ouput_size)的张量。
'''


def Dense(
        inputs,
        output_size,
        initializer=None,
        keep_rate=None,
        is_trainning=True,
        activition='relu',
        bias=False):

    outputs = tf.layers.dense(
        inputs=inputs,
        units=output_size,
        use_bias=bias,
        kernel_initializer=initializer)
    # outputs = tf.layers.batch_normalization(outputs,training=is_trainning)
    if activition is 'relu':
        outputs = tf.nn.relu(outputs)
    elif activition is 'leaky_relu':
        outputs = tf.nn.leaky_relu(outputs)
    elif activition is 'sigmoid':
        outputs = tf.nn.sigmoid(outputs)
    if keep_rate is not None:
        outputs = tf.nn.dropout(outputs, keep_prob=keep_rate)
    return outputs


'''
Multi-Head Attention的实现
'''


def multi_head_attention(
        Q,
        K,
        V,
        nb_head,
        size_per_head,
        initialzer=None,
        keep_rate=None,
        is_trainning=None,
        activation='relu',
        Q_len=None,
        V_len=None):
    # 对Q、K、V分别作线性映射
    query = Dense(inputs=Q,
                  output_size=nb_head * size_per_head,
                  keep_rate=keep_rate,
                  is_trainning=is_trainning,
                  initializer=initialzer,
                  activition=activation,
                  bias=False)
    query = tf.reshape(query, (-1, tf.shape(query)[1], nb_head, size_per_head))
    query = tf.transpose(query, [0, 2, 1, 3])
    key = Dense(inputs=K,
                output_size=nb_head * size_per_head,
                keep_rate=keep_rate,
                is_trainning=is_trainning,
                initializer=initialzer,
                activition=activation,
                bias=False)
    key = tf.reshape(key, (-1, tf.shape(key)[1], nb_head, size_per_head))
    key = tf.transpose(key, [0, 2, 1, 3])
    value = Dense(inputs=V,
                  output_size=nb_head * size_per_head,
                  keep_rate=keep_rate,
                  is_trainning=is_trainning,
                  initializer=initialzer,
                  activition=activation,
                  bias=False)
    value = tf.reshape(value, (-1, tf.shape(value)[1], nb_head, size_per_head))
    value = tf.transpose(value, [0, 2, 1, 3])
    # 计算内积，然后mask，然后softmax
    A = tf.matmul(query, key, transpose_b=True) / tf.sqrt(float(size_per_head))
    A = tf.transpose(A, [0, 3, 2, 1])
    A = Mask(A, V_len, mode='add')
    A = tf.transpose(A, [0, 3, 2, 1])
    A = tf.nn.softmax(A)
    # 输出并mask
    output = tf.matmul(A, value)
    output = tf.transpose(output, [0, 2, 1, 3])
    output = tf.reshape(
        output, (-1, tf.shape(output)[1], nb_head * size_per_head))
    output = Mask(output, Q_len, 'mul')
    return output


def feed_forward(
        inputs,
        initializer=None,
        keep_rate=None,
        is_training=True,
        activition='relu'):
    shapes = int(inputs.shape[-1])
    dense = Dense(inputs=inputs,
                  output_size=shapes * 2,
                  initializer=initializer,
                  keep_rate=keep_rate,
                  is_trainning=is_training,
                  activition=activition)
    dense = Dense(inputs=dense,
                  output_size=shapes,
                  initializer=initializer,
                  keep_rate=keep_rate,
                  is_trainning=is_training,
                  activition=activition)
    return dense


'''
前向传播encoder部分，输入是（batch_size*2, seq_len, word_size）形状，经过multi_head_attention
然后残差连接和norm，再经过两层全连接，最后残差连接和norm
输出是（batch_size, seq_len, word_size）形状
'''


def encoder(
        name,
        inputs,
        embedding_size,
        nb_layers,
        nb_head,
        size_per_head,
        initializer=None,
        Q_len=None,
        V_len=None,
        training=True,
        keep_rate=None,
        activition='relu'):
    with tf.variable_scope(name):
        position = Position_Embedding(
            inputs=inputs, position_size=embedding_size)
        batch = tf.concat([position, inputs], axis=-1)
        for i in range(nb_layers):
            mha_layer = multi_head_attention(
                Q=batch,
                K=batch,
                V=batch,
                nb_head=np.shape(batch)[2] //
                size_per_head,
                size_per_head=size_per_head,
                initialzer=initializer,
                keep_rate=keep_rate,
                is_trainning=training,
                activation=activition,
                Q_len=Q_len,
                V_len=V_len)
            add_layer_1 = tf.add(batch, mha_layer)
            ln_layer_1 = layer_norm(
                x=add_layer_1, scope=name + '_lnlayer_1_' + str(i))
            ff_layer = feed_forward(inputs=ln_layer_1,
                                    initializer=initializer,
                                    keep_rate=keep_rate,
                                    is_training=training,
                                    activition=activition)
            add_layer_2 = tf.add(ln_layer_1, ff_layer)
            batch = layer_norm(
                add_layer_2,
                scope=name +
                '_lnlayer_2_' +
                str(i))

        return batch


def conv2D(
        inputs,
        kernel_shape,
        strides,
        padding,
        kernel_name,
        training,
        activation='relu',
        dropuot_rate=None):
    kernel = tf.get_variable(
        dtype=tf.float32,
        shape=kernel_shape,
        name=kernel_name,
        regularizer=tf.contrib.layers.l2_regularizer(10e-6),
        initializer=tf.contrib.layers.xavier_initializer())
    conv_output = tf.nn.conv2d(
        input=inputs,
        filter=kernel,
        strides=strides,
        padding=padding)
    conv_output = tf.layers.batch_normalization(
        inputs=conv_output, training=training)
    if activation is 'relu':
        conv_output = tf.nn.relu(conv_output)
    elif activation is 'leaky_relu':
        conv_output = tf.nn.leaky_relu(conv_output)
    if dropuot_rate is not None:
        conv_output = tf.nn.dropout(conv_output, keep_prob=dropuot_rate)
    return conv_output


def dense_block(input, nb_layer, strides, keep_rate, training, padding, name):
    x = input
    for i in range(nb_layer):
        conv_out = conv2D(inputs=x,
                          kernel_shape=[3, 3, x.shape[3], 32],
                          strides=strides,
                          padding=padding,
                          dropuot_rate=keep_rate,
                          kernel_name=name + 'kernel' + str(i),
                          training=training)
        x = tf.concat([x, conv_out], axis=-1)
    return x


def transition_block(
        input,
        output_channel,
        keep_rate,
        padding,
        training,
        kernel_name):
    x = conv2D(inputs=input,
               kernel_shape=[1, 1, input.shape[3], output_channel],
               strides=[1, 1, 1, 1],
               padding=padding,
               dropuot_rate=keep_rate,
               kernel_name=kernel_name + 'kernel',
               training=training)
    x_output = tf.nn.max_pool(
        value=x, ksize=[
            1, 2, 2, 1], strides=[
            1, 2, 2, 1], padding=padding)
    return x_output


def Xavier_initializer(node_in, node_out):
    '''
    :param node_in: the number of input size
    :param node_out: the number of output size
    :return: a weight matrix
    '''
    W = tf.div(tf.Variable(np.random.randn(node_in,node_out).astype('float32')),np.sqrt(node_in))
    return W


def He_initializer(node_in, node_out):
    '''
    :param node_in: the number of input size
    :param node_out: the number of output size
    :return: a weight matrix
    '''
    W = tf.div(tf.Variable(np.random.randn(node_in, node_out).astype('float32')), np.sqrt(node_in / 2))
    return W


def layer_norm(x, scope='layer_norm'):
    '''
    :param x: the tensor with shape (batch_size,sq_len,hidden_size) or (batch_size,hidden_size)
    :param scope: the name of layer
    :return:
    '''
    return tf.contrib.layers.layer_norm(
        x, center=True, scale=True, scope=scope)
