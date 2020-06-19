from multi_extra.transformer_HAN_multi_extra import HAN_model
from DataProcess import DataProcess
import numpy as np
import config as cf
import tensorflow as tf
from tqdm import trange
from validation import validation

DP = DataProcess()
test_x_all, test_y_input_float, test_y_input_int, test_y_output_all= DP.read_file_test('./data_multi')
print('1、load data完成')
Model = HAN_model()
print('2、构造模型完成')
# # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
# opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Model.all_loss)
saver = tf.train.Saver(max_to_keep=10)
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    # merged = tf.summary.merge_all()
    # writer_train = tf.summary.FileWriter("logs_multi_extra/train", sess.graph)
    # writer_test = tf.summary.FileWriter("logs_multi_extra/test")
    # sess.run(init_op)
    ckpt_multi_extra = tf.train.get_checkpoint_state('./multi_extra/ckpt_multi_extra/')
    saver.restore(sess, save_path=ckpt_multi_extra.model_checkpoint_path)
    print('3、初始化完成')
    print('4、开始测试')
    xq_pre_all = []
    xq_true_label = []
    ft_pre_all = []
    ft_true_label = []
    zm_pre_all = []
    zm_true_label = []
    epoch_number = int(len(test_y_output_all)//cf.batch_size)
    batch_test= DP.batch_generator(all_data=test_x_all,
                                   all_label=test_y_output_all,
                                   all_input_float=test_y_input_float,
                                   all_input_int=test_y_input_int,
                                   batch_size=cf.batch_size,
                                   shuffle=True)
    for k in trange(epoch_number):
        batch_data_test,batch_input_test_float,batch_input_test_int, batch_label_test = next(batch_test)
        feed_dic_test = {Model.input_data: batch_data_test,
                         Model.input_shuxing_float:batch_input_test_float,
                         Model.input_shuxing_int:batch_input_test_int,
                         Model.input_label: batch_label_test,
                         Model.keep_rate: 1.0,
                         Model.is_trainning: False}
        ft,xq,zm = sess.run([Model.ft_max_index,Model.xq_max_index,Model.zm_max_index],feed_dict=feed_dic_test)
        ft_pre_all += list(ft)
        ft_true_label += list(np.array(batch_label_test)[:,0:1])
        xq_pre_all += list(xq)
        xq_true_label += list(np.array(batch_label_test)[:,1:2])
        zm_pre_all += list(zm)
        zm_true_label += list(np.array(batch_label_test)[:, 2:])
    Valid_ft = validation(label=ft_true_label, prediction=ft_pre_all)
    print('ft_acc: ', Valid_ft.acc())
    print("ft_f1: ", Valid_ft.f1())
    print("ft_MP: ", Valid_ft.MP())
    print("ft_MR: ", Valid_ft.MR())
    Valid_xq = validation(label=xq_true_label, prediction=xq_pre_all)
    print('xq_acc: ', Valid_xq.acc())
    print("xq_f1: ", Valid_xq.f1())
    print("xq_MP: ", Valid_xq.MP())
    print("xq_MR: ", Valid_xq.MR())
    Valid_zm = validation(label=zm_true_label, prediction=zm_pre_all)
    print('zm_acc: ', Valid_zm.acc())
    print("zm_f1: ", Valid_zm.f1())
    print("zm_MP: ", Valid_zm.MP())
    print("zm_MR: ", Valid_zm.MR())