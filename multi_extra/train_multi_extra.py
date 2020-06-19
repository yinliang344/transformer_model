from multi_extra.transformer_HAN_multi_extra import HAN_model
from DataProcess import DataProcess
import numpy as np
import config as cf
import tensorflow as tf
from tqdm import trange
from validation import validation
valid = validation()
DP = DataProcess()
train_x_all, train_y_input_float, train_y_input_int, train_y_output_all, test_x_all, test_y_input_float, test_y_input_int, test_y_output_all= DP.read_file_multi(r'D:\Project\transformer_model\data_multi')
print('1、load data完成')
Model = HAN_model()
print('2、构造模型完成')
# # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
# opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Model.all_loss)
saver = tf.train.Saver(max_to_keep=10)
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer_train = tf.summary.FileWriter("./multi_extra/logs_multi_extra/train", sess.graph)
    writer_test = tf.summary.FileWriter("./multi_extra/logs_multi_extra/test")
    sess.run(init_op)
    # ckpt_single_extra = tf.train.get_checkpoint_state('./ckpt_single_extra/')
    # saver.restore(sess, save_path=ckpt_single_extra.model_checkpoint_path)
    print('3、初始化完成')
    print('4、开始训练')
    rule_all_acc = 0
    zm_all_acc = 0
    xq_all_acc = 0
    smoothing = [0,0,0,0,0,0,0,0,0,0]
    smoothing_xq = 0
    epoch_number = int(len(train_y_output_all)//cf.batch_size)
    batch_train = DP.batch_generator(all_data=train_x_all,
                                     all_label=train_y_output_all,
                                     all_input_float=train_y_input_float,
                                     all_input_int=train_y_input_int,
                                     batch_size=cf.batch_size,
                                     shuffle=True)
    batch_test= DP.batch_generator(all_data=test_x_all,
                                   all_label=test_y_output_all,
                                   all_input_float=test_y_input_float,
                                   all_input_int=test_y_input_int,
                                   batch_size=cf.batch_size,
                                   shuffle=True)
    j=0
    for k in range(cf.epoch_size):
        for i in trange(epoch_number,desc='epoch-'+str(k+1)+'/'+str(cf.epoch_size)+':'):
            # print(batch_data)
            batch_data_train,batch_input_train_float,batch_input_train_int,batch_label_train = next(batch_train)

            # print(batch_label_xq)
            # print(batch_label_zm)
            feed_dic_train = {Model.input_data: np.array(batch_data_train),
                              Model.input_shuxing_float:batch_input_train_float,
                              Model.input_shuxing_int:batch_input_train_int,
                              Model.input_label:batch_label_train,
                              Model.keep_rate: 0.8,
                              Model.is_trainning: True}
            _,rs_train = sess.run([Model.train_op,merged],feed_dict=feed_dic_train)
            j+=1
            writer_train.add_summary(rs_train, j)
            if (j)%100==0:
                batch_data_test,batch_input_test_float,batch_input_test_int, batch_label_test = next(batch_test)
                feed_dic_test = {Model.input_data: batch_data_test,
                                 Model.input_shuxing_float:batch_input_test_float,
                                 Model.input_shuxing_int:batch_input_test_int,
                                 Model.input_label: batch_label_test,
                                 Model.keep_rate: 1.0,
                                 Model.is_trainning: False}
                xq_acc,rs_test = sess.run([Model.xq_acc,merged],feed_dict=feed_dic_test)
                writer_test.add_summary(rs_test,j)
                smoothing = smoothing[1:]+[xq_acc]
                smoothing_T = valid.smooth(List=smoothing)
                if smoothing_T>=smoothing_xq:
                    smoothing_xq = smoothing_T
                    saver.save(sess, save_path='./ckpt_multi_extra/model.ckpt', global_step=j)