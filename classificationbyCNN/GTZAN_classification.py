# coding=utf-8
# __author__ = 'xhm'

import numpy as np
import tensorflow as tf
import os

np.random.seed(0)

PART = 1
DATA_TYPE = "NCHW"
MEAN = [0, -0.000415, -0.000309]
STD = [1, 2.170058, 2.140856]


def z_score(data, mean, std):
    return tf.divide(tf.subtract(data, mean), std)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.00001, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, b, strides=1):
    tmp = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    ans = tf.nn.bias_add(tmp, b)
    return tf.nn.relu(ans)


def load_data(name_list):
    data_length = len(name_list)
    i = 0
    def_data = np.zeros((data_length, 40, 80, 1), dtype=float)
    for list_number in name_list:
        path = '../mfcc_input/' + list_number
        seperation_music = np.load(path)
        def_data[i] = np.reshape(seperation_music, (40, 80, 1))
        i += 1
    return def_data


def random_sampling():
    """
        random the data to get
        train_data, validation_data and testing_data
    """
    music_list = os.listdir('../mfcc_input')
    np.random.shuffle(music_list)
    train_data_name_d = music_list[:5000]
    validation_data_name_d = music_list[5000:7500]
    testing_data_name_d = music_list[7500:10000]
    return train_data_name_d, validation_data_name_d, testing_data_name_d


def generate_one_hot(name_list):
    """
        generate the one-hot list for the data
    """
    onehot_list = np.zeros((len(name_list), 10))
    i = 0
    for list_number in name_list:
        temp_cate = list_number[0:list_number.index('_')]
        onehot_list[i] = dict_onehot[temp_cate]
        i += 1
    return onehot_list


"""
    random_sampling
"""

train_data_name, validation_data_name, testing_data_name = random_sampling()


"""
    generate one_hot for category
"""
category_list = os.listdir('../raw_data/genres')

dict_onehot = {}
dict_i = 0
for cate in category_list:
    onehot = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    onehot[dict_i] = 1
    dict_onehot[cate] = onehot
    dict_i += 1

train_onehot = generate_one_hot(train_data_name)
validation_onehot = generate_one_hot(validation_data_name)
testing_onehot = generate_one_hot(testing_data_name)


x = tf.placeholder("float", [None, 40, 80, 1])
y_ = tf.placeholder("float", [None, 10])

h_z_score = z_score(x, MEAN[0], STD[0])

W_conv1 = weight_variable([10, 12, 1, 30])
b_conv1 = bias_variable([30])
x_image = tf.reshape(h_z_score, [-1, 40, 80, 1])

h_conv1 = conv2d(x_image, W_conv1, b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 20, 1], strides=[1, 1, 20, 1], padding='SAME')


W_fc1 = weight_variable([31*4*30, 200])
b_fc1 = bias_variable([200])

h_pool1_flat = tf.reshape(h_pool1, [-1, 31*4*30])
h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_pool1_flat, W_fc1), b_fc1))

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([200, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.nn.bias_add(tf.matmul(h_fc1_drop, W_fc2), b_fc2))

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
# cross_entropy = tf.reduce_sum(tf.square(tf.subtract(y_, y_conv)))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    f = open('result.txt', 'w')
    step = 1
    head = 0
    one_batch = 50
    while step < 100000:
        batch_x = load_data(train_data_name[head:head+one_batch])
        batch_y = train_onehot[head:head+one_batch]
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.8})
        if step % 100 == 0:
            # v_step = 0
            # while v_step < 250:
            #     loss, acc = sess.run([cross_entropy, accuracy],
            #                          feed_dict={x: load_data(validation_data_name[v_step:v_step+50]),
            #                                     y_: validation_onehot[v_step:v_step+50], keep_prob: 1.0})
            #     v_step += 50
            loss, acc = sess.run([cross_entropy, accuracy],  feed_dict={x: load_data(validation_data_name),
                                                                        y_: validation_onehot, keep_prob: 1.0})
            validation_fo = "Iter" + str(step) + ", Minibatch Loss=" + str(loss) + ", Training Accuracy=" + str(acc)
            print(validation_fo)
            f.write(validation_fo)
        step += 1
        head += one_batch
        if head == 5000:
            head = 0
    print("Optimization Finished")
    f.write("Optimization Finished")
    testing_result = sess.run(accuracy, feed_dict={x: load_data(testing_data_name), y_: testing_onehot, keep_prob: 1.0})
    print("Testing Accuracy:" + str(testing_result))
    f.write("Testing Accuracy:" + str(testing_result))
    f.close()
# x_image = tf.reshape(h_z_score, [-1, PART, 513, 100])
#
# h_conv1 = conv2d(x_image, W_conv1, b_conv1)
# h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 4, 4], strides=[1, 1, 2, 2], padding='VALID', data_format=DATA_TYPE)
#
# W_conv2 = weight_variable([8, 8, 16, 16])
# b_conv2 = bias_variable([16])
#
# h_conv2 = conv2d(h_pool1, W_conv2, b_conv2, 2)
# h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 1, 4, 4], strides=[1, 1, 2, 2], padding='VALID', data_format=DATA_TYPE)
#
# W_fc1 = weight_variable([11*9*16, 50])
# b_fc1 = bias_variable([50])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 11*9*16])
# h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_pool2_flat, W_fc1), b_fc1))
#
# # keep_prob = tf.placeholder("float")
# # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# W_fc2 = weight_variable([50, 10])
# b_fc2 = bias_variable([10])
#
# y_conv = tf.nn.softmax(tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2))
#
# cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# train_step = tf.train.AdamOptimizer(0.00001).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     f = open('result.txt', 'w')
#     step = 1
#     head = 0
#     one_batch = 1
#     while step < 100000:
#         batch_x = load_data(train_data_name[head:head+one_batch])
#         batch_y = train_onehot[head:head+one_batch]
#         sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
#         if step % 100 == 0:
#             # v_step = 0
#             # while v_step < 250:
#             #     loss, acc = sess.run([cross_entropy, accuracy],
#             #                          feed_dict={x: load_data(validation_data_name[v_step:v_step+50]),
#             #                                     y_: validation_onehot[v_step:v_step+50], keep_prob: 1.0})
#             #     v_step += 50
#             loss, acc = sess.run([cross_entropy, accuracy],  feed_dict={x: load_data(validation_data_name),
#                                                                         y_: validation_onehot})
#             validation_fo = "Iter" + str(step) + ", Minibatch Loss=" + str(loss) + ", Training Accuracy=" + str(acc)
#             print(validation_fo)
#             f.write(validation_fo)
#         step += 1
#         head += one_batch
#         if head == 500:
#             head = 0
#     print("Optimization Finished")
#     f.write("Optimization Finished")
#     testing_result = sess.run(accuracy, feed_dict={x: load_data(testing_data_name), y_: testing_onehot})
#     print("Testing Accuracy:" + str(testing_result))
#     f.write("Testing Accuracy:" + str(testing_result))
#     f.close()
