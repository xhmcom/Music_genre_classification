# coding=utf-8
# __author__ = 'xhm'

import numpy as np
import tensorflow as tf
import os
import datetime

np.random.seed(0)
GENRE_LIST = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
DATA_SOURCE = '../mel_raw_data/'
music_merge = 2
every_merge_size = 120


def weight_variable(shape):
    var = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.002)(var))
    return var


def bias_variable(shape):
    initial = tf.constant(0.00001, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, b, strides=1):
    tmp = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    ans = tf.nn.bias_add(tmp, b)
    return tf.nn.elu(ans)


def one_hot():
    category_list = os.listdir('../raw_data/genres')

    dict_onehot = []
    dict_i = 0
    for cate in category_list:
        onehot = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        onehot[dict_i] = 1
        dict_onehot.append(onehot)
        dict_i += 1
    return dict_onehot


def get_max_prediction(genre_list):
    count_list = np.zeros(10, dtype=int)
    max_num = 0
    arg_max = 0
    for i in range(genre_list.shape[0]):
        count_list[genre_list[i]] += 1
        if count_list[genre_list[i]] > max_num:
            max_num = count_list[genre_list[i]]
            arg_max = genre_list[i]
    return arg_max


def random_raw_data(path, cate=10, each_number=100):
    """
    get random raw data from mfcc_raw_data
    each genre is independent and randomized
    """
    music_list = os.listdir(path)
    random_list = []
    for i in range(cate):
        random_list.append(music_list[i*each_number:(i*each_number+each_number)])
        np.random.shuffle(random_list[i])
    return random_list


def get_random_music_data(path, length=every_merge_size):
    all_music = np.load(path)
    start = np.random.random_integers(0, all_music.shape[1]-length-1)
    cut_music = all_music[:, start:start+length]
    return cut_music


def get_random_music_name(genre_number):
    """
    know which genre
    get random music name from genre
    :return:
    path
    """
    music_number = np.random.random_integers(0, len(train_rawdata_name[genre_number])-1)
    music_name = train_rawdata_name[genre_number][music_number]
    path = DATA_SOURCE + music_name
    return path


def get_random_train_data(num):
    """
    get random train data from random raw data
    :return:
    one batch train data and their one-hot encoding
    """
    random_train_data = np.zeros((num, 40, every_merge_size*music_merge, 1), dtype=float)
    random_one_hot = np.zeros((num, 10), dtype=float)

    for i in range(num):
        genre = np.random.random_integers(0, 9)
        random_one_hot[i] = cate_onehot[genre]
        path = get_random_music_name(genre)
        music_data = get_random_music_data(path)
        for j in range(music_merge-1):
            # path = get_random_music_name(genre)
            data_part = get_random_music_data(path)
            music_data = np.concatenate((music_data, data_part), axis=1)

        random_train_data[i] = np.reshape(music_data, (40, every_merge_size*music_merge, 1))

    return random_train_data, random_one_hot


random_raw = random_raw_data(DATA_SOURCE)
cate_onehot = one_hot()

train_rawdata_name = []
testing_rawdata_name = []
for i in range(10):
    train_rawdata_name.append(random_raw[i][:75])
    testing_rawdata_name.append(random_raw[i][75:100])

x = tf.placeholder("float", [None, 40, every_merge_size*music_merge, 1])
y_ = tf.placeholder("float", [None, 10])

W_conv1 = weight_variable([10, 12, 1, 30])
b_conv1 = bias_variable([30])
x_image = tf.reshape(x, [-1, 40, every_merge_size*music_merge, 1])

h_conv1 = conv2d(x_image, W_conv1, b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 8, 1], strides=[1, 2, 8, 1], padding='SAME')

W_conv2 = weight_variable([3, 4, 30, 30])
b_conv2 = bias_variable([30])

h_conv2 = conv2d(h_pool1, W_conv2, b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv3 = weight_variable([2, 5, 30, 60])
b_conv3 = bias_variable([60])

h_conv3 = conv2d(h_pool2, W_conv3, b_conv3)

temp_time_axis = ((every_merge_size*music_merge-12+1)//8+1-4+1)//2-5+1

W_fc1 = weight_variable([6*temp_time_axis*60, 250])
b_fc1 = bias_variable([250])

h_pool2_flat = tf.reshape(h_conv3, [-1, 6*temp_time_axis*60])
h_fc1 = tf.nn.elu(tf.nn.bias_add(tf.matmul(h_pool2_flat, W_fc1), b_fc1))

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([250, 200])
b_fc2 = bias_variable([200])

h_fc2 = tf.nn.elu(tf.nn.bias_add(tf.matmul(h_fc1_drop, W_fc2), b_fc2))
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


W_fc3 = weight_variable([200, 10])
b_fc3 = bias_variable([10])

y_conv = tf.nn.softmax(tf.nn.bias_add(tf.matmul(h_fc2_drop, W_fc3), b_fc3))

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
# cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# cross_entropy = tf.reduce_sum(tf.square(tf.subtract(y_, y_conv)))
tf.add_to_collection("losses", cross_entropy)

final_losses = tf.add_n(tf.get_collection("losses"))
lr = tf.train.exponential_decay(learning_rate=0.0002, global_step=100000, decay_steps=2000,
                                decay_rate=0.96, staircase=True)
train_step = tf.train.AdamOptimizer(lr).minimize(final_losses)

every_genre_prediction = tf.argmax(y_conv, 1)

init = tf.global_variables_initializer()


def acc_show(data_source, testing=True, result=False):
    right = 0
    final_loss = 0
    if testing:
        music_range = 25
    else:
        music_range = 75
    for genre_number in range(10):
        for music_number in range(music_range):
            music_name = data_source[genre_number][music_number]
            music_path = DATA_SOURCE + music_name
            testing_batch = 15
            random_testing_data = np.zeros((testing_batch, 40, every_merge_size * music_merge, 1), dtype=float)
            testing_one_hot = np.zeros((testing_batch, 10), dtype=float)
            for t in range(testing_batch):
                music_data = get_random_music_data(music_path)
                for j in range(music_merge-1):
                    temp_part = get_random_music_data(music_path)
                    music_data = np.concatenate((music_data, temp_part), axis=1)
                random_testing_data[t] = np.reshape(music_data, (40, every_merge_size * music_merge, 1))
                testing_one_hot[t] = cate_onehot[genre_number]
            loss, genre_prediction_list = sess.run([final_losses, every_genre_prediction],
                                                   feed_dict={x: random_testing_data,
                                                              y_: testing_one_hot, keep_prob: 1.0})
            final_prediction = get_max_prediction(genre_prediction_list)
            if final_prediction == genre_number:
                right += 1
            final_loss += loss
            if result:
                f.write("music_name = " + music_name + ", label = " + GENRE_LIST[genre_number] +
                        ", prediction = " + GENRE_LIST[final_prediction] + '\n')
    final_accuracy = round(right / (music_range*10), 3)
    final_loss = round(final_loss / (music_range*10), 3)
    if testing:
        validation_fo = "Testing: " + "Step" + str(step) + ", Loss=" + str(final_loss) + ", Testing Accuracy=" + \
                        str(final_accuracy)
        f.write(validation_fo + '\n')
        print(validation_fo, end='')
        return final_accuracy
    else:
        validation_fo = "Step" + str(step) + ", Loss=" + str(final_loss) + ", Training Accuracy=" + \
                        str(final_accuracy)
        f.write(validation_fo + '\n')
        print(validation_fo, end='')


with tf.Session() as sess:
    sess.run(init)
    f = open('result.txt', 'w')
    step = 0
    batch_size = 50
    best_acc = 0
    early_stop = 0
    starttime = datetime.datetime.now()
    while step < 100000:
        step += 1
        batch_x, batch_y = get_random_train_data(batch_size)

        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.6})

        if step % 100 == 0:
            if step % 1000 == 0:
                now_acc = acc_show(testing_rawdata_name)
                # if now_acc > best_acc:
                #     best_acc = now_acc
                #     early_stop = 0
                # else:
                #     early_stop += 1
                #     if early_stop == 20:
                #         print("early stop")
                #         break
            else:
                acc_show(train_rawdata_name, testing=False)
            endtime = datetime.datetime.now()
            print(", Runtime = " + str((endtime - starttime).seconds) + "s")

    print("Optimization Finished")
    f.write("Optimization Finished"+'\n')
    acc_show(testing_rawdata_name, result=True)
    f.close()
