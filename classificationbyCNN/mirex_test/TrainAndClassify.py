# coding=utf-8
# __author__ = 'xhm'
import tensorflow as tf
import numpy as np
import datetime
import sys

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


def get_para():
    return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]


def separate_train_line(line):
    temp = line.split('\t')
    music_split = temp[0].split('/')
    return music_split[-1].strip(), temp[-1].strip()


def get_genre_list():
    file = open(train_path, 'r')
    lines = file.readlines()
    temp_list = []
    for line in lines:
        music_train_name, music_genre = separate_train_line(line)
        if music_genre not in temp_list:
            temp_list.append(music_genre)
    file.close()
    return temp_list, lines


def one_hot():
    dict_onehot = []
    dict_i = 0
    for cate in genre_list:
        onehot = np.zeros(genre_number)
        onehot[dict_i] = 1
        dict_onehot.append(onehot)
        dict_i += 1
    return dict_onehot


def get_random_music_data(path, length=every_merge_size):
    all_music = np.load(path)
    start = np.random.random_integers(0, all_music.shape[1]-length-1)
    cut_music = all_music[:, start:start+length]
    return cut_music


def get_random_train_data(num):
    """
    get random train data from random raw data
    :return:
    one batch train data and their one-hot encoding
    """
    random_train_data = np.zeros((num, 40, every_merge_size*music_merge, 1), dtype=float)
    random_one_hot = np.zeros((num, genre_number), dtype=float)

    for i in range(num):
        random_line = np.random.random_integers(0, len(train_path_lines)-1)
        music_train_name, music_genre = separate_train_line(train_path_lines[random_line])
        random_one_hot[i] = cate_onehot[genre_list.index(music_genre)]
        music_data_path = folder_path + "/" + music_train_name + ".npy"
        music_data = get_random_music_data(music_data_path)
        for j in range(music_merge-1):
            data_part = get_random_music_data(music_data_path)
            music_data = np.concatenate((music_data, data_part), axis=1)

        random_train_data[i] = np.reshape(music_data, (40, every_merge_size*music_merge, 1))

    return random_train_data, random_one_hot


def get_max_prediction(genre_predict_list):
    count_list = np.zeros(genre_number, dtype=int)
    max_num = 0
    arg_max = 0
    for i in range(genre_predict_list.shape[0]):
        count_list[genre_predict_list[i]] += 1
        if count_list[genre_predict_list[i]] > max_num:
            max_num = count_list[genre_predict_list[i]]
            arg_max = genre_predict_list[i]
    return arg_max


def testing():
    file = open(test_path, 'r')
    out_file = open(output_path, 'w')
    lines = file.readlines()
    for line in lines:
        music_split = line.split('/')
        music_test_name = music_split[-1].strip()
        music_data_path = folder_path + "/" + music_test_name + ".npy"

        # test_temp = music_test_name.split('.')
        # test_genre = test_temp[0].strip()

        testing_batch = 15
        random_testing_data = np.zeros((testing_batch, 40, every_merge_size * music_merge, 1), dtype=float)

        # testing_one_hot = np.zeros((testing_batch, genre_number), dtype=float)

        for t in range(testing_batch):
            music_data = get_random_music_data(music_data_path)
            for j in range(music_merge - 1):
                data_part = get_random_music_data(music_data_path)
                music_data = np.concatenate((music_data, data_part), axis=1)
            random_testing_data[t] = np.reshape(music_data, (40, every_merge_size * music_merge, 1))
            # testing_one_hot[t] = cate_onehot[genre_list.index(test_genre)]
        # loss, genre_prediction_list = sess.run([final_losses, every_genre_prediction],
        #                                        feed_dict={x: random_testing_data,
        #                                                   y_: testing_one_hot, keep_prob: 1.0})#

        genre_prediction_list = sess.run(every_genre_prediction, feed_dict={x: random_testing_data, keep_prob: 1.0})
        final_prediction = get_max_prediction(genre_prediction_list)
        # if final_prediction == genre_list.index(test_genre):
        #     right += 1
        # final_loss += loss
        out_file.write(line.strip() + "\t" + genre_list[final_prediction] + "\n")
    # final_accuracy = round(right / (len(lines)), 3)
    # final_loss = round(final_loss / len(lines), 3)
    # print("test = " + str(final_accuracy) + ", loss = " + str(final_loss))
    file.close()
    out_file.close()


# def acc_show():
#     right = 0
#     final_loss = 0
#     file = open(train_path, 'r')
#     lines = file.readlines()
#     for line in lines:
#         temp = line.split('\t')
#         music_split = temp[0].split('/')
#         music_test_name = music_split[-1].strip()
#         music_data_path = folder_path + "/" + music_test_name + ".npy"
#         testing_batch = 15
#         random_testing_data = np.zeros((testing_batch, 40, every_merge_size * music_merge, 1), dtype=float)
#         testing_one_hot = np.zeros((testing_batch, 10), dtype=float)
#         for t in range(testing_batch):
#             music_data = get_random_music_data(music_data_path)
#             for j in range(music_merge - 1):
#                 data_part = get_random_music_data(music_data_path)
#                 music_data = np.concatenate((music_data, data_part), axis=1)
#             random_testing_data[t] = np.reshape(music_data, (40, every_merge_size * music_merge, 1))
#             testing_one_hot[t] = cate_onehot[genre_list.index(temp[-1].strip())]
#         loss, genre_prediction_list = sess.run([final_losses, every_genre_prediction],
#                                                feed_dict={x: random_testing_data,
#                                                           y_: testing_one_hot, keep_prob: 1.0})
#         final_prediction = get_max_prediction(genre_prediction_list)
#         if final_prediction == genre_list.index(temp[-1].strip()):
#             right += 1
#         final_loss += loss
#     final_accuracy = round(right / (len(lines)), 3)
#     final_loss = round(final_loss / len(lines), 3)
#     validation_fo = "Step" + str(step) + ", Loss=" + str(final_loss) + ", Training Accuracy=" + \
#                     str(final_accuracy)
#     print(validation_fo)
#     file.close()

folder_path, train_path, test_path, output_path = get_para()
genre_list, train_path_lines = get_genre_list()
genre_number = len(genre_list)
cate_onehot = one_hot()


'''
    cnn training
'''
x = tf.placeholder("float", [None, 40, every_merge_size*music_merge, 1])
y_ = tf.placeholder("float", [None, genre_number])

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


W_fc3 = weight_variable([200, genre_number])
b_fc3 = bias_variable([genre_number])

y_conv = tf.nn.softmax(tf.nn.bias_add(tf.matmul(h_fc2_drop, W_fc3), b_fc3))

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))

tf.add_to_collection("losses", cross_entropy)

final_losses = tf.add_n(tf.get_collection("losses"))
lr = tf.train.exponential_decay(learning_rate=0.0002, global_step=100000, decay_steps=2000,
                                decay_rate=0.96, staircase=True)
train_step = tf.train.AdamOptimizer(lr).minimize(final_losses)

every_genre_prediction = tf.argmax(y_conv, 1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    batch_size = 50
    print("Training Start")
    starttime = datetime.datetime.now()

    while step < 100000:
        step += 1
        batch_x, batch_y = get_random_train_data(batch_size)
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.6})

        if step % 100 == 0:

            endtime = datetime.datetime.now()
            print("Iter" + str(step) + "....done, Runtime = " + str((endtime - starttime).seconds) + "s")

        # if step % 1000 == 0:
        #     acc_show()
        #     testing()

    print("Training Done")
    print("Testing Started")
    testing()
    print("Testing Done")
    print("All Done!")
