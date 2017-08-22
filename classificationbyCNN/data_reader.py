# coding=utf-8
# __author__ = 'xhm'

import numpy as np
import tensorflow as tf
import os

np.random.seed(0)


def random_sampling():
    """
        random the data to get
        train_data, validation_data and testing_data
    """
    music_list = os.listdir('../seperation_input')
    np.random.shuffle(music_list)
    train_data_name_d = music_list[:500]
    validation_data_name_d = music_list[500:750]
    testing_data_name_d = music_list[750:1000]
    print(train_data_name_d)
    print(validation_data_name_d)
    print(testing_data_name_d)
    return train_data_name_d, validation_data_name_d, testing_data_name_d


train_data_name, validation_data_name, testing_data_name = random_sampling()

tf.train.string_input_producer(train_data_name)

