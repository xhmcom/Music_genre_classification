# coding=utf-8
# __author__ = 'xhm'
import numpy as np
import os

np.random.seed(0)
GENRE_LIST = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
DATA_SOURCE = '../mel_raw_data/'
music_merge = 2
every_merge_size = 120


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
    one batch train data
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


for step in range(1000):
    batch_x, batch_y = get_random_train_data(50)
    save_path_x = '../train_data_set/train_data/data_' + str(step).zfill(5) + '.npy'
    np.save(save_path_x, batch_x)
    save_path_y = '../train_data_set/train_data_label/label_' + str(step).zfill(5) + '.npy'
    np.save(save_path_y, batch_y)
