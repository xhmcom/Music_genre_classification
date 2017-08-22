# coding=utf-8
# __author__ = 'xhm'

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


GENRE_LIST = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


def getmel(path):
    y, sr = librosa.load(path=path, offset=0)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=3072, hop_length=768, fmax=8000)
    re = librosa.power_to_db(S)
    return re


def data_test():
    for i in range(100):
        print(np.random.random_integers(0, 1))


def get_pic():

    # path = "raw_data/genres/blues/blues.00000.au"
    # y, sr = librosa.load(path=path, offset=0)
    #
    # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=3072, hop_length=768, fmax=8000)
    # re = librosa.power_to_db(S)

    for genre in range(10):
        fig = plt.figure(genre, figsize=(30, 20))
        fig.suptitle(GENRE_LIST[genre])
        path_head = "raw_data/genres/" + GENRE_LIST[genre] + "/" + GENRE_LIST[genre] + ".0000"
        for i in range(10):
            path = path_head + str(i) + ".au"
            plt.subplot(10, 1, i+1)
            re = getmel(path)
            if i == 9:
                librosa.display.specshow(re, x_axis='time', y_axis='mel')
            else:
                librosa.display.specshow(re, y_axis='mel')
        # plt.show()
        plt.savefig(GENRE_LIST[genre])
        print(genre)


def mfcc_every_music(category_number, music_number, music_offset):
    music_path = 'raw_data/genres/' + category_number + '/' + music_number

    y, sr = librosa.load(path=music_path, offset=music_offset, duration=2.78)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=3072, hop_length=768, fmax=8000)
    return S


def stft_every_music(category_number, music_number):
    music_path = 'raw_data/genres/' + category_number + '/' + music_number

    y, sr = librosa.load(path=music_path)
    return librosa.stft(y, n_fft=1024, win_length=1024, dtype=float)


def mfcc_data():
    category_list = os.listdir('raw_data/genres')
    for category_number in category_list:
        data_number = 0
        music_list = os.listdir('raw_data/genres/%s' % category_number)
        for music_number in music_list:
            music_path = 'raw_data/genres/' + category_number + '/' + music_number
            y, sr = librosa.load(path=music_path)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=3072, hop_length=768, fmax=8000)
            re = librosa.power_to_db(S)
            save_path = 'mel_raw_data/' + category_number + '_' + str(data_number) + '.npy'
            np.save(save_path, re)
            data_number += 1


def set_mirex_data():
    category_list = os.listdir('G:/PycharmProjects/musicclassification/raw_data/genres')
    f = open("Feature_extraction_list_file.txt", 'w')
    for category_number in category_list:
        music_list = os.listdir('raw_data/genres/%s' % category_number)
        for music_number in music_list:
            music_path = 'G:/PycharmProjects/musicclassification/raw_data/genres/' + category_number + '/' + music_number
            f.write(music_path + '\n')
    f.close()


def set_traintest_file():
    path_list = os.listdir('G:/PycharmProjects/musicclassification/classificationbyCNN/mirex_test/folder')
    train_set = []
    test_set = []
    for i in range(10):
        temp = path_list[i * 100:i * 100 + 100]
        np.random.shuffle(temp)
        temptrain = temp[0:75]
        temptest = temp[75:100]
        train_set.extend(temptrain)
        test_set.extend(temptest)
    np.random.shuffle(train_set)
    np.random.shuffle(test_set)
    file = open("trainListFile.txt", 'w')
    for i in range(len(train_set)):
        temp = train_set[i].split('.')

        file.write("G:/PycharmProjects/musicclassification/classificationbyCNN/mirex_test/folder/"
                   + train_set[i].replace(".npy", "") + '\t' + temp[0].strip() + '\n')
    file.close()
    file = open("testListFile.txt", 'w')
    for i in range(len(test_set)):
        file.write("G:/PycharmProjects/musicclassification/classificationbyCNN/mirex_test/folder/"
                   + test_set[i].replace(".npy", "") + '\n')
    file.close()


def read_data_set():
    category_list = os.listdir('raw_data/genres')

    # music_input = pd.DataFrame(columns=[str(x) for x in range(10)])
    # every_music_input = pd.Series([1,2,3,4,5,6,7,8,9,10], index=[str(x) for x in range(10)])
    # music_input = music_input.append(every_music_input, ignore_index=True)
    for category_number in category_list:
        music_list = os.listdir('raw_data/genres/%s' % category_number)
        data_number = 0
        for music_number in music_list:
            print(music_number)
            seperation_number = 10
            for i in range(seperation_number):
                every_music_input = mfcc_every_music(category_number, music_number, i*2.78)
                save_path = 'mfcc_input/' + category_number + '_' + str(data_number) + '.npy'
                np.save(save_path, every_music_input)
                data_number += 1


def data_set_seperation():
    music_list = os.listdir('stft_input')

    for music_number in music_list:
        print(music_number)
        temp_input = np.load('stft_input/%s' % music_number)
        temp_np = np.zeros((16, 513, 100), dtype=float)
        temp_input = temp_input[:, :1600]
        for i in range(16):
            temp_i = temp_input[:, i*100:i*100+100]
            temp_np[i] = temp_i
        save_path = 'seperation_input/' + music_number
        np.save(save_path, temp_np)


def data_set_label():
    category_list = os.listdir('raw_data/genres')

    dict_onehot = {}
    dict_i = 0
    for cate in category_list:
        onehot = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        onehot[dict_i] = 1
        dict_onehot[cate] = onehot
        dict_i += 1

    music_list = os.listdir('seperation_input')

    for music_number in music_list:
        print(music_number)
        temp_cate = music_number[0:music_number.index('.')]
        save_path = 'seperation_label/' + music_number
        np.save(save_path, dict_onehot[temp_cate])
        print(dict_onehot[temp_cate])


def result_analysis():
    path = "resultgenre.txt"
    f = open(path)
    line = f.readline()
    anal = np.zeros((10, 10), dtype=int)

    while line:
        line = f.readline()

    f.close()


def main(argv=None):
    """
    fist use read_data_set(),
    then use data_set_seperation().
    data_test() just for some test.
    """
    # read_data_set()
    data_test()
    # set_mirex_data()
    # set_traintest_file()
    # mfcc_data()
    # data_set_seperation()
    # data_set_label()

if __name__ == '__main__':
    main()



