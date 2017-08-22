# coding=utf-8
# __author__ = 'xhm'
import sys
import librosa
import librosa.display
import numpy as np


def get_para():
    return sys.argv[1], sys.argv[2]


def get_music_name(name_path):
    split = name_path.split('/')
    music_name = split[-1].strip()
    return music_name


def mel_data(folder_path, txt_path):
    file = open(txt_path, 'r')
    lines = file.readlines()
    for line in lines:
        y, sr = librosa.load(path=line.strip())
        s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=3072, hop_length=768, fmax=8000)
        re = librosa.power_to_db(s)
        music_name = get_music_name(line)
        save_path = folder_path + '/' + music_name
        np.save(save_path, re)
        print("Processing " + line.strip() + "...... done")
    file.close()


def main(argv=None):
    folder_path, txt_path = get_para()
    mel_data(folder_path, txt_path)
    print("Preprocessing and Extracting Features........All done")
if __name__ == '__main__':
    main()
