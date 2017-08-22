# coding=utf-8
# __author__ = 'xhm'

import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

y_true = []
y_pred = []

file = open('../outputListFile.txt', 'r')
lines = file.readlines()
for line in lines:
    label_split = line.split('\t')
    y_pred.append(label_split[-1].strip())
    temp = label_split[0].split('/')
    temp1 = temp[-1].split('.')
    y_true.append(temp1[0].strip())
    # y_true.append(label_split[5].strip(','))
    # y_pred.append(label_split[8])
file.close()
tick_marks = np.array(range(len(labels))) + 0.5


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
print(cm)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)
# set the fontsize of label.
# for label in plt.gca().xaxis.get_ticklabels():
#    label.set_fontsize(8)
# text portion
ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.50:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=7, va='center', ha='center')
    else:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
plt.show()
