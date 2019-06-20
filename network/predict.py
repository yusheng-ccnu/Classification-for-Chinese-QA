
from keras.models import load_model
from util.transform_martrix import transform_to_matrix_char, transform_to_matrix
import numpy as np
from sklearn.metrics import classification_report

model = load_model('../models/best_weights_235.hdf5')

label_index = {}
batch_size = 32


def load_data(path):
    x_ = []
    y_ = []
    with open(path, 'r', encoding='utf-8') as test:
        for line in test:
            terms = line.strip('\n').split('\t')
            label = terms[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            x_.append(terms[1].split(' '))
            y_.append(label_index[label])
    return x_, y_


def load_sen(path):
    x_ = []
    with open(path, 'r', encoding='utf-8') as test:
        for line in test:
            x_.append(line)
    return x_


def get_label(i):
    return [k for k, v in label_index.items() if v == i][0]


x_test, y_test = load_data('../set/test_seg.txt')
print(label_index)
x_test = transform_to_matrix(X=x_test, vec_size=200, padding_size=25)

x_test_char, y_test_char = load_data('../set/test_seg_char.txt')
x_test_char = transform_to_matrix_char(X=x_test_char, vec_size=200, padding_size=50)

pre_vec = np.argmax(model.predict([x_test, x_test_char], batch_size=batch_size), axis=1)

sens = load_sen('../set/test_data.txt')

f = open('../set/wrong.txt', 'w', encoding='utf-8')
print(classification_report(y_test, pre_vec, target_names=['NUM', 'OBJ', 'LOC', 'TIME', 'HUM', 'DES'], digits=4))
n = 0
for i in pre_vec:
    if y_test[n] != i:
        f.write(get_label(i) + '\t' + sens[n])
    n = n + 1
f.flush()
