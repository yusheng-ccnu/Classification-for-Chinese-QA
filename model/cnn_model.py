import jieba
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

label_index = {}


def max_count(path):
    max = 0;
    maxList = []
    with open(path, 'r') as f:
        for line in f:
            words = line.strip('\n').split('\t')
            if max < len(words[1].split(' ')):
                max = len([word for word in words[1].split(' ') if len(word) > 0])
                maxList = [word for word in words[1].split(' ') if len(word) > 0]
    print(max)
    print(maxList)
    print(len(maxList[5]))


def load_stopword(path):
    stop_word = []
    with open(path, 'r', encoding='UTF-8') as file:
        for line in file:
            stop_word.append(line.strip('\n'))
    return stop_word


def particle_word(in_path, out_path, stop_path):
    f_test = open(out_path, 'w')
    label_index = {}
    stop_word = load_stopword(stop_path)
    with open(in_path, 'r') as file:
        for line in file:
            terms = line.strip('\n').split('\t')
            label = terms[0].split('_')[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            words = jieba.cut(terms[1].strip(' '))
            f_test.write(label + '\t' + ' '.join([word for word in words if (len(word) > 0 and word not in stop_word)]) + '\n')


#问题中最长的有18个词组成，默认使用128维的词向量
def transform_to_matrix(padding_size = 18, vec_size = 128, X=[]):
    res = []
    model = Word2Vec.load('../embedding/word_embedding_128')
    for sen in X:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist())
            except:
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res


def load_data(path):
    x_train = []
    y_train = []

    with open(path, 'r') as file:
        for line in file:
            words = line.strip('\n').split('\t')
            label = words[0]
            if label not in label_index:
                label_index[label] = len(label_index)
            x_train.append(words[1].strip('\n').split(' '))
            y_train.append(label_index[label])
    return x_train, y_train


x_train, y_train = load_data('../data/train_question.txt')
x_train = transform_to_matrix(X = x_train)
x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1], 1)
y_train = np.array(y_train)
#y_train = y_train.reshape(y_train.shape[0], len(label_index))
print(y_train.shape)
print(len(label_index))
# set parameters:
batch_size = 32
n_filter = 16
filter_length = 2
nb_epoch = 5
n_pool = 2

# 新建一个sequential的模型
model = Sequential()
model.add(Convolution2D(n_filter, (filter_length, filter_length), input_shape=(128, 18, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
# 后面接上一个ANN
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))
# compile模型
model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=0)