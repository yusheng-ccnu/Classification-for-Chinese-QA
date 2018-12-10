import jieba
import numpy as np
from gensim.models import Word2Vec
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import plot_model
from keras.models import load_model
from sklearn import metrics
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


print('load train datasets')
x_train, y_train = load_data('../data/train_question.txt')
x_train = transform_to_matrix(X = x_train)
x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1], 1)
y_train = np.array(y_train)
y_train = to_categorical(y_train)

print('load test datasets')
x_test, y_test = load_data('../data/test_question.txt')
x_test = transform_to_matrix(X = x_test)
x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[1], 1)
y_test = to_categorical(np.array(y_test))

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


def on_epoch_end(self, epoch, logs={}):
    val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
    val_targ = self.model.validation_data[1]
    _val_f1 = f1_score(val_targ, val_predict)
    _val_recall = recall_score(val_targ, val_predict)
    _val_precision = precision_score(val_targ, val_predict)
    self.val_f1s.append(_val_f1)
    self.val_recalls.append(_val_recall)
    self.val_precisions.append(_val_precision)
    print('-val_f1: %.4f --val_precision: %.4f --val_recall: %.4f' % (_val_f1, _val_precision, _val_recall))
    return

batch_size = 16
n_filter = 16
filter_length = 2
nb_epoch = 100
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
model.add(Dense(7))
model.add(Activation('softmax'))
# compile模型
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=2, validation_split=0.25)
model.save('../models/cnn.execise.h5')
#model = load_model('../models/cnn.execise.h5')
#plot_model(execise, to_file='execise.png')

scores = model.evaluate(x_test, y_test, verbose=0)
print('CNN test score:', scores[0])
print('CNN test accuracy:', scores[1])
p = []
r = []
f1 = []
pre_vec = model.predict(x_test, batch_size = batch_size)
y_pred = np.argmax(pre_vec, axis=1)
y_ = np.argmax(y_test, axis=1)
prec = metrics.precision_score(y_, y_pred, average='weighted')
p.append(prec)
print('precision:{0:.3f}'.format(prec))
rec = metrics.recall_score(y_, y_pred, average='weighted')
r.append(rec)
print('recall:{0:0.3f}'.format(rec))
f1sco = metrics.f1_score(y_, y_pred, average='weighted')
f1.append(f1sco)
print('f1-score:{0:.3f}'.format(f1sco))
