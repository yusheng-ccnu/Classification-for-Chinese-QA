import numpy as np
from gensim.models import Word2Vec
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import GRU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from util.data_preprocess import load_data
from keras.models import load_model
from util.common_metrics import evaluate
import warnings
warnings.filterwarnings('ignore')

label_index = {}


def max_count(path):
    max = 0;
    maxList = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip('\n').split('\t')
            if max < len(words[1].split(' ')):
                max = len([word for word in words[1].split(' ') if len(word) > 0])
                maxList = [word for word in words[1].split(' ') if len(word) > 0]
    print(max)
    print(maxList)
    print(len(maxList[5]))


#问题中最长的有20个词组成，默认使用128维的词向量
def transform_to_matrix(padding_size=20, vec_size=300, X=[]):
    res = []
    model = Word2Vec.load('../embedding/word_embedding_300')
    for sen in X:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist())
            except:
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res



#max_count('../data/train_question.txt')


print('load train datasets')
x_train, y_train = load_data('../data/train_question.txt')
x_train = transform_to_matrix(X=x_train)
x_train = np.array(x_train)
print()
print('x_train shape', x_train.shape)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1])
y_train = np.array(y_train)
y_train = to_categorical(y_train)

print('load test datasets')
x_test, y_test = load_data('../data/test_question.txt')
x_test = transform_to_matrix(X=x_test)
x_test = np.array(x_test)
print('x_test.shape', x_test.shape)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[1])
y_test = to_categorical(np.array(y_test))


batch_size = 64
n_filter = 64
filter_length = 2
nb_epoch = 100
n_pool = 2

IS_TRAIN = True
if IS_TRAIN:
    # 新建一个sequential的模型
    model = Sequential()
    model.add(GRU(128, return_sequences=True, input_shape=x_train.shape[1:]))
    model.add(Dropout(0.6))
    model.add(Activation('relu'))
    model.add(Flatten())
    # 后面接上一个ANN
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    # compile模型
    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=2)
    model.save('../models/gru_model.h5')
    evaluate(model, x_test, y_test)
else:
    model = load_data('../models/gru_model.h5')
    evaluate(model, x_test, y_test)
