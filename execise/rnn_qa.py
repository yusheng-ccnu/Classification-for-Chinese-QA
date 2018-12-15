import numpy as np
from gensim.models import Word2Vec
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import SimpleRNN
from keras import initializers
from util.common_metrics import evaluate
from util.data_preprocess import load_data
import warnings
warnings.filterwarnings('ignore')

#问题中最长的有18个词组成，默认使用128维的词向量
def transform_to_matrix(padding_size = 18, vec_size = 300, X=[]):
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


print('load train datasets')
x_train, y_train = load_data('../data/train_question.txt')
x_train = transform_to_matrix(X=x_train)
x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
y_train = np.array(y_train)
y_train = to_categorical(y_train)

print('load test datasets')
x_test, y_test = load_data('../data/test_question.txt')
x_test = transform_to_matrix(X=x_test)
x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
y_test = to_categorical(np.array(y_test))

print(x_train.shape)
num_classes = 7
epochs = 200
hidden_units = 100
BATCH_INDEX = 0

learning_rate = 1e-3
clip_norm = 1.0
batch_size = 128

# 新建一个sequential的模型
model = Sequential()
model.add(SimpleRNN(hidden_units,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=x_train.shape[1:]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
sgd = SGD(lr=learning_rate, decay=1e-2, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2)
evaluate(model, x_test, y_test)