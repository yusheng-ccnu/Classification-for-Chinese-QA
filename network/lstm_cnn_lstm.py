import sys
sys.path.append("..")
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import adam
from util.data_preprocess import load_data
from keras.layers import Bidirectional
from util.transform_martrix import transform_to_matrix, transform_to_matrix_char
from keras.utils import to_categorical
from util.common_metrics import evaluate

max_features = 23
maxlen = 1000
embedding_size = 128

# Convolution
filter_length = 3
nb_filter = 128
pool_length = 3

# LSTM
lstm_output_size = 128

# Training
batch_size = 32
nb_epoch = 100


print('load train datasets')
x_train, y_train = load_data('../set/train_seg_char_2gram.txt')
x_train = transform_to_matrix_char(X=x_train, vec_size=300, padding_size=100)
x_train = np.array(x_train)
print('x_train shape', x_train.shape)

print('load test datasets')
x_test, y_test = load_data('../set/test_seg_char_2gram.txt')
x_test = transform_to_matrix_char(X=x_test, vec_size=300, padding_size=100)
x_test = np.array(x_test)
print('x_test shape', x_test.shape)

y_ = to_categorical(np.array(y_train + y_test))
y_train = y_[:x_train.shape[0]]
y_test = y_[x_train.shape[0]:]

model = Sequential()
model.add(Bidirectional(LSTM(256, return_sequences=True,
                                 input_shape=x_train.shape[1:],
                                 use_bias=True,
                                 dropout=0.5,
                                 activation='tanh')))  # 返回维度为 128 的向量序列
#model.add(Bidirectional(LSTM(128, return_sequences=True, use_bias=True, dropout=0.5, activation='tanh')))
#model.add(Bidirectional(LSTM(128, dropout=0.5, return_sequences=True, activation='tanh')))
model.add(Convolution1D(nb_filter=nb_filter, filter_length=5, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_length=3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
adam_ad = adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam_ad, metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
evaluate(model, x_test, y_test, batch_size=batch_size)
print('Test accuracy:', acc)
print('***********************************************************************')