import sys
sys.path.append("..")
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import adam
from util.data_preprocess import load_data
from util.transform_martrix import transform_to_matrix, transform_to_matrix_char
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from util.common_metrics import evaluate

max_features = 23
maxlen = 1000
embedding_size = 128

# Convolution
filter_length = 3
nb_filter = 256
pool_length = 3

# LSTM
lstm_output_size = 256

# Training
batch_size = 32
nb_epoch = 100


print('load train datasets')
x_train, y_train = load_data('../set/train_seg_char.txt')
x_train = transform_to_matrix_char(X=x_train, vec_size=300, padding_size=50)
x_train = np.array(x_train)
print('x_train shape', x_train.shape)

print('load test datasets')
x_test, y_test = load_data('../set/test_seg_char.txt')
x_test = transform_to_matrix_char(X=x_test, vec_size=300, padding_size=50)
x_test = np.array(x_test)
print('x_test shape', x_test.shape)

y_ = to_categorical(np.array(y_train + y_test))
y_train = y_[:x_train.shape[0]]
y_test = y_[x_train.shape[0]:]

model = Sequential()
model.add(Convolution1D(nb_filter=nb_filter, filter_length=5, activation='relu', input_shape=x_train.shape[1:]))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(Convolution1D(nb_filter=nb_filter, filter_length=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_length=pool_length))

model.add(LSTM(lstm_output_size, activation='tanh', dropout=0.5, return_sequences=True))
model.add(LSTM(lstm_output_size, activation='tanh', dropout=0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
adam_ad = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam_ad, metrics=['accuracy'])

print('Train...')
best_weights_filepath = '../models/conv_lstm.hdf5'
earlyStopping = EarlyStopping(monitor='val_acc', patience=30, verbose=1, mode='max')
saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1,
                                               save_best_only=True, mode='max')
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(x_test, y_test),
          callbacks=[earlyStopping, saveBestModel])

model = load_model(best_weights_filepath)
evaluate(model, x_test, y_test, batch_size=batch_size)
