import numpy as np
import sys
sys.path.append("..")
from util.transform_martrix import transform_to_matrix_gram
from util.data_preprocess import load_data
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPool2D, Reshape, Dense, LSTM, Flatten, Input, Dropout,Activation,MaxPooling1D,TimeDistributed,Bidirectional
from keras.models import Sequential
from keras.optimizers import SGD, adam
from util.common_metrics import evaluate
from keras.layers import concatenate
from keras.models import Model



learning_rate = 1e-2
batch_size = 16
epochs = 70


x_train, y_train = load_data('../set/train_seg_char_2gram.txt')
x_train = np.array(transform_to_matrix_gram(X=x_train, vec_size=200, padding_size=100))
print(x_train.shape)

x_train_cnn = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])

x_test, y_test = load_data('../set/test_seg_char_2gram.txt')
x_test = np.array(transform_to_matrix_gram(X=x_test, vec_size=200, padding_size=100))

y_ = to_categorical(np.array(y_train + y_test))
y_train = y_[:x_train.shape[0]]
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
y_test = y_[x_train.shape[0]:]
x_test_cnn = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])


model = Sequential()
model.add(TimeDistributed(Conv1D(128, 3, padding='same', input_shape=x_train.shape[1:], use_bias=True)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling1D(pool_size=3)))
model.add(TimeDistributed(Conv1D(128, 5, padding='same', use_bias=True)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling1D(pool_size=3)))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dropout(0.5)))
model.add(LSTM(128, dropout=0.5, return_sequences=True, use_bias=True))
model.add(LSTM(128, dropout=0.5, use_bias=True))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

#model1 = Sequential()
#model1.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=x_train.shape[1:], use_bias=True, dropout=0.5, activation='tanh')))
#model1.add(Bidirectional(LSTM(128, return_sequences=True, use_bias=True, dropout=0.5, activation='tanh')))
#model1.add(Bidirectional(LSTM(128, dropout=0.5)))

#sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
adam = adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
model.fit(x_train_cnn, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test_cnn, y_test))
model.save('../models/lstm_cnn.model.h5')
evaluate(model, x_test_cnn, y_test)