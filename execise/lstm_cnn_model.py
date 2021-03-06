import numpy as np
import sys
sys.path.append("..")
from util.transform_martrix import transform_to_matrix, transform_to_matrix_char
from util.data_preprocess import load_data
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPool1D, Dense, LSTM, Flatten, Input, Dropout
from keras.optimizers import SGD, adam
from util.common_metrics import evaluate_union
from keras.layers import concatenate,Bidirectional
from keras.models import Model
import warnings
warnings.filterwarnings('ignore')

learning_rate = 0.01
batch_size = 32
epochs = 80

x_train, y_train = load_data('../set/train_seg.txt')
x_train = np.array(transform_to_matrix(X=x_train, vec_size=200))
print(x_train.shape)

x_test, y_test = load_data('../set/test_seg.txt')
x_test = np.array(transform_to_matrix(X=x_test, vec_size=200))

y_ = to_categorical(np.array(y_train + y_test))
y_train = y_[:x_train.shape[0]]
y_test = y_[x_train.shape[0]:]


print('load train_char datasets')
x_train_char, y_train_char = load_data('../set/train_seg_char.txt')
x_train_char = transform_to_matrix_char(X=x_train_char, vec_size=200, padding_size=50)
x_train_char = np.array(x_train_char)

print('x_train_char shape', x_train.shape)
y_train_char = np.array(y_train_char)
y_train_char = to_categorical(y_train_char)

print('load test_char datasets')
x_test_char, y_test_char = load_data('../set/test_seg_char.txt')
x_test_char = transform_to_matrix_char(X=x_test_char, vec_size=200, padding_size=50)
x_test_char = np.array(x_test_char)
print('x_test.shape', x_test_char.shape)
y_test_char = to_categorical(np.array(y_test_char))
input_lstm = Input(shape=(x_train.shape[1:]))
input_cnn = Input(shape=(x_train_char.shape[1:]))

x1 = input_lstm
x1 = Bidirectional(LSTM(128, input_shape=x_train.shape[1:], return_sequences=True, activation='relu', dropout=0.5))(x1)
x1 = Bidirectional(LSTM(128, activation='relu', dropout=0.5))(x1)
x1 = Dense(128, activation='relu')(x1)
x1 = Dropout(0.5)(x1)

x2 = input_cnn
x2 = Conv1D(128, 3, input_shape=x_train.shape[1:], use_bias=True, activation='relu', padding='same')(x2)
x2 = MaxPool1D(pool_size=3)(x2)
x2 = Conv1D(128, 5, use_bias=True, activation='relu', padding='same')(x2)
x2 = MaxPool1D(pool_size=5)(x2)
x2 = Flatten()(x2)
x2 = Dense(128, activation='relu')(x2)
merge_layer = concatenate([x1, x2], axis=1)
result_layer = Dense(6, activation='softmax')(merge_layer)


model = Model([input_lstm, input_cnn], result_layer)

# sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
adam = adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit([x_train, x_train_char], y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=[[x_test, x_test_char], y_test])
model.save('../models/lstm_cnn.model.h5')
evaluate_union(model, x_test, x_test_char, y_test, batch_size=batch_size)