import numpy as np
import sys
sys.path.append("..")
from util.transform_martrix import transform_to_matrix, transform_to_matrix_char, transform_to_matrix_gram
from util.data_preprocess import load_data
from keras.utils import to_categorical
from keras.layers import Dense, LSTM, Input, Dropout
from keras.optimizers import SGD, adam
from util.common_metrics import evaluate_union
from keras.layers import concatenate, Bidirectional
from keras.models import Model
import warnings
warnings.filterwarnings('ignore')

learning_rate = 0.01
batch_size = 32
epochs = 100

x_train, y_train = load_data('../set/train_seg.txt')
x_train = np.array(transform_to_matrix(X=x_train, vec_size=300, padding_size=25))
print(x_train.shape)

x_test, y_test = load_data('../set/test_seg.txt')
x_test = np.array(transform_to_matrix(X=x_test, vec_size=300, padding_size=25))

y_ = to_categorical(np.array(y_train + y_test))
y_train = y_[:x_train.shape[0]]
y_test = y_[x_train.shape[0]:]


print('load train_char datasets')
x_train_char, y_train_char = load_data('../set/train_seg_char_2gram.txt')
x_train_char =np.array(transform_to_matrix_gram(X=x_train_char, padding_size=100, vec_size=300))
print('x_train_char shape', x_train.shape)

print('load test_char datasets')
x_test_char, y_test_char = load_data('../set/test_seg_char_2gram.txt')
x_test_char = np.array(transform_to_matrix_gram(X=x_test_char, padding_size=100, vec_size=300))

print('x_test.shape', x_test_char.shape)

y_char = to_categorical(np.array(y_train_char + y_test_char))

y_train_char = y_char[:x_train_char.shape[0]]
y_test_char = y_char[x_train_char.shape[0]:]
print(x_train_char.shape)

p = []
a = []
r = []
f = []
for i in range(2):
    print('this is %s times train.' % (i + 1))
    input_lstm = Input(shape=(x_train.shape[1:]))
    input_char = Input(shape=(x_train_char.shape[1:]))

    x1 = input_lstm
    x1 = Bidirectional(LSTM(128, return_sequences=True, activation='tanh', dropout=0.5, use_bias=True))(x1)
    # x1 = Bidirectional(LSTM(128, return_sequences=True, activation='tanh', dropout=0.5, use_bias=True))(x1)
    x1 = Bidirectional(LSTM(128, activation='tanh', dropout=0.5, use_bias=True))(x1)
    x1 = Dense(128, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)

    x2 = input_char
    x2 = Bidirectional(LSTM(128, return_sequences=True, activation='tanh', dropout=0.5, use_bias=True))(x2)
    # x2 = Bidirectional(LSTM(128, return_sequences=True, activation='tanh', dropout=0.5, use_bias=True))(x2)
    x2 = Bidirectional(LSTM(128, activation='tanh', dropout=0.5))(x2)
    x2 = Dense(128, activation='relu')(x2)

    x2 = Dropout(0.5)(x2)
    merge_layer = concatenate([x1, x2], axis=1)
    result_layer = Dense(6, activation='softmax')(merge_layer)

    model = Model([input_lstm, input_char], result_layer)

    # sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    adam_ad = adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam_ad,
                  metrics=['accuracy'])
    model.fit([x_train, x_train_char], y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=[[x_test, x_test_char], y_test])
    #model.save('../models/lstm_cnn.model.h5')
    acc, prec, recall, f1 = evaluate_union(model, x_test, x_test_char, y_test, batch_size=batch_size)
    a.append(acc)
    p.append(prec)
    r.append(recall)
    f.append(f1)

print('shape word and char_ngram', x_train.shape)
print('lstm_lstm model 5 times average accuracy:{0:.4f}'.format(np.mean(a)))
print('5 times average precision:{0:.4f}'.format(np.mean(p)))
print('5 times average recall:{0:0.4f}'.format(np.mean(r)))
print('5 times average f1-score:{0:.4f}'.format(np.mean(f)))