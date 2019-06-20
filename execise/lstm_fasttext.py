import numpy as np
import sys
sys.path.append("..")
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, GlobalAvgPool1D, Input, LSTM, Dropout, Flatten
from keras.optimizers import adam
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot,Tokenizer
from util.data_preprocess import n_gram, load_data, load_data_line
from keras.utils import to_categorical
from util.common_metrics import evaluate, evaluate_union
from util.transform_martrix import transform_to_matrix
from keras.layers import concatenate
from keras.layers import Bidirectional
from keras.models import Model


max_lengths = 55
batch_size = 32
epochs = 100

def max_count(path):
    max = 0
    maxList = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip('\n').split('\t')
            if max < len(words[1].split(' ')):
                max = len([word for word in words[1].split(' ') if len(word) > 0])
                maxList = [word for word in words[1].split(' ') if len(word) > 0]
    print(max)
    print(maxList)


x_train, y_train = load_data_line('../set/train_seg_char_2gram.txt')

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(x_train)
seg = tokenizer.texts_to_sequences(x_train)
print('find %s unique words', len(tokenizer.word_index))
pad_seg = pad_sequences(seg, maxlen=max_lengths, padding='post')
y_train = to_categorical(np.asarray(y_train))

print('train data shape %s, and labels shape %s', pad_seg.shape, y_train.shape)


x_test, y_test = load_data_line('../set/test_seg_char_2gram.txt')

tokenizer1 = Tokenizer(num_words=20000)
tokenizer1.fit_on_texts(x_train)
seg_test = tokenizer1.texts_to_sequences(x_test)
print('find %s unique words', len(tokenizer1.word_index))
pad_seg_test = pad_sequences(seg_test, maxlen=max_lengths, padding='post')
y_test = to_categorical(np.asarray(y_test))

print('test data shape %s, and labels shape %s', pad_seg_test.shape, y_test.shape)

x_train_lstm, y_train_lstm = load_data('../set/train_seg.txt')
x_train_lstm = np.array(transform_to_matrix(X=x_train_lstm, vec_size=200))
print(x_train_lstm.shape)

x_test_lstm, y_test_lstm = load_data('../set/test_seg.txt')
x_test_lstm = np.array(transform_to_matrix(X=x_test_lstm, vec_size=200))

y_ = to_categorical(np.array(y_train_lstm + y_test_lstm))
y_train_lstm = y_[:x_train_lstm.shape[0]]
y_test_lstm = y_[x_train_lstm.shape[0]:]

input_fast = Input(shape=pad_seg.shape[1:])
input_lstm = Input(shape=x_train_lstm.shape[1:])

x1 = input_lstm
x1 = Bidirectional(LSTM(128, input_shape=x_train_lstm.shape[1:], return_sequences=True, activation='tanh'))(x1)
x1 = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(x1)
x1 = Dense(128, activation='relu')(x1)
x1 = Dropout(0.5)(x1)
x1 = Flatten()(x1)

x2 = input_fast
x2 = Embedding(len(tokenizer.word_index) + 1, 200, input_length=max_lengths)(x2)
x2 = GlobalAvgPool1D()(x2)
x2 = Dense(256, activation='relu')(x2)
x2 = Dropout(0.5)(x2)
merge_layer = concatenate([x1, x2], axis=1)
result_layer = Dense(6, activation='softmax')(merge_layer)


model = Model([input_lstm, input_fast], result_layer)
print(x_train_lstm.shape, pad_seg.shape)
#sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
adam = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit([x_train_lstm, pad_seg], y_train_lstm,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=[[x_test_lstm, pad_seg_test], y_test_lstm])
model.save('../models/lstm_fast.model.h5')
evaluate_union(model, x_test_lstm, pad_seg_test, y_test_lstm, batch_size=batch_size)

