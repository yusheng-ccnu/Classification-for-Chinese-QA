import sys
sys.path.append("..")
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, GlobalAvgPool1D, Dropout
from keras.optimizers import adam
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot,Tokenizer
from util.data_preprocess import n_gram, load_data, load_data_line
from keras.utils import to_categorical
from util.common_metrics import evaluate
import numpy as np

max_lengths = 160


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


x_train, y_train = load_data_line('../set/train_char_3gram.txt')

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(x_train)
seg = tokenizer.texts_to_sequences(x_train)
print('find %s unique words', len(tokenizer.word_index))
pad_seg = pad_sequences(seg, maxlen=max_lengths, padding='post')
y_train = to_categorical(np.asarray(y_train))

print('train data shape %s, and labels shape %s', pad_seg.shape, y_train.shape)


x_test, y_test = load_data_line('../set/test_char_3gram.txt')

tokenizer1 = Tokenizer(num_words=20000)
tokenizer1.fit_on_texts(x_train)
seg_test = tokenizer1.texts_to_sequences(x_test)
print('find %s unique words', len(tokenizer1.word_index))
pad_seg_test = pad_sequences(seg_test, maxlen=max_lengths, padding='post')
y_test = to_categorical(np.asarray(y_test))

print('test data shape %s, and labels shape %s', pad_seg_test.shape, y_test.shape)

IS_TRAIN = True

if IS_TRAIN:
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, 300, input_length=max_lengths))
    model.add(Dropout(0.5))
    model.add(GlobalAvgPool1D())
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    adam = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(pad_seg, y_train, batch_size=32, epochs=60, verbose=2)
    evaluate(model, pad_seg_test, y_test, batch_size=32)

