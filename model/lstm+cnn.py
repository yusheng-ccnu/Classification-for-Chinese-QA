#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@version: python 3.6.1
@author: Error404LiuY
@file: lstm+cnn.py
@time: 2018/1/16 17:56
"""

import os
import numpy as np
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, concatenate, Dropout, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Bidirectional
from keras.optimizers import adam
from keras.models import Model
from sklearn import metrics

BASE_DIR = '..'
GLOVE_DIR = BASE_DIR + '/embedding/'
TEXT_DATA_DIR = BASE_DIR + '/data/'
MAX_SEQUENCE_LENGTH = 30  #max 26
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')
w2vmodel = gensim.models.Word2Vec.load(os.path.join(GLOVE_DIR, 'word_embedding_100'))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
path = os.path.join(TEXT_DATA_DIR, 'train_seg.txt')
print (path)
with open(path, 'r', encoding='UTF-8') as trainfile:
	for line in trainfile:
		linelist = line.replace('\n', '').split('\t')
		if linelist[0] not in labels_index:
			label_id = len(labels_index)
			labels_index[linelist[0]] = label_id
		texts.append(linelist[1])
		labels.append(labels_index[linelist[0]])
num_trainText = len(texts)   #4981
path = os.path.join(TEXT_DATA_DIR, 'test_seg.txt')
with open(path, 'r', encoding='UTF-8') as testfile:
	for line in testfile:
		linelist = line.replace('\n', '').split('\t')
		texts.append(linelist[1])
		label = linelist[0]
		labels.append(labels_index[label])

print('Found %s texts.' % len(texts))
print(np.array(labels).shape)
# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)    # 填充长度为MAX_SEQUENCE_LENGTH

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
print(labels)

x_train = data[:num_trainText]
y_train = labels[:num_trainText]
x_val = data[num_trainText:]
y_val = labels[num_trainText:]

print ('shape of train texts: ', x_train.shape)
print ('shape of val texts: ', x_val.shape)
print ('shape of y_val texts: ', y_val.shape)

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
	if i > MAX_NB_WORDS:
		continue
	try:
		embedding_vector = w2vmodel[word]
	except:
		embedding_vector = np.zeros(EMBEDDING_DIM,)
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector # word_index to word_embedding_vector ,<10000(nb_words)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words + 1,
							EMBEDDING_DIM,
							input_length=MAX_SEQUENCE_LENGTH,
							weights=[embedding_matrix],
							trainable=False)

# train a 1D convnet with global maxpoolinnb_wordsg
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

#***************************
print ('Now is char part, use CNN execise.')

MAX_SEQUENCE_LENGTH_CHAR = 60  #max 58

print('Indexing char vectors.')
w2vmodel_char = gensim.models.Word2Vec.load(os.path.join(GLOVE_DIR, 'char_100.bin'))

# second, prepare text samples and their labels
print('Processing text dataset to char texts.')

texts_char = []  # list of text samples

for line in texts:
	line_char = (' ').join(list(('').join(line.split(' '))))
	texts_char.append(line_char)

print('Found %s char texts.' % len(texts_char))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer_char = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer_char.fit_on_texts(texts_char)
sequences_char = tokenizer_char.texts_to_sequences(texts_char)

word_index_char = tokenizer_char.word_index
print('Found %s unique char tokens.' % len(word_index_char))

data_char = pad_sequences(sequences_char, maxlen=MAX_SEQUENCE_LENGTH_CHAR)     #填充长度为MAX_SEQUENCE_LENGTH

x_train_char = data_char[:num_trainText]
x_val_char = data_char[num_trainText:]

print ('shape of train texts: ', x_train_char.shape)
print ('shape of val texts: ', x_val_char.shape)

print('Preparing embedding char matrix.')

# prepare embedding matrix
nb_words_char = min(MAX_NB_WORDS, len(word_index_char))
embedding_matrix_char = np.zeros((nb_words_char + 1, EMBEDDING_DIM))
for word, i in word_index_char.items():
	if i > MAX_NB_WORDS:
		continue
	try:
		embedding_vector = w2vmodel_char[word]
	except:
		embedding_vector = np.zeros(EMBEDDING_DIM,)
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix_char[i] = embedding_vector # word_index to word_embedding_vector ,<10000(nb_words)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer_char = Embedding(nb_words_char + 1,
							EMBEDDING_DIM,
							input_length=MAX_SEQUENCE_LENGTH_CHAR,
							weights=[embedding_matrix_char],
							trainable=False)

# train a 1D convnet with global maxpoolinnb_wordsg
sequence_input_char = Input(shape=(MAX_SEQUENCE_LENGTH_CHAR,), dtype='int32')
embedded_sequences_char = embedding_layer_char(sequence_input_char)
print(embedded_sequences_char.get_shape())
print('Training execise.')

p = []
r = []
f1 = []
for i in range(10):
	print ('this is %s times train.' % (i + 1))
	x1 = embedded_sequences
	x1 = LSTM(256, activation='relu', dropout=0.5, return_sequences=True)(x1)
	x1 = LSTM(256, activation='relu', dropout=0.5)(x1)
	x1 = Dense(256, activation='relu')(x1)
	x2 = embedded_sequences_char
	x2 = Conv1D(128, 3, padding='same', activation='relu')(x2)
	x2 = Dropout(0.5)(x2)
	x2 = MaxPooling1D(5)(x2)
	x2 = Conv1D(128, 5, padding='same', activation='relu')(x2)
	x2 = Dropout(0.5)(x2)
	x2 = MaxPooling1D(5)(x2)
	x2 = Dense(128, activation='relu')(x2)
	flatten = Flatten()(x2)
	#print ('shape of flatten: %s' % np.shape(flatten))
	merged_tensor = concatenate([x1, flatten], axis = 1)
	preds = Dense(len(labels_index), activation='softmax')(merged_tensor)

	adam = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model = Model([sequence_input, sequence_input_char], preds)
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	model.fit([x_train, x_train_char], y_train, epochs = 20, batch_size = 16, verbose = 1, shuffle = True, validation_data=[[x_val,x_val_char],y_val])

	pre_vec = model.predict([x_val, x_val_char], batch_size = 16)
	y_pred = np.argmax(pre_vec, axis=1)
	y_ = np.argmax(y_val, axis=1)
	prec = metrics.precision_score(y_, y_pred, average='weighted')
	p.append(prec)
	print('precision:{0:.3f}'.format(prec))
	rec = metrics.recall_score(y_, y_pred, average='weighted')
	r.append(rec)
	print('recall:{0:0.3f}'.format(rec))
	f1sco = metrics.f1_score(y_, y_pred, average='weighted')
	f1.append(f1sco)
	print('f1-score:{0:.3f}'.format(f1sco))

sum = 0
for s in p:
	sum += s
print('average precision: %s' % (sum / len(p)))

sum = 0
for s in r:
	sum += s
print('average recall: %s' % (sum / len(r)))

sum = 0
for s in f1:
	sum += s
print('average f1-score: %s' % (sum / len(f1)))