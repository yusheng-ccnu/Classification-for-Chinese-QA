from keras.models import Sequential
from keras.layers import Dense, Activation
from gensim.models import Word2Vec
from keras.utils import to_categorical
import numpy as np
from util.common_metrics import evaluate
from util.data_preprocess import load_data
import warnings
warnings.filterwarnings("ignore")


def transform_to_matrix(X = [], vec_size = 300):
    vec = Word2Vec.load('../embedding/word_embedding_300')
    res = []
    for sen in X:
        mark = np.zeros(vec_size)
        for word in sen:
            try:
                mark += vec[word]
            except:
                mark += np.zeros(vec_size)
        res.append(mark)
    return res


x_train, y_train = load_data('../data/train_question.txt')
x_train = np.array(transform_to_matrix(X=x_train))
y_train = to_categorical(y_train)
print(x_train.shape)

x_test, y_test = load_data('../data/test_question.txt')
x_test = np.array(transform_to_matrix(X=x_test))
y_test = to_categorical(y_test)

print('defined nn model')
model = Sequential()
model.add(Dense(412, input_dim=300))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200, batch_size=128, verbose=2)
scores = model.evaluate(x_test, y_test, verbose=0)
print('NN test score:', scores[0])
print('NN test accuracy:', scores[1])

evaluate(model, x_test, y_test)
