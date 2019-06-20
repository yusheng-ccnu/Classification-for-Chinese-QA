from keras.models import Sequential
from keras.layers import Dense, Activation
from gensim.models import Word2Vec
from keras.utils import to_categorical
import numpy as np
from util.common_metrics import evaluate
from util.data_preprocess import load_data
from util.data_preprocess import load_data_pred
from util.data_preprocess import get_labels
import warnings
warnings.filterwarnings("ignore")


def transform_to_matrix(X=[], vec_size=300):
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
print(x_train.shape)

labels = get_labels()

x_test, y_test = load_data('../data/test_question.txt')
x_test = np.array(transform_to_matrix(X=x_test))
print(x_test.shape)
print(np.array(y_train + y_test).shape)

y_ = to_categorical(np.array(y_train + y_test))
print(y_.shape)
y_train = y_[:x_train.shape[0]]
y_test = y_[x_train.shape[0]:]

x_pred_data, y_pred, x_data = load_data_pred('../dataset/nlp_qa_seg.txt', '../dataset/nlp_qa.txt')
x_pred = np.array(transform_to_matrix(X=x_pred_data))

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
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=2)
model.save('../models/nn.model')
scores = model.evaluate(x_test, y_test, verbose=0)
print('NN test score:', scores[0])
print('NN test accuracy:', scores[1])

pred = np.argmax(model.predict(x_pred), axis=1)
print(pred[:10])
print(labels)
file = open('../dataset/nlp_bz.txt', 'w', encoding='utf-8')
for i in range(len(pred)):
    print(i)
    file.write(list(labels.keys())[list(labels.values()).index(pred[i])] + '\t' + x_data[i] + '\n')
evaluate(model, x_test, y_test)
