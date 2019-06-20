import numpy as np
import sys
sys.path.append("..")
from keras.optimizers import SGD, adam
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM, Dropout, GRU, Activation, Flatten,GlobalAvgPool1D
from util.common_metrics import evaluate
from util.data_preprocess import load_data
from util.transform_martrix import transform_to_matrix, transform_to_matrix_char
from util.data_preprocess import get_labels
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')


print('load train datasets')
x_train, y_train = load_data('../set/train_seg_char.txt')
x_train = transform_to_matrix_char(X=x_train, vec_size=200, padding_size=50)
x_train = np.array(x_train)
print('x_train shape', x_train.shape)
labels = get_labels()

print('load test datasets')
x_test, y_test = load_data('../set/test_seg_char.txt')
x_test = transform_to_matrix_char(X=x_test, vec_size=200, padding_size=50)
x_test = np.array(x_test)
print('x_test shape', x_test.shape)

y_ = to_categorical(np.array(y_train + y_test))
y_train = y_[:x_train.shape[0]]
y_test = y_[x_train.shape[0]:]


print(x_train.shape)
num_classes = 6
epochs = 80
learning_rate = 1e-2
clip_norm = 1.0
batch_size = 32

p = []
a = []
r = []
f = []

# 新建一个sequential的模型
for i in range(3):
    print('this is %s times train.' % (i + 1))
    model = Sequential()
    model.add(GRU(128, return_sequences=True,
                                 input_shape=x_train.shape[1:],
                                 use_bias=True,
                                 dropout=0.5,
                                 activation='tanh')) # 返回维度为 128 的向量序列
    model.add(GRU(128, return_sequences=True, use_bias=True, dropout=0.5, activation='tanh'))
    model.add(GRU(128, dropout=0.5, use_bias=True))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    adam_ap = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam_ap,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=[x_test, y_test])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    acc, prec, recall, f1 = evaluate(model, x_test, y_test)
    a.append(acc)
    p.append(prec)
    r.append(recall)
    f.append(f1)

print('shape', x_train.shape)
print('LSTM 10 times average accuracy:{0:.4f}'.format(np.mean(a)))
print('LSTM 10 times average precision:{0:.4f}'.format(np.mean(p)))
print('LSTM 10 times average recall:{0:0.4f}'.format(np.mean(r)))
print('LSTM 10 times average f1-score:{0:.4f}'.format(np.mean(f)))