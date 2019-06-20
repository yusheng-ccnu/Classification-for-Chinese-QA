import numpy as np
import sys
sys.path.append("..")
from keras.optimizers import SGD, adam
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from util.data_preprocess import load_data
from util.transform_martrix import transform_to_matrix_char, transform_to_matrix
from util.common_metrics import evaluate
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

label_index = {}


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


print('load train datasets')
x_train, y_train = load_data('../set/train_seg_char.txt')
x_train = transform_to_matrix_char(X=x_train, vec_size=200, padding_size=50)
x_train = np.array(x_train)
print('x_train shape', x_train.shape)
y_train = np.array(y_train)
y_train = to_categorical(y_train)

print('load test datasets')
x_test, y_test = load_data('../set/test_seg_char.txt')
x_test = transform_to_matrix_char(X=x_test, vec_size=200, padding_size=50)
x_test = np.array(x_test)
print('x_test.shape', x_test.shape)
y_test = to_categorical(np.array(y_test))


batch_size = 32
n_filter = 128
filter_length = 3
nb_epoch = 10
n_pool = 3
num_classes = 6

IS_TRAIN = True
p = []
a = []
r = []
f = []
for i in range(10):
    print('this is %s times train.' % (i + 1))
    if IS_TRAIN:
        # 新建一个sequential的模型
        model = Sequential()
        model.add(Convolution1D(n_filter, 3, padding='same', input_shape=x_train.shape[1:], use_bias=True, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=n_pool))
        model.add(Convolution1D(n_filter, 5, padding='same', use_bias=True, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=n_pool))
        model.add(SeqSelfAttention())
        model.add(Flatten())
        # 后面接上一个ANN
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        # compile模型
        adam_op = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=2, validation_data=[x_test, y_test])
        #model.save('../model/textcnn_model.h5')

        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(0, nb_epoch), history.history['loss'], label='train_loss')
        plt.title("Training Loss and Accuracy on sar classifier")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper right")
        plt.savefig("Loss_Accuracy_alexnet_{:d}e.png".format(nb_epoch))
        acc, prec, recall, f1 = evaluate(model, x_test, y_test)
        a.append(acc)
        p.append(prec)
        r.append(recall)
        f.append(f1)
    else:
        model = load_data('../models/textcnn_model.h5')
        #evaluate(model, x_test, y_test, batch_size=batch_size)

print('shape', x_train.shape)
print('textcnn 10 times average accuracy:{0:.4f}'.format(np.mean(a)))
print('10 times average precision:{0:.4f}'.format(np.mean(p)))
print('10 times average recall:{0:0.4f}'.format(np.mean(r)))
print('10 times average f1-score:{0:.4f}'.format(np.mean(f)))