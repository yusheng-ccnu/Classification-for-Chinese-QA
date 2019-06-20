import sys
sys.path.append("..")
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import SimpleRNN
from keras.optimizers import adam
from util.data_preprocess import load_data
from keras.callbacks import EarlyStopping, ModelCheckpoint
from util.common_metrics import evaluate
from util.transform_martrix import transform_to_matrix, transform_to_matrix_char
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt


# Convolution
filter_length = 3
nb_filter = 256
pool_length = 3

# LSTM
lstm_output_size = 256

# Training
batch_size = 32
nb_epoch = 10


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

p = []
a = []
r = []
f = []

for i in range(3):
    model = Sequential()
    model.add(SimpleRNN(lstm_output_size, dropout=0.5, use_bias=True, return_sequences=True,
                                 input_shape=x_train.shape[1:]))
    model.add(SimpleRNN(lstm_output_size,  dropout=0.5, use_bias=True, return_sequences=True))
    model.add(SimpleRNN(lstm_output_size,  dropout=0.5, use_bias=True))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    adam_ad = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam_ad, metrics=['accuracy'])

    print('Train...', i)
    best_weights_filepath = '../models/best_weights.hdf5'
    earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='max')
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1,
                                    save_best_only=True, mode='max')
    history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(x_test, y_test), callbacks=[earlyStopping, saveBestModel])
    model = load_model(best_weights_filepath)
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, nb_epoch), history['loss'], label='train_loss')
    plt.title("Training Loss and Accuracy on sar classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig("Loss_Accuracy_alexnet_{:d}e.jpg".format(nb_epoch))
    acc, prec, recall, f1 = evaluate(model, x_test, y_test)
    a.append(acc)
    p.append(prec)
    r.append(recall)
    f.append(f1)
print('shape', x_train.shape)
print('rnn 5 times average accuracy:{0:.4f}'.format(np.mean(a)))
print('rnn 5 average precision:{0:.4f}'.format(np.mean(p)))
print('rnn 5 average recall:{0:0.4f}'.format(np.mean(r)))
print('rnn 5 average f1-score:{0:.4f}'.format(np.mean(f)))