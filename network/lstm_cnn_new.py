import numpy as np
import sys
sys.path.append("..")
from util.transform_martrix import transform_to_matrix, transform_to_matrix_char
from util.data_preprocess import load_data, get_labels
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPool1D, Dense, LSTM, Flatten, Input, Dropout
from keras.optimizers import adam
from util.common_metrics import evaluate_union
from keras.layers import Bidirectional, add, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

learning_rate = 0.01
batch_size = 32
epochs = 50

x_train, y_train = load_data('../irdata/train_seg.txt')
x_train = np.array(transform_to_matrix(X=x_train, vec_size=200, padding_size=25))
print(x_train.shape)


x_test, y_test = load_data('../irdata/test_seg.txt')
x_test = np.array(transform_to_matrix(X=x_test, vec_size=200, padding_size=25))

y_ = to_categorical(np.array(y_train + y_test))
y_train = y_[:x_train.shape[0]]
y_test = y_[x_train.shape[0]:]


print('load train_char datasets')
x_train_char, y_train_char = load_data('../irdata/train_seg_char.txt')
x_train_char = transform_to_matrix_char(X=x_train_char, vec_size=200, padding_size=50)
x_train_char = np.array(x_train_char)

print('x_train_char shape', x_train_char.shape)
y_train_char = np.array(y_train_char)
y_train_char = to_categorical(y_train_char)

print('load test_char datasets')
x_test_char, y_test_char = load_data('../irdata/test_seg_char.txt')
x_test_char = transform_to_matrix_char(X=x_test_char, vec_size=200, padding_size=50)
x_test_char = np.array(x_test_char)
print('x_test.shape', x_test_char.shape)
y_test_char = to_categorical(np.array(y_test_char))
print(get_labels())

p = []
a = []
r = []
f = []
for i in range(1):
    print('this is %s times train.' % (i + 1))
    input_lstm = Input(shape=(x_train.shape[1:]))
    input_char = Input(shape=(x_train_char.shape[1:]))

    x1 = input_lstm
    x1 = Bidirectional(LSTM(256, activation='tanh', return_sequences=True, dropout=0.5, use_bias=True))(x1)
    #x1 = Bidirectional(LSTM(128, return_sequences=True, activation='tanh', dropout=0.5, use_bias=True))(x1)
    x1 = Bidirectional(LSTM(256, activation='tanh', dropout=0.5, use_bias=True, name='lstm_2'))(x1)

    x2 = input_char
    x2 = Bidirectional(LSTM(256, return_sequences=True, activation='tanh', dropout=0.5, use_bias=True))(x2)
    #x2 = Bidirectional(LSTM(128, return_sequences=True, activation='tanh', dropout=0.5, use_bias=True))(x2)
    x2 = Bidirectional(LSTM(256, activation='tanh', dropout=0.5, use_bias=True, return_sequences=True))(x2)

    merge_layer = add([x1, x2])
    kernel = [2, 3, 5]
    cnn = []
    for size in kernel:
        c = Conv1D(256, 5, use_bias=True, activation='relu')(merge_layer)
        pool = MaxPool1D(pool_size=3)(c)
        cnn.append(pool)

    cnn = concatenate([pool for pool in cnn])
    cnn = Conv1D(256, 3, use_bias=True, activation='relu')(cnn)
    cnn = MaxPool1D(pool_size=3)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(256, activation='relu')(cnn)
    cnn = Dropout(0.5)(cnn)
    result_layer = Dense(6, activation='softmax')(cnn)

    model = Model([input_lstm, input_char], result_layer)

    # sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    adam_ad = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam_ad,
                  metrics=['accuracy'])
    best_weights_filepath = '../models/best_weights_235.hdf5'
    earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='max')
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1,
                                               save_best_only=True, mode='max')
    history = model.fit([x_train, x_train_char], y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=[[x_test, x_test_char], y_test], callbacks=[earlyStopping, saveBestModel])
    model = load_model(best_weights_filepath)
    acc, prec, recall, f1 = evaluate_union(model, x_test, x_test_char, y_test, batch_size=batch_size)
    a.append(acc)
    p.append(prec)
    r.append(recall)
    f.append(f1)
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(history.epoch, history.history['loss'], label='Train_loss')
    plt.title("Training Loss on BiLSTM-CNN classifier")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.legend(loc="upper right")
    plt.savefig("bilstm_235cnn_{:d}_{:d}e.png".format(len(history.epoch), i))

    print(history.epoch)

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(history.epoch, history.history['val_loss'], label='Test_loss')
    plt.title("Test Loss on BiLSTM-CNN classifier")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.legend(loc="upper right")
    plt.savefig("bilstm_235cnn_test_{:d}_{:d}e.png".format(len(history.epoch), i))

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(history.epoch, history.history['acc'], label='Train_Acc')
    plt.title("Train acc on BiLSTM-CNN classifier")
    plt.xlabel("Epoch")
    plt.ylabel("Train acc")
    plt.legend(loc="upper right")
    plt.savefig("bilstm_235cnn_test_acc_{:d}_{:d}e.png".format(len(history.epoch), i))

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(history.epoch, history.history['val_acc'], label='Test_Acc')
    plt.title("Test acc on BiLSTM-CNN classifier")
    plt.xlabel("Epoch")
    plt.ylabel("Test acc")
    plt.legend(loc="upper right")
    plt.savefig("bilstm_235cnn_train_acc_{:d}_{:d}e.png".format(len(history.epoch), i))

print('covolution size 5 3')
print('shape word', x_train.shape)
print('word shape', x_train_char.shape)
print('lstm_cnn_new 3 times average accuracy:{0:.4f}'.format(np.mean(a)))
print('lstm_cnn_new 3 times average precision:{0:.4f}'.format(np.mean(p)))
print('lstm_cnn_new 3 times average recall:{0:0.4f}'.format(np.mean(r)))
print('lstm_cnn_new 3 times average f1-score:{0:.4f}'.format(np.mean(f)))
