import numpy as np
from util.data_preprocess import load_data
from util.transform_martrix import transform_to_matrix_char, transform_to_matrix
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Input
from keras.optimizers import adam
from keras.layers import concatenate
from keras.models import Model
from util.common_metrics import evaluate_union
import warnings
warnings.filterwarnings('ignore')

batch_size = 32
epochs = 100


x_train, y_train = load_data('../set/train_seg.txt')
x_train = np.array(transform_to_matrix(X=x_train, vec_size=200))
print(x_train.shape)

x_test, y_test = load_data('../set/test_seg.txt')
x_test = np.array(transform_to_matrix(X=x_test, vec_size=200))

y_ = to_categorical(np.array(y_train + y_test))
y_train = y_[:x_train.shape[0]]
y_test = y_[x_train.shape[0]:]

print('load train_char datasets')
x_train_char, y_train_char = load_data('../set/train_seg_char.txt')
x_train_char = transform_to_matrix_char(X=x_train_char, vec_size=200, padding_size=50)
x_train_char = np.array(x_train_char)

print('x_train_char shape', x_train.shape)
y_train_char = np.array(y_train_char)
y_train_char = to_categorical(y_train_char)

print('load test_char datasets')
x_test_char, y_test_char = load_data('../set/test_seg_char.txt')
x_test_char = transform_to_matrix_char(X=x_test_char, vec_size=200, padding_size=50)
x_test_char = np.array(x_test_char)

input_cnn = Input(shape=(x_train.shape[1:]))
input_char = Input(shape=(x_train_char.shape[1:]))

x1 = input_cnn
x1 = Conv1D(128, 3, input_shape=x_train.shape[1:], use_bias=True, activation='relu', padding='same')(x1)
x1 = MaxPool1D(pool_size=3)(x1)
x1 = Conv1D(128, 5, use_bias=True, activation='relu', padding='same')(x1)
x1 = MaxPool1D(pool_size=5)(x1)
x1 = Flatten()(x1)
x1 = Dense(128, activation='relu')(x1)

x2 = input_cnn
x2 = Conv1D(128, 3, input_shape=x_train.shape[1:], use_bias=True, activation='relu', padding='same')(x2)
x2 = MaxPool1D(pool_size=3)(x2)
x2 = Conv1D(128, 5, use_bias=True, activation='relu', padding='same')(x2)
x2 = MaxPool1D(pool_size=5)(x2)
x2 = Flatten()(x2)
x2 = Dense(128, activation='relu')(x2)
merge_layer = concatenate([x1, x2], axis=1)
result_layer = Dense(6, activation='softmax')(merge_layer)

model = Model([input_cnn, input_char], result_layer)


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