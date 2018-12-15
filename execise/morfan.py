from keras.layers import SimpleRNN, Activation, Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras import initializers
import numpy as np

INPUT_SIZE = 28
TIME_STEP = 28
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.01

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
x_train = x_train.reshape(-1, 28, 28)
print(x_train.shape)
x_test = x_test.reshape(-1, 28, 28)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(y_test.shape[0])

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(SimpleRNN(50,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu'))
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))
adam = Adam(LR)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

for step in range(60000):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = x_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= x_train.shape[0] else BATCH_INDEX
    if step % 500 == 0:
        cost, accuracy = model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)

