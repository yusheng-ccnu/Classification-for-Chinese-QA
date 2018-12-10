from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
import numpy as np

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

#这个能生成一个OneHot的10维向量，作为Y_train的一行，这样Y_train就有60000行OneHot作为输出
Y_train = (np.arange(10) == y_train[:, None]).astype(int)  # 整理输出
Y_test = (np.arange(10) == y_test[:, None]).astype(int)    #np.arange(5) = array([0,1,2,3,4])


model.fit(X_train, Y_train, batch_size=128, epochs=100, shuffle=True, verbose=2, validation_split=0.3)

scores = model.evaluate(X_test,Y_test, batch_size=128, verbose=0)
print("")
print("The test loss is %f", scores)
