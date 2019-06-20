import tensorflow as tf
from util.data_preprocess import load_data
from util.transform_martrix import transform_to_matrix
from keras.utils import to_categorical
import numpy as np


def model(data_placegolder, drop_placeholser):
    conv0 = tf.layers.conv1d(data_placegolder, 40, 3, activation=tf.nn.relu)
    pool0 = tf.layers.max_pooling1d(conv0, 2, 1)

    conv1 = tf.layers.conv1d(pool0, 20, 3, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling1d(conv1, 2, 1)

    flatten = tf.layers.flatten(pool1)
    fc = tf.layers.dense(flatten, 100, activation=tf.nn.relu)
    dp = tf.layers.dropout(fc, drop_placeholser)

    logits = tf.layers.dense(dp, 6)
    return logits


#数据获取
x_train, y_train = load_data('../set/train_seg.txt')
datas = np.asarray(transform_to_matrix(X=x_train, vec_size=300))
labels = to_categorical(y_train)
print('datas shape', datas.shape)
print('labels shape', labels.shape)

x_test, y_test = load_data('../set/test_seg.txt')
x_test = np.asarray(transform_to_matrix(X=x_test, vec_size=300))
y_test = to_categorical(y_test)
print('datas shape', x_test.shape)
print('labels shape', y_test.shape)

datas_placeholder = tf.placeholder(tf.float32, [None, datas.shape[1], datas.shape[2]])
labels_placeholder = tf.placeholder(tf.int32, [None, 6])
dropout_placeholdr = tf.placeholder(tf.float32)
batch_size = tf.placeholder(tf.int32)

logits = model(datas_placeholder, dropout_placeholdr)
#损失函数
losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=labels_placeholder,
    logits=logits
)
mean_loss = tf.reduce_mean(losses)
predicted_labels = tf.argmax(logits, 1)
accu = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)), tf.float32))
test_accu = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, 1), tf.argmax(logits, 1)), tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(losses)
saver = tf.train.Saver()

train = False
with tf.Session() as sess:
    if train:
        print("训练模型")
        sess.run(tf.global_variables_initializer())
        train_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0.25,
            batch_size: 32
        }

        for step in range(300):
            _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)
            acc = sess.run(accu, feed_dict=train_feed_dict)
            if step % 10 == 0:
                print("train accuracy = {}\tmean loss = {}".format(acc, mean_loss_val))

        saver.save(sess, '../model/cnn_model')
    else:
        saver.restore(sess, '../model/cnn_model')
        test_feed_dict = {
            datas_placeholder: x_test,
            labels_placeholder: y_test,
            dropout_placeholdr: 0,
            batch_size: 32
        }
        test_acc = sess.run(test_accu, feed_dict=test_feed_dict)
        print("test accuracy = {}".format(test_acc))