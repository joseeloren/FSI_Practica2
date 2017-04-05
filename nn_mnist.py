# -*- coding: utf-8 -*-
import gzip
import pickle

import tensorflow as tf
import numpy as np


def xrange(*args, **kwargs):
    return iter(range(*args, **kwargs))

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f)
f.close()

# TODO: the neural net!!

x_train, y_train = train_set
x_valid, y_valid = valid_set
x_test, y_test = test_set

y_train = one_hot(y_train,10)
y_valid = one_hot(y_valid,10)
y_test = one_hot(y_test,10)

x = tf.placeholder("float", [None, 28*28])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(28*28, 10*10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10*10)) * 0.1)
W2 = tf.Variable(np.float32(np.random.rand(10*10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
# info https://db-blog.web.cern.ch/blog/luca-canali/2016-07-neural-network-scoring-engine-plsql-recognizing-handwritten-digits
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)  # learning rate: 0.01

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 100

error = 1;
epoch = 0;
last_error =-1;
import matplotlib.pyplot as plt
array = []
while 1:
    for jj in xrange((int)(len(x_train) / batch_size)):
        batch_xs = x_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error = sess.run(loss, feed_dict={x: x_valid, y_: y_valid})
    array.append(error)



    print ("Epoch #:", epoch, "Error: ", error)
    epoch += 1
    if (abs(error - last_error)) < 0.001:
        break

    last_error = error

print ("---------------------------------Test set-----------------------------------------")
#result = sess.run(y, feed_dict={x: x_test})
#count = 0
#for b, r in zip(y_test, result):
#    count+=1
#    print(b, "-->", r)
#    if count == 20:
#        break
error = sess.run(loss, feed_dict={x: x_test, y_: y_test})
print 'Error =', error
print("----------------------------------------------------------------------------------")
plt.plot(array)
print array
plt.show()  # Let's see a sample