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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

x_train = x_data[0:(int)(0.7*len(x_data)), :]
y_train = y_data[0:(int)(0.7*len(x_data)), :]

x_valid = x_data[(int)(0.7*len(x_data)):(int)(0.85*len(x_data)), :]
y_valid = y_data[(int)(0.7*len(x_data)):(int)(0.85*len(x_data)), :]

x_test = x_data[(int)(0.85*len(x_data)):, :]
y_test = y_data[(int)(0.85*len(x_data)):, :]


x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01
evaluation = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
acu = tf.reduce_mean(tf.cast(evaluation,tf.float32))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 20

error = 2.0;
epoch = 0;
last_error = 1000
array = []
import matplotlib.pyplot as plt
while 1:
    for jj in xrange((int)(len(x_train) / batch_size)):
        batch_xs = x_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error = sess.run(loss, feed_dict={x: x_valid, y_: y_valid})
    print ("Epoch #:", epoch, "Error: ", error)
    epoch += 1
    array.append(error)
    #result = sess.run(y, feed_dict={x: x_valid})
    #for b, r in zip(y_valid, result):
     #   print (b, "-->", r)
    #print ("----------------------------------------------------------------------------------")
    if  abs(last_error - error) < 0.0001 and epoch>10:
        break

    last_error = error

print ("---------------------------------Test set-----------------------------------------")
result = sess.run(y, feed_dict={x: x_test})
ok = 0
for b, r in zip(y_test, result):
    if b.argmax() == r.argmax():
        ok += 1

acuu = sess.run(acu,feed_dict={x: x_test, y_: y_test})
error = sess.run(loss, feed_dict={x: x_test, y_: y_test})
print ('Error =', error)
print ('Porcentaje aciertos = ', acuu*100, '%')
print("----------------------------------------------------------------------------------")
plt.plot(array)
plt.ylabel('Error calculado por TensorFlow')
plt.xlabel('Épocas realizadas')
plt.show()  # Let's see a sample