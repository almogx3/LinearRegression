import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from progressbar import *
from LMSFunctions import LMS_function_class

# Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 20
display_step = 50

# data parameters
# x dimensions
d = 1000
# number of samples
n = 100
# The effective dimension
k = 10
# noise amp
noise_amp = 1

# Network parameters
n_hidden_1 = 3
n_output_classes = 1
n_input = d

LMS = LMS_function_class()
R = LMS.create_cov_matrix(d, k)
train_x, train_y = LMS.create_random_vectors(n, R, np.zeros((d,)))
train_x = train_x[1:, :].T
train_y = train_y + noise_amp * np.random.rand(train_y.shape[0], )

# tf Graph Input
X = tf.placeholder("float64", shape=(n, d))
Y = tf.placeholder("float64", shape=(n, ))

# set weights and bias
W = tf.Variable(np.random.randn(d, 1), name="weight", )
b = tf.Variable(np.random.randn(1, 1), name="bias")

pred = tf.add(tf.linalg.matmul(X, W), b)

# MSE
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / n

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize variables
init = tf.global_variables_initializer()

# Start training

with tf.Session() as sess:
    # initialize
    sess.run(init)

    # Fit all training data
    for step in range(num_steps):
        if step % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_x, Y: train_y})
            print("step", '%04d' % step, "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b", sess.run(b))

    print("optimization Finished")
    training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Testing example, as requested (Issue #2)
    test_x, test_y = LMS.create_random_vectors(n, R, np.zeros((d,)))
    test_x = test_x[1:, :].T
    test_y = test_y + noise_amp * np.random.rand(test_y.shape[0], )
    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_mean(tf.pow(pred - Y, 2)),
        feed_dict={X: test_x, Y: test_y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

print("done")
