import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from progressbar import *
from LMSFunctions import LMS_function_class


def calc_layer_weights(x, n_in, n_out):
    """
    calc_layer_weights calculates the weights one time
    :param x: input data
    :param n_in: number of inputs (dimension of the input data)
    :param n_out: number of outputs (dimension of the output data)
    :return:
            predict: the predicted result
    """
    W = tf.Variable(tf.zeros((n_in,n_out)))
    b = tf.Variable(tf.zeros((1, n_out)))
    predict = tf.add(tf.linalg.matmul(x, W), b)

    return predict

def LMS_TF_one_neuron(d=1000, n=100, k=10, noise_amp=1, batch_size=20, total_runs=100):
    # Parameters
    learning_rate = 8e-3
    num_steps = 1000
    display_step = 50

    # data parameters
    # x dimensions
    d = 1000
    # number of samples
    n = 100
    # The effective dimension
    k = 10



    # Network parameters
    n_hidden_1 = 3
    n_output_classes = 1
    n_input = d

    LMS = LMS_function_class()
    R = LMS.create_cov_matrix(d, k)
    train_x, train_y = LMS.create_random_vectors(n, R, np.zeros((d,)))
    train_x = train_x[1:, :].T
    real_y = train_y
    train_y = train_y + noise_amp * np.random.rand(train_y.shape[0], )

    # Pseudo inverse calculations
    train_x_PI = np.column_stack((np.ones((n, 1)), train_x))
    W_PI = np.matmul(np.matmul(train_x_PI.T, np.linalg.inv(np.matmul(train_x_PI, train_x_PI.T))), train_y)
    SNR = 10 * np.log10(np.linalg.norm(real_y) / np.linalg.norm((np.abs(train_y - real_y))))  # SNR in db

    # tf Graph Input
    X = tf.placeholder("float64", shape=(n, d))
    Y = tf.placeholder("float64", shape=(n,))

    # set weights and bias
    # W = tf.Variable(np.random.randn(d, 1), name="weight", )
    # b = tf.Variable(np.random.randn(1, 1), name="bias")
    W = tf.Variable(np.zeros((d, 1)), name="weight", )
    b = tf.Variable(np.zeros((1, 1)), name="bias")

    pred = tf.add(tf.linalg.matmul(X, W), b)

    # MSE
    cost = tf.reduce_mean(tf.pow(pred - Y, 2))

    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize variables
    init = tf.global_variables_initializer()

    # Start training
    costs_vec_neuron = np.zeros((total_runs,))
    costs_vec_PI = np.zeros((total_runs,))
    with tf.Session() as sess:
        # initialize
        sess.run(init)

        # Fit all training data
        for step in range(num_steps):
            if step % display_step == 0:
                c = sess.run(cost, feed_dict={X: train_x, Y: train_y})
                # print("step", '%04d' % step, "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b", sess.run(b))

        print("optimization Finished")
        training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y})
        # print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
        for num_run in range(total_runs):
            test_x, test_y = LMS.create_random_vectors(n, R, np.zeros((d,)))
            test_x = test_x[1:, :].T
            test_y = test_y + noise_amp * np.random.rand(test_y.shape[0], )
            testing_cost = sess.run(
                tf.reduce_mean(tf.pow(pred - Y, 2)),
                feed_dict={X: test_x, Y: test_y})  # same function as cost above
            costs_vec_neuron[num_run] = testing_cost
            test_x_pi = np.column_stack((np.ones((n, 1)), test_x))
            y_PI = np.matmul(test_x_pi, W_PI)
            costs_vec_PI[num_run] = np.mean((test_y - y_PI) ** 2)

    return SNR, np.mean(costs_vec_neuron), np.mean(costs_vec_PI)
