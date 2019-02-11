import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from progressbar import *
from LMSFunctions import LMS_function_class


def build_model(d, learning_rate=1e-3):
    """
    build_model builds model for input size d
    :param d: input shape
    :param learning_rate: model learning rate
    :return:
            model
    """
    model = tf.keras.Sequential([tf.layers.Dense(1, input_shape=[d])])
    # model.add(Dense(output, batch_input_shape=(None, 500)))
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    return model


def LMS_layers(d=1000, n=100, k=10, noise_amp=1, batch_size=1, total_runs=100):
    learning_rate = 1e-3
    model = build_model(d, learning_rate)
    model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # create training data
    LMS = LMS_function_class()
    R = LMS.create_cov_matrix(d, k)
    train_x, train_y = LMS.create_random_vectors(n, R, np.zeros((d,)))
    real_y = train_y
    train_y = train_y + noise_amp * np.random.rand(train_y.shape[0], )
    W_PI = np.matmul(np.matmul(train_x, np.linalg.inv(np.matmul(train_x.T, train_x))), train_y)
    train_x = train_x[1:, :]
    # train process
    print('start training process')
    num_steps = 1000
    history = model.fit(train_x.T, train_y, epochs=num_steps, batch_size=batch_size, callbacks=[early_stop])
    weights = model.weights

    MSE_LMS = np.zeros((total_runs, 1))
    MSE_PI = np.zeros((total_runs, 1))
    SNR = 10 * np.log10(np.linalg.norm(real_y) / np.linalg.norm((np.abs(train_y - real_y))))  # SNR in db
    for i in range(total_runs):
        test_x, test_y = LMS.create_random_vectors(n, R, np.zeros((d,)))
        test_y = test_y + noise_amp * np.random.rand(train_y.shape[0], )
        y_PI = np.matmul(test_x.T, W_PI)
        test_x = test_x[1:, :].T
        loss, mse = model.evaluate(test_x, test_y)
        pred = model.predict(test_x).flatten()
        MSE_LMS[i] = mse
        MSE_PI[i] = np.mean((test_y - y_PI) ** 2)

    return SNR, np.mean(MSE_LMS), np.mean(MSE_PI)


def calc_layer_weights(x, n_in, n_out):
    """
    calc_layer_weights calculates the weights one time
    :param x: input data
    :param n_in: number of inputs (dimension of the input data)
    :param n_out: number of outputs (dimension of the output data)
    :return:
            predict: the predicted result
    """
    W = tf.Variable(tf.zeros((n_in, n_out)))
    b = tf.Variable(tf.zeros((1, n_out)))
    predict = tf.add(tf.linalg.matmul(x, W), b)

    return predict


