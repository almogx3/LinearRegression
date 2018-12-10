import pickle
import numpy as np
import matplotlib.pyplot as plt
from progressbar import *

"""LMSFunctions.py: Contains all LMS function"""

__author__ = "Almog Zer"


class LMS_function_class():
    def create_cov_matrix(self, d, k):
        """
        :param d: dimension of vector x (x = [1,x(d dimension)])
        :param k: the effective dimension of x
        :return: diagonal cov matrix, where lambda_j ~ 0 for j>k
        """
        random_values = np.random.rand(d, 1)
        random_values[k:] = random_values[k:] / 100000
        random_values = random_values.reshape((random_values.shape[0],))
        R = np.diag(random_values)
        return R

    def create_random_vectors(self, num_of_vectors=100, cov_matrix=np.eye(100), mean=np.zeros((100,))):
        """
        create_random_vectors creates random vectors using cov matrix
        :param num_of_vectors: number of wanted vectors
        :param cov_matrix: covariance matrix of vectors
        :param mean: mean vector of the created vectors
        :return: x: random vector matrix dimension (mean dim)X(num_of_vectors)
                 y: desired result for random vector (1 or 0) size (num_of_vectors)X1
        """
        x = np.random.multivariate_normal(mean, cov_matrix, num_of_vectors).T
        # x matrix dimension (mean dim)X(num_of_vectors)
        # y will be if sum(x)>0
        sum_x = np.sum(x, axis=0)
        y = sum_x
        # adding 1 to all x
        x = np.vstack((np.ones((1, num_of_vectors)), x))
        return x, y

    def LMS(self, x, y, step=1, weights=None, eps_value=5e-3):
        """
        LMS runs LMS algorithm one time on x,y
        :param x: input signal vector
        :param y: desired signal vector
        :param step: step size (filter length)
        :param weights: starting point for weights (optional)
        :param eps_value: weights step size (changing weights each time with error*eps_value)
        :return: weights: final weights from LMS process
        """
        if weights is None:
            weights = np.zeros((x.shape[0],))
        for ind in np.arange(0, len(y), step):
            x_current = x[:, ind:(ind + step)]
            error_current = y[ind:(ind + step)] - np.matmul(x_current.T, weights)
            error_eps = (error_current * eps_value)
            weights = weights + np.dot(x_current, error_eps)

        return weights

    def LMS_multiple_times(self, x, y, step=1, eps_value=5e-3, num_runs=1):
        """
        LMS_multiple_times runs LMS process multiple times according to num_runs
        :param x: input signal vector
        :param y: desired signal vector
        :param step: step size (filter length)
        :param eps_value: weights step size (changing weights each time with error*eps_value)
        :param num_runs: number if times to run LMS process
        :return:
                weights: final weights from LMS process
        """
        weights = self.LMS(x, y, step=step, eps_value=eps_value)
        for counts in range(num_runs - 1):
            weights = self.LMS(x, y, step=step, weights=weights, eps_value=eps_value)
        #     option to change for to while and add converge condition (using error or weights change)
        return weights

    def train(self, d, n, k, noise_amp=1, batch_size=1, num_runs_LMS=1):
        """
        train function create data fits to d, n, k, noise amp,
        and trains it using LMS and pseudo inverse (PI) method
        :param d: dimension of vector x (x = [1,x(d dimension)])
        :param n: number of wanted vectors
        :param k: the effective dimension of x
        :param noise_amp: amplitude of the noise added to y (optional)
        :param batch_size: batch size for LMS process (optional)
        :param num_runs_LMS: number of times to run the LMS process (optional)
        :return: trained weights and SNR
                weights_LMS: trained weights using LMS method
                weights_PI: trained weights using pseudo inverse method
                SNR: SNR of the train data (fits to noise_amp)
        """
        R = self.create_cov_matrix(d, k)
        x, y = self.create_random_vectors(n, R, np.zeros((d,)))
        y_real = y
        y = y + noise_amp * np.random.rand(y.shape[0], )
        SNR = 10 * np.log10(np.linalg.norm(y_real) / np.linalg.norm((np.abs(y - y_real))))  # SNR in db
        weights_LMS = self.LMS_multiple_times(x, y, step=batch_size, num_runs=num_runs_LMS)
        weights_PI = np.matmul(np.matmul(x, np.linalg.inv(np.matmul(x.T, x))), y)

        return weights_LMS, weights_PI, SNR

    def test(self, d, n, k, weights_LMS, weights_PI):
        """
        test function creates test data and test weights of LMS and Pseudo inverse algorithm
        :param d:dimension of vector x (x = [1,x(d dimension)])
        :param n: number of wanted vectors
        :param k: the effective dimension of x
        :param weights_LMS: weights of the train data using LMS method
        :param weights_PI: weights of the train data using pseudo inverse (PI) method
        :return:
                RMSE_LMS: RMSE (root mean square error) of LMS method
                RMSE_PI: RMSE (root mean square error) of pseudo inverse method
        """
        R = self.create_cov_matrix(d, k)
        x, y = self.create_random_vectors(n, R, np.zeros((d,)))
        y_PI = np.matmul(x.T, weights_PI)
        y_LMS = np.matmul(x.T, weights_LMS)
        RMSE_LMS = np.mean((y - y_LMS) ** 2) ** 0.5
        RMSE_PI = np.mean((y - y_PI) ** 2) ** 0.5

        return RMSE_LMS, RMSE_PI
