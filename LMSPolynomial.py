import pickle
import numpy as np
import matplotlib.pyplot as plt
from progressbar import *
from LMSFunctions import LMS_function_class

"""LMSPolynom.py: Contains all LMS to create polynomial functions"""
__author__ = "Almog Zer"

LMS = LMS_function_class

class LMS_polynomial():

    def train(self, d, n, k, noise_amp=1, batch_size=1, num_runs_LMS=1):
        """
        train function creates data fits to d, n, k, noise amp,
        and trains it to fitting polynomial to data
        using LMS and pseudo inverse (PI) method
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
        R = LMS.create_cov_matrix(d, k)
        x, y = self.create_random_vectors_polynomial(n, R, np.zeros((d,)),d)
        y_real = y
        y = y + noise_amp * np.random.rand(y.shape[0], )
        SNR = 10 * np.log10(np.linalg.norm(y_real) / np.linalg.norm((np.abs(y - y_real))))  # SNR in db
        weights_LMS = LMS.LMS_multiple_times(x, y, step=batch_size, num_runs=num_runs_LMS)
        weights_PI = np.matmul(np.matmul(x, np.linalg.inv(np.matmul(x.T, x))), y)

        return weights_LMS, weights_PI, SNR


    def create_random_vectors_polynomial(self, num_of_vectors=100, cov_matrix=np.eye(100), mean=np.zeros((100,)), d):
        """
        create_random_vectors_polynomial : creates x and y fitting to x cov matrix
        and d dimension and polynomial with d degree.
        :param num_of_vectors: number of wanted vectors
        :param cov_matrix: covariance matrix of vectors
        :param mean: mean vector of the created vectors
        :param d: d is the maximum polynomial degree
        :return: x: random vector  and it powers (d degrees) matrix dimension NX(num_of_vectors),
                    N = (mean dim) X (d+1)
                 y: desired result for random vector (num_of_vectors)X1
        """
        x = np.random.multivariate_normal(mean, cov_matrix, num_of_vectors).T
        # adding 1 to all x
        x = np.vstack((np.ones((1, num_of_vectors)), x))
        x = self.x_to_power_vector(x, d)
        y = np.sum(x, axis=0)

        return x, y


    def x_to_power_vector(self, x, d):
        """
        x_to_power_vector creates vector of power of x.
        :param x: x is vector mx1
        :param d: d is the maximum polynomial degree
        :return: x_new: vector of powers of x [[x^0] ,[x^1],...[x^d]]
        """
        x_new = np.array([])
        for power_count in range(d + 1):
            x_new = np.concatenate((x_new, np.power(x, power_count)), axis=0)
        return x_new


poly = LMS_polynomial()

