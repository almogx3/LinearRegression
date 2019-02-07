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
                 y: desired result for random vector size (num_of_vectors)X1
        """
        x = np.random.multivariate_normal(mean, cov_matrix, num_of_vectors).T
        # x matrix dimension (mean dim)X(num_of_vectors)
        sum_x = np.sum(x, axis=0)
        y = sum_x
        # adding 1 to all x
        x = np.vstack((np.ones((1, num_of_vectors)), x))
        return x, y

    def create_random_vectors_polynomial(self, num_of_vectors=100, cov_matrix=np.eye(100), mean=np.zeros((100,)),
                                         d=100):
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
        # x = self.x_to_power_vector(x, d)
        for i in range(len(x)):
            x[i] = x[i] ** i
        y = np.sum(x, axis=0)
        return x, y

    def create_x_y_power_vector(self, cov_matrix=np.eye(100), mean=np.zeros((100,)),
                                d=100):
        """
        create_x_power_vector creates vector x in dimension d.
        :param cov_matrix: covariance matrix of vectors
        :param mean: mean vector of the created vectors
        :param d: d is the maximum polynomial degree
        :return:
                x_mat: vector of powers of x [[x^0] ,[x^1],...[x^d]]
                y: sum rows (x_mat)
        """
        x = np.random.multivariate_normal(mean, cov_matrix, 1).T
        x_mat = np.power(x, range(d + 1))
        y = np.sum(x_mat, axis=1)
        return x_mat

    def LMS(self, x, y, batch=1, weights=None, mu=5e-3):
        """
        LMS runs LMS algorithm one time on x,y
        :param x: input signal vector
        :param y: desired signal vector
        :param batch: batch size (filter length)
        :param weights: starting point for weights (optional)
        :param mu: weights step size (changing weights each time with error*mu)
        :return: weights: final weights from LMS process
        """
        if weights is None:
            weights = np.zeros((x.shape[0],))
        for ind in np.arange(0, len(y), batch):
            x_current = x[:, ind:(ind + batch)]
            error_current = y[ind:(ind + batch)] - np.matmul(x_current.T, weights)
            error_eps = (error_current * mu)
            weights = weights + np.dot(x_current, error_eps)

        return weights

    def LMS_multiple_times(self, x, y, batch=1, mu=1e-3, num_runs=1):
        """
        LMS_multiple_times runs LMS process multiple times according to num_runs
        :param x: input signal vector
        :param y: desired signal vector
        :param batch: batch size (filter length)
        :param mu: weights step size (changing weights each time with error*mu)
        :param num_runs: number if times to run LMS process
        :return:
                weights: final weights from LMS process
        """
        weights = self.LMS(x, y, batch=batch, mu=mu)
        weights_old = weights
        for counts in range(num_runs - 1):
            weights = self.LMS(x, y, batch=batch, weights=weights, mu=mu)
            if np.array_equal(weights, weights_old):
                break
            weights_old = weights
        #     option to change for to while and add converge condition (using error or weights change)
        return weights

    def train(self, d, n, k, mu=8e-3, noise_amp=1, batch_size=1, num_runs_LMS=1, polynomial=0):
        """
        train function create data fits to d, n, k, noise amp,
        and trains it using LMS and pseudo inverse (PI) method
        :param d: dimension of vector x (x = [1,x(d dimension)])
        :param n: number of wanted vectors
        :param k: the effective dimension of x
        :param mu: weights step size (changing weights each time with error*mu)
        :param noise_amp: amplitude of the noise added to y (optional)
        :param batch_size: batch size for LMS process (optional)
        :param num_runs_LMS: number of times to run the LMS process (optional)
        :param polynomial: is it polynomial mode (1) or not (0 - defaulted) (optional)
        :return: trained weights and SNR
                weights_LMS: trained weights using LMS method
                weights_PI: trained weights using pseudo inverse method
                SNR: SNR of the train data (fits to noise_amp)
        """
        R = self.create_cov_matrix(d, k)
        if polynomial == 0:
            x, y = self.create_random_vectors(n, R, np.zeros((d,)))
        else:
            x, y = self.create_random_vectors_polynomial(n, R, np.zeros((d,)))
        y_real = y
        y = y + noise_amp * np.random.rand(y.shape[0], )
        SNR = 10 * np.log10(np.linalg.norm(y_real) / np.linalg.norm((np.abs(y - y_real))))  # SNR in db
        weights_LMS = self.LMS_multiple_times(x, y, batch=batch_size, num_runs=num_runs_LMS, mu=mu)
        weights_PI = np.matmul(np.matmul(x, np.linalg.inv(np.matmul(x.T, x))), y)

        return weights_LMS, weights_PI, SNR

    def test(self, d, n, k, weights_LMS, weights_PI, noise_amp=1, polynomial=0):
        """
        test function creates test data and test weights of LMS and Pseudo inverse algorithm
        :param d:dimension of vector x (x = [1,x(d dimension)])
        :param n: number of wanted vectors
        :param k: the effective dimension of x
        :param weights_LMS: weights of the train data using LMS method
        :param weights_PI: weights of the train data using pseudo inverse (PI) method
        :param noise_amp: amplitude of the noise added to y (optional)
        :param polynomial: is it polynomial mode (1) or not (0 - defaulted) (optional)
        :return:
                MSE_LMS: MSE (mean square error) of LMS method
                MSE_PI: MSE (mean square error) of pseudo inverse method
        """
        R = self.create_cov_matrix(d, k)
        if polynomial == 0:
            x, y = self.create_random_vectors(n, R, np.zeros((d,)))
        else:
            x, y = self.create_random_vectors_polynomial(n, R, np.zeros((d,)))
        # y = y + noise_amp * np.random.rand(y.shape[0], )
        y_PI = np.matmul(x.T, weights_PI)
        y_LMS = np.matmul(x.T, weights_LMS)
        MSE_LMS = np.mean((y - y_LMS) ** 2)
        MSE_PI = np.mean((y - y_PI) ** 2)

        return MSE_LMS, MSE_PI
