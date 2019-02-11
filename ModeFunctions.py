import pickle
import numpy as np
import matplotlib.pyplot as plt
from progressbar import *
from LMSFunctions import LMS_function_class

"""ModeFunctions.py: Contains all different  mode functions"""

__author__ = "Almog Zer"

LMS = LMS_function_class()


class mode_functions_class():

    def SNR_mode(self, d, n, k, noise_amp_vec, num_runs=1, num_runs_LMS=1, batch_size=1, polynomial=0):
        """
        SNR_mode function runs train for data with different SNR (fits to noise_amp)
        and returns SNR of the trained data and RMSE of LMS and pseudo inverse method
        :param d:dimension of vector x (x = [1,x(d dimension)])
        :param n: number of wanted vectors
        :param k: the effective dimension of x
        :param noise_amp_vec: vector of noise amplitudes that added to y
        :param num_runs: num of times to run the test algorithm (optional)
        :param num_runs_LMS: number of runs of the LMS process (optional)
        :param batch_size: batch size for LMS process (optional)
        :param polynomial: is it polynomial mode (1) or not (0 - defaulted) (optional)
        :return:
                SNR_sorted: SNR of the train data
                MSE_LMS_sorted: MSE (mean square error) of test data using LMS method
                MSE_PI_sorted: MSE (mean square error) of test data using pseudo inverse method
        """
        # definitions
        MSE_LMS = np.zeros((len(noise_amp_vec), 1))
        MSE_PI = np.zeros((len(noise_amp_vec), 1))

        MSE_LMS_run = np.zeros((num_runs, 1))
        MSE_PI_run = np.zeros((num_runs, 1))

        SNR = np.zeros((len(noise_amp_vec), 1))
        SNR_run = np.zeros((num_runs, 1))

        widgets = ['Processing estimation SNR mode: ', Percentage(), ' ', Bar()]
        max_val = len(noise_amp_vec) * num_runs
        bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()

        for i in range(len(noise_amp_vec)):
            noise_amp = noise_amp_vec[i]
            weights_LMS, weights_PI, SNR[i] = LMS.train(d, n, k, noise_amp=noise_amp, batch_size=batch_size,
                                                        num_runs_LMS=num_runs_LMS)
            for l in range(num_runs):
                MSE_LMS_run[l], MSE_PI_run[l] = LMS.test(d, n, k, weights_LMS, weights_PI, noise_amp, polynomial)
                bar.update(i * num_runs + (l + 1))
            MSE_PI[i] = np.mean(MSE_PI_run)
            MSE_LMS[i] = np.mean(MSE_LMS_run)

        # sorting by SNR
        ind_sort = np.argsort(SNR, axis=0)
        SNR_sorted = np.choose(ind_sort, SNR)
        MSE_LMS_sorted = np.choose(ind_sort, MSE_LMS)
        MSE_PI_sorted = np.choose(ind_sort, MSE_PI)

        bar.finish()

        return SNR_sorted, MSE_LMS_sorted, MSE_PI_sorted

    def batch_size_mode(self, d, n, k, batch_size_vec, num_runs=1, noise_amp=1, num_runs_LMS=1, polynomial=0):
        """
        batch_size_mode function runs train for data with different batch sizes
        and returns RMSE of LMS and pseudo inverse method
        :param d:dimension of vector x (x = [1,x(d dimension)])
        :param n: number of wanted vectors
        :param k: the effective dimension of x
        :param batch_size_vec: vector of batch size for LMS process
        :param num_runs: num of times to run the test algorithm (optional)
        :param num_runs_LMS: number of runs of the LMS process (optional)
        :param noise_amp: noise amplitude that added to y (optional)
        :param polynomial: is it polynomial mode (1) or not (0 - defaulted) (optional)
        :return:
                MSE_LMS: MSE (mean square error) of test data using LMS method
                MSE_PI: MSE (mean square error) of test data using pseudo inverse method
        """
        # definitions
        MSE_LMS = np.zeros((len(batch_size_vec), 1))
        MSE_PI = np.zeros((len(batch_size_vec), 1))

        MSE_LMS_run = np.zeros((num_runs, 1))
        MSE_PI_run = np.zeros((num_runs, 1))

        SNR = np.zeros((len(batch_size_vec), 1))
        SNR_run = np.zeros((num_runs, 1))

        widgets = ['Processing estimation batch size mode: ', Percentage(), ' ', Bar()]
        max_val = len(batch_size_vec) * num_runs
        bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()

        for i in range(len(batch_size_vec)):
            batch_size = batch_size_vec[i]
            weights_LMS, weights_PI, SNR[i] = LMS.train(d, n, k, noise_amp=noise_amp, batch_size=batch_size,
                                                        num_runs_LMS=num_runs_LMS)
            for l in range(num_runs):
                MSE_LMS_run[l], MSE_PI_run[l] = LMS.test(d, n, k, weights_LMS, weights_PI, noise_amp, polynomial)
                bar.update(i * num_runs + (l + 1))
            MSE_PI[i] = np.mean(MSE_PI_run)
            MSE_LMS[i] = np.mean(MSE_LMS_run)

        bar.finish()

        return MSE_LMS, MSE_PI

    def num_LMS_runs_mode(self, d, n, k, num_runs_LMS_vec, num_runs=1, noise_amp=1, batch_size=1, polynomial=0):
        """
        num_LMS_runs_mode function runs train for data with different number of runs of the LMS process
        and returns RMSE of LMS and pseudo inverse method
        :param d:dimension of vector x (x = [1,x(d dimension)])
        :param n: number of wanted vectors
        :param k: the effective dimension of x
        :param num_runs_LMS_vec: vector of number of runs of the LMS process
        :param num_runs: num of times to run the test algorithm (optional)
        :param noise_amp: noise amplitude that added to y (optional)
        :param batch_size: batch size for LMS process (optional)
        :param polynomial: is it polynomial mode (1) or not (0 - defaulted) (optional)
        :return:
                MSE_LMS: MSE (mean square error) of test data using LMS method
                MSE_PI: MSE (mean square error) of test data using pseudo inverse method
        """
        # definitions
        MSE_LMS = np.zeros((len(num_runs_LMS_vec), 1))
        MSE_PI = np.zeros((len(num_runs_LMS_vec), 1))

        MSE_LMS_run = np.zeros((num_runs, 1))
        MSE_PI_run = np.zeros((num_runs, 1))

        SNR = np.zeros((len(num_runs_LMS_vec), 1))

        widgets = ['Processing estimation num LMS runs mode: ', Percentage(), ' ', Bar()]
        max_val = len(num_runs_LMS_vec) * num_runs
        bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()

        for i in range(len(num_runs_LMS_vec)):
            num_runs_LMS = num_runs_LMS_vec[i]
            weights_LMS, weights_PI, SNR[i] = LMS.train(d, n, k, noise_amp=noise_amp, batch_size=batch_size,
                                                        num_runs_LMS=num_runs_LMS)
            for l in range(num_runs):
                MSE_LMS_run[l], MSE_PI_run[l] = LMS.test(d, n, k, weights_LMS, weights_PI, noise_amp, polynomial)
                bar.update(i * num_runs + (l + 1))
            MSE_PI[i] = np.mean(MSE_PI_run)
            MSE_LMS[i] = np.mean(MSE_LMS_run)

        bar.finish()

        return MSE_LMS, MSE_PI

    def n_samples_mode(self, d, n_sample_vec, k, num_runs=1, noise_amp=1, batch_size=1, num_runs_LMS=1, polynomial=0):
        """
        n_samples_mode function runs train for data with different numbers of samples
        and returns RMSE of LMS and pseudo inverse method
        :param d:dimension of vector x (x = [1,x(d dimension)])
        :param n_sample_vec: vector of number of wanted samples
        :param k: the effective dimension of x
        :param num_runs: num of times to run the test algorithm (optional)
        :param batch_size: batch size for LMS process (optional)
        :param num_runs_LMS: number of runs of the LMS process (optional)
        :param noise_amp: noise amplitude that added to y (optional)
        :param polynomial: is it polynomial mode (1) or not (0 - defaulted) (optional)
        :return:
                MSE_LMS: MSE (mean square error) of test data using LMS method
                MSE_PI: MSE (mean square error) of test data using pseudo inverse method
        """
        # definitions
        MSE_LMS = np.zeros((len(n_sample_vec), 1))
        MSE_PI = np.zeros((len(n_sample_vec), 1))

        MSE_LMS_run = np.zeros((num_runs, 1))
        MSE_PI_run = np.zeros((num_runs, 1))

        SNR = np.zeros((len(n_sample_vec), 1))
        SNR_run = np.zeros((num_runs, 1))

        widgets = ['Processing estimation number samples mode: ', Percentage(), ' ', Bar()]
        max_val = len(n_sample_vec) * num_runs
        bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()

        for i in range(len(n_sample_vec)):
            n = int(n_sample_vec[i])
            weights_LMS, weights_PI, SNR[i] = LMS.train(d, n, k, noise_amp=noise_amp, batch_size=batch_size,
                                                        num_runs_LMS=num_runs_LMS)
            for l in range(num_runs):
                MSE_LMS_run[l], MSE_PI_run[l] = LMS.test(d, n, k, weights_LMS, weights_PI, noise_amp, polynomial)
                bar.update(i * num_runs + (l + 1))
            MSE_PI[i] = np.mean(MSE_PI_run)
            MSE_LMS[i] = np.mean(MSE_LMS_run)

        bar.finish()

        return MSE_LMS, MSE_PI

    def effective_dimension_mode(self, d, n, k_vec, num_runs=1, noise_amp=1, batch_size=1, num_runs_LMS=1,
                                 polynomial=0):
        """
        effective_dimension_mode function runs train for data with different effective dimension
        and returns RMSE of LMS and pseudo inverse method
        :param d:dimension of vector x (x = [1,x(d dimension)])
        :param n: number of wanted samples
        :param k_vec: vector of the different effective dimensions of x
        :param num_runs: num of times to run the test algorithm (optional)
        :param batch_size: batch size for LMS process (optional)
        :param num_runs_LMS: number of runs of the LMS process (optional)
        :param noise_amp: noise amplitude that added to y (optional)
        :param polynomial: is it polynomial mode (1) or not (0 - defaulted) (optional)
        :return:
                MSE_LMS: MSE (mean square error) of test data using LMS method
                MSE_PI: MSE (mean square error) of test data using pseudo inverse method
        """
        # definitions
        MSE_LMS = np.zeros((len(k_vec), 1))
        MSE_PI = np.zeros((len(k_vec), 1))

        MSE_LMS_run = np.zeros((num_runs, 1))
        MSE_PI_run = np.zeros((num_runs, 1))

        SNR = np.zeros((len(k_vec), 1))
        SNR_run = np.zeros((num_runs, 1))

        widgets = ['Processing estimation effective dimension mode: ', Percentage(), ' ', Bar()]
        max_val = len(k_vec) * num_runs
        bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()

        for i in range(len(k_vec)):
            k = int(k_vec[i])
            weights_LMS, weights_PI, SNR[i] = LMS.train(d, n, k, noise_amp=noise_amp, batch_size=batch_size,
                                                        num_runs_LMS=num_runs_LMS)
            for l in range(num_runs):
                MSE_LMS_run[l], MSE_PI_run[l] = LMS.test(d, n, k, weights_LMS, weights_PI, noise_amp, polynomial)
                bar.update(i * num_runs + (l + 1))
            MSE_PI[i] = np.mean(MSE_PI_run)
            MSE_LMS[i] = np.mean(MSE_LMS_run)

        bar.finish()

        return MSE_LMS, MSE_PI

    def step_size_mode(self, d, n, k, mu_vec, num_runs=1, noise_amp=1, batch_size=1, num_runs_LMS=1, polynomial=0):
        """
        step_size_mode function runs train for data with different effective dimension
        and returns RMSE of LMS and pseudo inverse method
        :param d:dimension of vector x (x = [1,x(d dimension)])
        :param n: number of wanted samples
        :param k: effective dimensions of x
        :param mu_vec: vector of all different weights step size (changing weights each time with error*mu)
        :param num_runs: num of times to run the test algorithm (optional)
        :param batch_size: batch size for LMS process (optional)
        :param num_runs_LMS: number of runs of the LMS process (optional)
        :param noise_amp: noise amplitude that added to y (optional)
        :param polynomial: is it polynomial mode (1) or not (0 - defaulted) (optional)
        :return:
                MSE_LMS: MSE (mean square error) of test data using LMS method
                MSE_PI: MSE (mean square error) of test data using pseudo inverse method
        """
        # definitions
        MSE_LMS = np.zeros((len(mu_vec), 1))
        MSE_PI = np.zeros((len(mu_vec), 1))

        MSE_LMS_run = np.zeros((num_runs, 1))
        MSE_PI_run = np.zeros((num_runs, 1))

        SNR = np.zeros((len(mu_vec), 1))
        SNR_run = np.zeros((num_runs, 1))

        widgets = ['Processing estimation weight step size mode: ', Percentage(), ' ', Bar()]
        max_val = len(mu_vec) * num_runs
        bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()

        for i in range(len(mu_vec)):
            mu = mu_vec[i]
            weights_LMS, weights_PI, SNR[i] = LMS.train(d, n, k, mu=mu, noise_amp=noise_amp, batch_size=batch_size,
                                                        num_runs_LMS=num_runs_LMS)
            for l in range(num_runs):
                MSE_LMS_run[l], MSE_PI_run[l] = LMS.test(d, n, k, weights_LMS, weights_PI, noise_amp, polynomial)
                bar.update(i * num_runs + (l + 1))
            MSE_PI[i] = np.mean(MSE_PI_run)
            MSE_LMS[i] = np.mean(MSE_LMS_run)

        bar.finish()

        return MSE_LMS, MSE_PI
