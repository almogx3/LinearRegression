import pickle
import numpy as np
import matplotlib.pyplot as plt
from progressbar import *
from ModeFunctions import mode_functions_class


mfc = mode_functions_class()

def PCAprocess(x):
    """
    PCAprocess executes PCA process
    :param x: training vectors, data set, x[:,i] is vector i
    :return: eigenvalues - sorted high to low
             eigenvectors - The column v[:, i] is the normalized eigenvector
             corresponding to the eigenvalue i (sorted high to low)
             S - covariance matrix of x
    """
    mean_x = np.mean(x, axis=1).T
    mean_x = mean_x.reshape(mean_x.shape[0], 1)
    x_around_mean = x - mean_x
    S = np.matmul(x_around_mean, x_around_mean.T) / x_around_mean.shape[1]

    eigenvalues, eigenvectors = np.linalg.eigh(S)
    index_bottom_to_top = np.argsort(eigenvalues)
    index_top_to_buttom = index_bottom_to_top[-1::-1]
    index_top_to_buttom = index_top_to_buttom.reshape((index_top_to_buttom.shape[0],))
    eigenvalues_sorted = eigenvalues[index_top_to_buttom]
    eigenvectors_sorted = eigenvectors[:, index_top_to_buttom]

    return eigenvalues_sorted, eigenvectors_sorted, S


def projection(x, k, eigenvectors):
    """
    projection returns projection of x
    :param x: training vectors
    :param k:  the effective dimension of x
    :param eigenvectors: eigenvectors sorted from high to low eigenvalues
    :return: projection of x, k dimension
             eigenvectors - first k'th eigenvectors
    """
    eigenvectors_current = eigenvectors[:, 0:k]
    projcetion_x = np.matmul(eigenvectors_current.T, x)

    return projcetion_x, eigenvectors_current


def plot(x, y, color='g', title=None, xlabel=None, ylabel=None):
    plt.figure()
    plt.plot(x, y, color)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    if not title is None:
        plt.title(title)
    if not xlabel is None:
        plt.xlabel(xlabel)
    if not ylabel is None:
        plt.ylabel(ylabel)
    plt.show()


def main():
    plt.interactive(True)
    # x dimensions
    d = 1000
    # number of samples
    n = 100
    # The "real dimension
    k = 10
    # number of runs
    num_runs = 10

    # different SNR mode
    # num_SNR = 10
    # noise_amp_vec = np.concatenate([np.arange(num_SNR), np.arange(num_SNR) * 10])
    # noise_amp_vec += np.ones_like(noise_amp_vec)
    noise_amp_vec = [1, 10, 50, 100, 500, 1000, 1500, 2000, 10000]
    SNR, RMSE_LMS_SNR, RMSE_PI_SNR = mfc.SNR_mode(d, n, k, noise_amp_vec, num_runs=num_runs, batch_size=1)

    # plotting the results SNR
    plot(SNR, RMSE_LMS_SNR, color='g', title='RMSEs as function of SNR', xlabel='SNR [dB]',
         ylabel='Mean Root Mean Square RMSE')
    plt.plot(SNR, RMSE_PI_SNR, color='b')
    plt.legend(('weights LMS', 'weights pseudo inverse'), loc='upper right')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\effective LMS\figures\SNR_RMSE.pickle', 'wb'))

    # batch size mode
    batch_size = [1, 2, 5, 10, 50, 100]
    RMSE_LMS_batch, RMSE_PI_batch = mfc.batch_size_mode(d, n, k, batch_size_vec=batch_size, num_runs=num_runs, noise_amp=5)

    # plotting the results batch size
    plot(batch_size, RMSE_LMS_batch, color='g', title='RMSEs as function of Batch size', xlabel='Batch size',
         ylabel='Mean Root Mean Square RMSE')
    plt.plot(batch_size, RMSE_PI_batch, color='b')
    plt.legend(('weights LMS', 'weights pseudo inverse'), loc='upper right')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\effective LMS\figures\Batch_RMSE.pickle', 'wb'))

    # num of runs LMS
    num_runs_LMS_vec = list(np.arange(10) + 1)
    num_runs_LMS_vec.append(50)
    num_runs_LMS_vec.append(100)
    num_runs_LMS_vec.append(200)
    num_runs_LMS_vec.append(500)
    RMSE_LMS_runs, RMSE_PI_runs = mfc.num_LMS_runs_mode(d, n, k, num_runs_LMS_vec=num_runs_LMS_vec, num_runs=num_runs,
                                                    noise_amp=5, batch_size=1)
    # plotting the results num runs LMS
    plot(num_runs_LMS_vec, RMSE_LMS_runs, color='g', title='RMSEs as function of LMS #runs', xlabel='#runs LMS process',
         ylabel='Mean Root Mean Square RMSE')
    plt.plot(num_runs_LMS_vec, RMSE_PI_runs, color='b')
    plt.legend(('weights LMS', 'weights pseudo inverse'), loc='upper right')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\effective LMS\figures\N_runs_LMS_RMSE.pickle',
                     'wb'))

    # change n sample mode
    n_vec = [d / 1000, d / 500, d / 250, d / 100, d / 50, d / 25, d / 10, d / 5, d / 2]
    RMSE_n_sample, RMSE_PI_n_sample = mfc.n_samples_mode(d, n_vec, k, num_runs=num_runs, noise_amp=5, batch_size=1, num_runs_LMS=1)
    # plotting the results n samples
    plot(n_vec, RMSE_n_sample, color='g', title='RMSEs as function of #samples', xlabel='#sample',
         ylabel='Mean Root Mean Square RMSE')
    plt.plot(n_vec, RMSE_PI_n_sample, color='b')
    plt.legend(('weights LMS', 'weights pseudo inverse'), loc='upper right')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\effective LMS\figures\N_samples_RMSE.pickle',
                     'wb'))

    # effective dimension mode
    # k_vec = np.arange(15) + 5
    k_vec = [d / 500, d / 100, d / 50, d / 10, d / 5, d / 2, d]
    RMSE_k_effective, RMSE_PI_k_effective = mfc.effective_dimension_mode(d, n, k_vec, num_runs=1,
                                                                     noise_amp=5, batch_size=1, num_runs_LMS=1)
    # plotting the results SNR
    plot(k_vec, RMSE_k_effective, color='g', title='RMSEs as function of effective dimension',
         xlabel='effective dimension', ylabel='Mean Root Mean Square RMSE')
    plt.plot(k_vec, RMSE_PI_k_effective, color='b')
    plt.legend(('weights LMS', 'weights pseudo inverse'), loc='upper right')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\effective LMS\figures\effective_dim_RMSE.pickle',
                     'wb'))

    print('done')


if __name__ == '__main__':
    main()
