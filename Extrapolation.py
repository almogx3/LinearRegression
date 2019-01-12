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


def plotyy(x, y1, y2, color1='g', color2='b', title=None, xlabel=None, y1label=None, y2label=None):
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, color=color1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label)
    plt.title(title)
    ax1.tick_params('y', colors=color1)
    ax2 = ax1.twinx()
    ax2.plot(x, y2, color=color2)
    ax2.set_ylabel(y2label)
    ax2.tick_params('y', colors=color2)
    fig.tight_layout()
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    plt.show()


def main():
    plt.interactive(True)
    # x dimensions
    d = 1000
    # number of samples
    n = 100
    # The effective dimension
    k = 10
    # number of runs
    num_runs = 100
    noise_amp_all = 4
    batch_size_all = 5
    num_runs_LMS_all = 1
    polynomial_mode = 0

    # different SNR mode
    # num_SNR = 10
    # noise_amp_vec = np.concatenate([np.arange(num_SNR), np.arange(num_SNR) * 10])
    # noise_amp_vec += np.ones_like(noise_amp_vec)

    noise_amp_vec = [0.25,0.5, 0.75, 1, 10,20, 30, 40, 50, 100, 500]
    SNR, MSE_LMS_SNR, MSE_PI_SNR = mfc.SNR_mode(d, n, k, noise_amp_vec, num_runs=num_runs, batch_size=batch_size_all,
                                                polynomial=polynomial_mode, num_runs_LMS=num_runs_LMS_all)

    # plotting the results SNR
    plot(SNR, MSE_LMS_SNR, color='g', title='MSEs as function of SNR', xlabel='SNR [dB]',
         ylabel='Mean of MSE')
    plt.plot(SNR, MSE_PI_SNR, color='b')
    plt.legend(('weights LMS', 'weights pseudo inverse'), loc='upper right')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\SNR_MSE.pickle',
                     'wb'))
    plotyy(SNR, MSE_LMS_SNR, MSE_PI_SNR, color1='g', color2='b',
           title='MSEs as function of SNR', xlabel='SNR',
           y1label='Mean of MSE - LMS', y2label='Mean of MSE - PI')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\SNR_MSE-yy.pickle',
                     'wb'))

    # batch size mode
    batch_size = [1, 2, 5, 10, 15, 50]
    MSE_LMS_batch, MSE_PI_batch = mfc.batch_size_mode(d, n, k, batch_size_vec=batch_size, num_runs=num_runs,
                                                      noise_amp=noise_amp_all, polynomial=polynomial_mode,
                                                      num_runs_LMS=num_runs_LMS_all)

    # plotting the results batch size
    plot(batch_size, MSE_LMS_batch, color='g', title='MSEs as function of Batch size', xlabel='Batch size',
         ylabel='Mean of MSE')
    plt.plot(batch_size, MSE_PI_batch, color='b')
    plt.legend(('weights LMS', 'weights pseudo inverse'), loc='upper right')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\Batch_MSE.pickle', 'wb'))
    plotyy(batch_size, MSE_LMS_batch, MSE_PI_batch, color1='g', color2='b',
           title='MSEs as function of Batch size', xlabel='Batch size',
           y1label='Mean of MSE - LMS', y2label='Mean of MSE - PI')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\Batch_MSE-yy.pickle', 'wb'))

    # num of runs LMS
    num_runs_LMS_vec = list(np.arange(10) + 1)
    num_runs_LMS_vec.append(50)
    num_runs_LMS_vec.append(100)
    num_runs_LMS_vec.append(200)
    num_runs_LMS_vec.append(500)
    MSE_LMS_runs, MSE_PI_runs = mfc.num_LMS_runs_mode(d, n, k, num_runs_LMS_vec=num_runs_LMS_vec, num_runs=num_runs,
                                                      noise_amp=noise_amp_all, batch_size=batch_size_all,
                                                      polynomial=polynomial_mode)
    # plotting the results num runs LMS
    plot(num_runs_LMS_vec, MSE_LMS_runs, color='g', title='MSEs as function of LMS #runs', xlabel='#runs LMS process',
         ylabel='Mean of MSE')
    plt.plot(num_runs_LMS_vec, MSE_PI_runs, color='b')
    plt.legend(('weights LMS', 'weights pseudo inverse'), loc='upper right')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\N_runs_LMS_MSE.pickle', 'wb'))
    plotyy(num_runs_LMS_vec, MSE_LMS_runs, MSE_PI_runs, color1='g', color2='b',
           title='MSEs as function LMS #runs', xlabel='#runs LMS process',
           y1label='Mean of MSE - LMS', y2label='Mean of MSE - PI')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\N_runs_LMS_MSE-yy.pickle', 'wb'))

    # change n sample mode
    n_vec = [d / 1000, d / 500, d / 250, d / 100, d / 50, d / 25, d / 10, d / 5, d / 2, d]
    MSE_n_sample, MSE_PI_n_sample = mfc.n_samples_mode(d, n_vec, k, num_runs=num_runs, noise_amp=noise_amp_all,
                                                       batch_size=batch_size_all,
                                                       num_runs_LMS=num_runs_LMS_all, polynomial=polynomial_mode)
    # plotting the results n samples
    plot(n_vec, MSE_n_sample, color='g', title='MSEs as function of #samples', xlabel='#sample',
         ylabel='Mean of MSE')
    plt.plot(n_vec, MSE_PI_n_sample, color='b')
    plt.legend(('weights LMS', 'weights pseudo inverse'), loc='upper right')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\N_samples_MSE.pickle',
                     'wb'))
    plotyy(n_vec, MSE_n_sample, MSE_PI_n_sample, color1='g', color2='b',
           title='MSEs as function of #samples', xlabel='#sample',
           y1label='Mean of MSE - LMS', y2label='Mean of MSE - PI')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\N_samples_MSE-yy.pickle',
                     'wb'))

    # effective dimension mode
    # k_vec = np.arange(15) + 5
    k_vec = [d / 500, d / 100, d / 50, d / 10, d / 5, d / 2, d]
    MSE_k_effective, MSE_PI_k_effective = mfc.effective_dimension_mode(d, n, k_vec, num_runs=num_runs,
                                                                       noise_amp=noise_amp_all,
                                                                       batch_size=batch_size_all,
                                                                       num_runs_LMS=num_runs_LMS_all,
                                                                       polynomial=1)
    # plotting the results SNR
    plot(k_vec, MSE_k_effective, color='g', title='MSEs as function of effective dimension',
         xlabel='effective dimension', ylabel='Mean of MSE')
    plt.plot(k_vec, MSE_PI_k_effective, color='b')
    plt.legend(('weights LMS', 'weights pseudo inverse'), loc='upper right')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\effective_dim_MSE.pickle',
                     'wb'))
    plotyy(k_vec, MSE_k_effective, MSE_PI_k_effective, color1='g', color2='b',
           title='MSEs as function of effective dimension', xlabel='effective dimension',
           y1label='Mean of MSE - LMS', y2label='Mean of MSE - PI')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\effective_dim_MSE-yy.pickle',
                     'wb'))

    # step size mode
    mu_vec = np.array([1e-4, 1e-3, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 10, 20, 50, 100]) * 1e-4
    MSE_mu, MSE_PI_mu = mfc.step_size_mode(d, n, k, mu_vec=mu_vec, num_runs=num_runs,
                                           noise_amp=noise_amp_all,
                                           batch_size=batch_size_all,
                                           num_runs_LMS=num_runs_LMS_all,
                                           polynomial=polynomial_mode)

    # plotting the results SNR
    plot(mu_vec, MSE_mu, color='g', title='MSEs as function of effective dimension',
         xlabel='step size', ylabel='Mean of MSE')
    plt.plot(mu_vec, MSE_PI_mu, color='b')
    plt.legend(('weights LMS', 'weights pseudo inverse'), loc='upper right')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\step_size_MSE.pickle',
                     'wb'))
    plotyy(mu_vec, MSE_mu, MSE_PI_mu, color1='g', color2='b', title='MSEs as function of step size', xlabel='step size',
           y1label='Mean of MSE - LMS', y2label='Mean of MSE - PI')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\step_size_MSE-yy.pickle',
                     'wb'))

    print('done')
    plt.close('all')


if __name__ == '__main__':
    main()
