import pickle
import numpy as np
import matplotlib.pyplot as plt
# from progress.bar import Bar
# from tqdm import tqdm
from progressbar import *


def create_cov_matrix(d, k):
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


def create_random_vectors(num_of_vectors=100, cov_matrix=np.eye(100), mean=np.zeros((100,))):
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


def LMS(x, y, step=1, weights=None, eps_value=5e-3):
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


def LMS_multiple_times(x, y, step=1, eps_value=5e-3, num_runs=1):
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
    weights = LMS(x, y, step=step, eps_value=eps_value)
    for counts in range(num_runs - 1):
        weights = LMS(x, y, step=step, weights=weights, eps_value=eps_value)
    #     option to change for to while and add converge condition (using error or weights change)
    return weights


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


def train(d, n, k, noise_amp=1, batch_size=1, num_runs_LMS=1):
    """
    train function create data fits to d, n, k, noise amp,
    and trains it using LMS and pseudo inverse (PI) method
    using batch size
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
    R = create_cov_matrix(d, k)
    x, y = create_random_vectors(n, R, np.zeros((d,)))
    y_real = y
    y = y + noise_amp * np.random.rand(y.shape[0], )
    SNR = 10 * np.log10(np.linalg.norm(y_real) / np.linalg.norm((np.abs(y - y_real))))  # SNR in db
    weights_LMS = LMS_multiple_times(x, y, step=batch_size, num_runs=num_runs_LMS)
    weights_PI = np.matmul(np.matmul(x, np.linalg.inv(np.matmul(x.T, x))), y)

    return weights_LMS, weights_PI, SNR


def test(d, n, k, weights_LMS, weights_PI):
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
    R = create_cov_matrix(d, k)
    x, y = create_random_vectors(n, R, np.zeros((d,)))
    y_PI = np.matmul(x.T, weights_PI)
    y_LMS = np.matmul(x.T, weights_LMS)
    RMSE_LMS = np.mean((y - y_LMS) ** 2) ** 0.5
    RMSE_PI = np.mean((y - y_PI) ** 2) ** 0.5

    return RMSE_LMS, RMSE_PI


def SNR_mode(d, n, k, noise_amp_vec, num_runs=1, num_runs_LMS=1, batch_size=1):
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
    :return:
            SNR_sorted: SNR of the train data
            RMSE_LMS_sorted: RMSE (root mean square error) of test data using LMS method
            RMSE_PI_sorted: RMSE (root mean square error) of test data using pseudo inverse method
    """
    # definitions
    RMSE_LMS = np.zeros((len(noise_amp_vec), 1))
    RMSE_PI = np.zeros((len(noise_amp_vec), 1))

    RMSE_LMS_run = np.zeros((num_runs, 1))
    RMSE_PI_run = np.zeros((num_runs, 1))

    SNR = np.zeros((len(noise_amp_vec), 1))
    SNR_run = np.zeros((num_runs, 1))

    widgets = ['Processing estimation SNR mode: ', Percentage(), ' ', Bar()]
    max_val = len(noise_amp_vec) * num_runs
    bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()

    for i in range(len(noise_amp_vec)):
        noise_amp = noise_amp_vec[i]
        weights_LMS, weights_PI, SNR[i] = train(d, n, k, noise_amp=noise_amp, batch_size=batch_size,
                                                num_runs_LMS=num_runs_LMS)
        for l in range(num_runs):
            RMSE_LMS_run[l], RMSE_PI_run[l] = test(d, n, k, weights_LMS, weights_PI)
            bar.update(i * num_runs + (l + 1))
        RMSE_PI[i] = np.mean(RMSE_PI_run)
        RMSE_LMS[i] = np.mean(RMSE_LMS_run)

    # sorting by SNR
    ind_sort = np.argsort(SNR, axis=0)
    SNR_sorted = np.choose(ind_sort, SNR)
    RMSE_LMS_sorted = np.choose(ind_sort, RMSE_LMS)
    RMSE_PI_sorted = np.choose(ind_sort, RMSE_PI)

    bar.finish()

    return SNR_sorted, RMSE_LMS_sorted, RMSE_PI_sorted


def batch_size_mode(d, n, k, batch_size_vec, num_runs=1, noise_amp=1, num_runs_LMS=1):
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
    :return:
            RMSE_LMS: RMSE (root mean square error) of test data using LMS method
            RMSE_PI: RMSE (root mean square error) of test data using pseudo inverse method
    """
    # definitions
    RMSE_LMS = np.zeros((len(batch_size_vec), 1))
    RMSE_PI = np.zeros((len(batch_size_vec), 1))

    RMSE_LMS_run = np.zeros((num_runs, 1))
    RMSE_PI_run = np.zeros((num_runs, 1))

    SNR = np.zeros((len(batch_size_vec), 1))
    SNR_run = np.zeros((num_runs, 1))

    widgets = ['Processing estimation batch size mode: ', Percentage(), ' ', Bar()]
    max_val = len(batch_size_vec) * num_runs
    bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()

    for i in range(len(batch_size_vec)):
        batch_size = batch_size_vec[i]
        weights_LMS, weights_PI, SNR[i] = train(d, n, k, noise_amp=noise_amp, batch_size=batch_size,
                                                num_runs_LMS=num_runs_LMS)
        for l in range(num_runs):
            RMSE_LMS_run[l], RMSE_PI_run[l] = test(d, n, k, weights_LMS, weights_PI)
            bar.update(i * num_runs + (l + 1))
        RMSE_PI[i] = np.mean(RMSE_PI_run)
        RMSE_LMS[i] = np.mean(RMSE_LMS_run)

    bar.finish()

    return RMSE_LMS, RMSE_PI


def num_LMS_runs_mode(d, n, k, num_runs_LMS_vec, num_runs=1, noise_amp=1, batch_size=1):
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
    :return:
            RMSE_LMS: RMSE (root mean square error) of test data using LMS method
            RMSE_PI: RMSE (root mean square error) of test data using pseudo inverse method
    """
    # definitions
    RMSE_LMS = np.zeros((len(num_runs_LMS_vec), 1))
    RMSE_PI = np.zeros((len(num_runs_LMS_vec), 1))

    RMSE_LMS_run = np.zeros((num_runs, 1))
    RMSE_PI_run = np.zeros((num_runs, 1))

    SNR = np.zeros((len(num_runs_LMS_vec), 1))

    widgets = ['Processing estimation num LMS runs mode: ', Percentage(), ' ', Bar()]
    max_val = len(num_runs_LMS_vec) * num_runs
    bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()

    for i in range(len(num_runs_LMS_vec)):
        num_runs_LMS = num_runs_LMS_vec[i]
        weights_LMS, weights_PI, SNR[i] = train(d, n, k, noise_amp=noise_amp, batch_size=batch_size,
                                                num_runs_LMS=num_runs_LMS)
        for l in range(num_runs):
            RMSE_LMS_run[l], RMSE_PI_run[l] = test(d, n, k, weights_LMS, weights_PI)
            bar.update(i * num_runs + (l + 1))
        RMSE_PI[i] = np.mean(RMSE_PI_run)
        RMSE_LMS[i] = np.mean(RMSE_LMS_run)

    bar.finish()

    return RMSE_LMS, RMSE_PI


def n_samples_mode(d, n_sample_vec, k, num_runs=1, noise_amp=1, batch_size=1, num_runs_LMS=1):
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
    :return:
            RMSE_LMS: RMSE (root mean square error) of test data using LMS method
            RMSE_PI: RMSE (root mean square error) of test data using pseudo inverse method
    """
    # definitions
    RMSE_LMS = np.zeros((len(n_sample_vec), 1))
    RMSE_PI = np.zeros((len(n_sample_vec), 1))

    RMSE_LMS_run = np.zeros((num_runs, 1))
    RMSE_PI_run = np.zeros((num_runs, 1))

    SNR = np.zeros((len(n_sample_vec), 1))
    SNR_run = np.zeros((num_runs, 1))

    widgets = ['Processing estimation number samples mode: ', Percentage(), ' ', Bar()]
    max_val = len(n_sample_vec) * num_runs
    bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()

    for i in range(len(n_sample_vec)):
        n = int(n_sample_vec[i])
        weights_LMS, weights_PI, SNR[i] = train(d, n, k, noise_amp=noise_amp, batch_size=batch_size,
                                                num_runs_LMS=num_runs_LMS)
        for l in range(num_runs):
            RMSE_LMS_run[l], RMSE_PI_run[l] = test(d, n, k, weights_LMS, weights_PI)
            bar.update(i * num_runs + (l + 1))
        RMSE_PI[i] = np.mean(RMSE_PI_run)
        RMSE_LMS[i] = np.mean(RMSE_LMS_run)

    bar.finish()

    return RMSE_LMS, RMSE_PI


def effective_dimension_mode(d, n, k_vec, num_runs=1, noise_amp=1, batch_size=1, num_runs_LMS=1):
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
    :return:
            RMSE_LMS: RMSE (root mean square error) of test data using LMS method
            RMSE_PI: RMSE (root mean square error) of test data using pseudo inverse method
    """
    # definitions
    RMSE_LMS = np.zeros((len(k_vec), 1))
    RMSE_PI = np.zeros((len(k_vec), 1))

    RMSE_LMS_run = np.zeros((num_runs, 1))
    RMSE_PI_run = np.zeros((num_runs, 1))

    SNR = np.zeros((len(k_vec), 1))
    SNR_run = np.zeros((num_runs, 1))

    widgets = ['Processing estimation effective dimension mode: ', Percentage(), ' ', Bar()]
    max_val = len(k_vec) * num_runs
    bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()

    for i in range(len(k_vec)):
        k = int(k_vec[i])
        weights_LMS, weights_PI, SNR[i] = train(d, n, k, noise_amp=noise_amp, batch_size=batch_size,
                                                num_runs_LMS=num_runs_LMS)
        for l in range(num_runs):
            RMSE_LMS_run[l], RMSE_PI_run[l] = test(d, n, k, weights_LMS, weights_PI)
            bar.update(i * num_runs + (l + 1))
        RMSE_PI[i] = np.mean(RMSE_PI_run)
        RMSE_LMS[i] = np.mean(RMSE_LMS_run)

    bar.finish()

    return RMSE_LMS, RMSE_PI


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
    SNR, RMSE_LMS_SNR, RMSE_PI_SNR = SNR_mode(d, n, k, noise_amp_vec, num_runs=num_runs, batch_size=1)

    # plotting the results SNR
    plot(SNR, RMSE_LMS_SNR, color='g', title='RMSEs as function of SNR', xlabel='SNR [dB]',
         ylabel='Mean Root Mean Square RMSE')
    plt.plot(SNR, RMSE_PI_SNR, color='b')
    plt.legend(('weights LMS', 'weights pseudo inverse'), loc='upper right')
    pickle.dump(plt.gcf(),
                open(r'D:\Documents\GitWorks\LinearRegression\results\effective LMS\figures\SNR_RMSE.pickle', 'wb'))

    # batch size mode
    batch_size = [1, 2, 5, 10, 50, 100]
    RMSE_LMS_batch, RMSE_PI_batch = batch_size_mode(d, n, k, batch_size_vec=batch_size, num_runs=num_runs, noise_amp=5)

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
    RMSE_LMS_runs, RMSE_PI_runs = num_LMS_runs_mode(d, n, k, num_runs_LMS_vec=num_runs_LMS_vec, num_runs=num_runs,
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
    n_vec = [d / 1000, d / 500, d / 250, d / 100, d / 50, d / 25, d / 10, d / 5, d / 2, d, 2*d]
    RMSE_n_sample, RMSE_PI_n_sample = n_samples_mode(d, n_vec, k, num_runs=num_runs, noise_amp=5, batch_size=1, num_runs_LMS=1)
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
    RMSE_k_effective, RMSE_PI_k_effective = effective_dimension_mode(d, n, k_vec, num_runs=1,
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
