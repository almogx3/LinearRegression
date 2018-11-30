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


def LMSprocess(x, y, batch_size=1):
    """
    LMSprocess run LMS process on x,y
    :param x: training vectors
    :param y: desired answer
    :param batch_size: batch size for calculating error in LMS
    :return: weights according to LMS process
    """
    eps_value = 5e-3
    weights = np.random.rand(x.shape[0], ) * np.max(y)
    error_mean_old = np.Inf
    Emax = 100 * len(y)
    maxIter = 2000
    i = 0
    while ((i < maxIter)):
        for ind in range(len(y) - batch_size + 1):
            x_current = x[:, ind:(ind + batch_size)]
            error_current = y[ind:(ind + batch_size)] - np.matmul(x_current.T, weights)
            error_mean_current = np.sum((y - np.matmul(x.T, weights)) ** 2) ** 0.5
            if ((error_mean_current > Emax) or (abs(error_mean_current) > 1.1 * abs(error_mean_old))):
                weights = np.random.rand(x.shape[0], )
                error_mean_old = np.Inf
                i = 0
                print('new weights')
            else:
                error_eps = (error_current * eps_value)
                # error_eps = error_eps.reshape((error_eps.shape[0],1))
                weights = weights + np.dot(x_current, error_eps) / len(error_eps)

            if (abs(error_mean_current) == abs(error_mean_old)):
                i = maxIter + 5
                break
                print('Saturation')
            error_mean_old = error_mean_current
        i += 1
        if i == maxIter:
            print('maxIter')

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


def create_n_calc_all_data(d, n, k, noise_amp=1, batch_size=1):
    """

    :param d: dimension of vector x (x = [1,x(d dimension)])
    :param n: number of wanted vectors
    :param k: the effective dimension of x
    :param noise_amp: amplitude of the noise added to y
    :param batch_size: batch size for LMS process
    :return:  y_real - the real y without noise
              y_LMS - estimation of y after noising using LMS
              y_pseudo_inverse - estimation of y after noising using pseudo inverse
              SNR - SNR of y
    """
    R = create_cov_matrix(d, k)
    x, y = create_random_vectors(n, R, np.zeros((d,)))
    y_real = y
    y = y + noise_amp * np.random.rand(y.shape[0], )
    SNR = 10 * np.log10(np.linalg.norm(y_real) / np.linalg.norm((np.abs(y - y_real))))  # SNR in db
    weights = LMSprocess(x, y, batch_size=batch_size)
    weights_real = LMSprocess(x, y_real, batch_size=batch_size)
    # run PCA process
    eigenvalues_sorted, eigenvectors_sorted, S = PCAprocess(x)
    projcetion_x, eigenvectors_k = projection(x, k, eigenvectors_sorted)
    weights_projection_PCA = LMSprocess(projcetion_x, y, batch_size=batch_size)
    weights_PCA = np.matmul(eigenvectors_k, weights_projection_PCA)

    theta_at = np.matmul(np.matmul(x, np.linalg.inv(np.matmul(x.T, x))), y)
    y_pseudo_inverse = np.matmul(x.T, theta_at)
    y_LMS = np.matmul(x.T, weights)
    # y_LMS_real - results for data without noise
    y_LMS_real = np.matmul(x.T, weights_real)
    return y_real, y_LMS, y_pseudo_inverse, SNR


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

    """
    # different SNR mode
    num_SNR = 10
    noise_amp_vec = np.concatenate([np.arange(num_SNR), np.arange(num_SNR) * 10])
    noise_amp_vec += np.ones_like(noise_amp_vec)
    # noise_amp_vec = [1, 10, 100, 1000]
    
    error_LMS = np.zeros((len(noise_amp_vec), 1))
    error_pseudo_inv = np.zeros((len(noise_amp_vec), 1))
    SNR = np.zeros((len(noise_amp_vec), 1))
    error_LMS_run = np.zeros((num_runs, 1))
    error_pseudo_inv_run = np.zeros((num_runs, 1))
    SNR_run = np.zeros((num_runs, 1))

    widgets = ['Processing estimation: ', Percentage(), ' ', Bar()]
    max_val = len(noise_amp_vec) * num_runs
    bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()
    for i in range(len(noise_amp_vec)):
        noise_amp = noise_amp_vec[i]
        for l in range(num_runs):
            y_real, y_LMS, y_pseudo_inverse, SNR_current = create_n_calc_all_data(d, n, k, noise_amp, batch_size)
            error_pseudo_inv_run[l] = np.sum((y_real - y_pseudo_inverse) ** 2) ** 0.5
            error_LMS_run[l] = np.sum((y_real - y_LMS) ** 2) ** 0.5
            SNR_run[l] = SNR_current
            bar.update((i + 1) * (l + 1))

        error_pseudo_inv[i] = np.mean(error_pseudo_inv_run)
        error_LMS[i] = np.mean(error_LMS_run)
        SNR[i] = np.mean(SNR_run)
        print('iteration: ' + str(i))
    bar.finish()

    # sorting by SNR
    ind_sort = np.argsort(SNR, axis=0)
    # ploting the results SNR
    plt.figure()
    plt.plot(np.choose(ind_sort, SNR), np.choose(ind_sort, error_LMS), 'g')
    plt.plot(np.choose(ind_sort, SNR), np.choose(ind_sort, error_pseudo_inv), 'b')
    plt.legend(('weights LMS', 'weights pseudo inverse'),
               loc='upper right')
    plt.title('Errors as function of SNR')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-', linewidth='1.1')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Root Mean Square Error')
    plt.show()
    
    """

    # batch size mode
    batch_size = [1, 2, 5, 10, 50]

    error_LMS = np.zeros((len(batch_size), 1))
    error_pseudo_inv = np.zeros((len(batch_size), 1))
    SNR = np.zeros((len(batch_size), 1))
    error_LMS_run = np.zeros((num_runs, 1))
    error_pseudo_inv_run = np.zeros((num_runs, 1))
    SNR_run = np.zeros((num_runs, 1))

    # Progressbar
    widgets = ['Processing estimation: ', Percentage(), ' ', Bar()]
    max_val = len(batch_size) * num_runs
    bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()
    for i in range(len(batch_size)):
        batch_size_current = batch_size[i]
        for l in range(num_runs):
            y_real, y_LMS, y_pseudo_inverse, SNR_current = create_n_calc_all_data(d, n, k, noise_amp=3,
                                                                                  batch_size=batch_size_current)
            error_pseudo_inv_run[l] = np.sum((y_real - y_pseudo_inverse) ** 2) ** 0.5
            error_LMS_run[l] = np.sum((y_real - y_LMS) ** 2) ** 0.5
            SNR_run[l] = SNR_current
            bar.update((i + 1) * (l + 1))

        error_pseudo_inv[i] = np.mean(error_pseudo_inv_run)
        error_LMS[i] = np.mean(error_LMS_run)
        SNR[i] = np.mean(SNR_run)
        print('iteration: ' + str(i))
    bar.finish()

    # plotting the batch size results
    plt.figure()
    plt.plot(batch_size, error_LMS, 'g')
    plt.plot(batch_size, error_pseudo_inv, 'b')
    plt.legend(('weights LMS', 'weights pseudo inverse'),
               loc='upper right')
    plt.title('Errors as function of batch size')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-', linewidth='1.1')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    plt.xlabel('Batch size')
    plt.ylabel('Root Mean Square Error')
    plt.show()

    print('Error y LMS mean:' + str(np.mean(error_LMS)))
    print('Error y pseudo inverse mean:' + str(np.mean(error_pseudo_inv)))
    print('done')


if __name__ == '__main__':
    main()
