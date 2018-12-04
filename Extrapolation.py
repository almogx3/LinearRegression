import numpy as np
import matplotlib.pyplot as plt
# from progress.bar import Bar
# from tqdm import tqdm
from progressbar import *
import pickle


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
    :param batch_size: batch size for calculating RMSE in LMS
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
            error_mean_current = np.mean((y - np.matmul(x.T, weights)) ** 2) ** 0.5
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


def train(d, n, k, noise_amp=1, batch_size=1):
    """
    train function create data fits to d, n, k, noise amp,
    and trains it using LMS and pseudo inverse (PI) method
    using batch size
    :param d: dimension of vector x (x = [1,x(d dimension)])
    :param n: number of wanted vectors
    :param k: the effective dimension of x
    :param noise_amp: amplitude of the noise added to y (optional)
    :param batch_size: batch size for LMS process (optional)
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
    weights_LMS = LMSprocess(x, y, batch_size=batch_size)
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
    theta_at = np.matmul(np.matmul(x, np.linalg.inv(np.matmul(x.T, x))), y)
    y_pseudo_inverse = np.matmul(x.T, theta_at)
    y_LMS = np.matmul(x.T, weights)

    return y_real, y_LMS, y_pseudo_inverse, SNR


def SNR_mode(d, n, k, noise_amp_vec, num_runs=1, batch_size=1):
    """
    SNR_mode function running train for data with different SNR (fits to noise_amp)
    :param d:
    :param n:
    :param k:
    :param noise_amp_vec:
    :param num_runs:
    :param batch_size:
    :return:
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
        for l in range(num_runs):
            weights_LMS, weights_PI, SNR_run[l] = train(d, n, k, noise_amp=noise_amp, batch_size=batch_size)
            RMSE_LMS_run[l], RMSE_PI_run[l] = test(d, n, k, weights_LMS, weights_PI)
            bar.update(i * num_runs + (l + 1))
        RMSE_PI[i] = np.mean(RMSE_PI_run)
        RMSE_LMS[i] = np.mean(RMSE_LMS_run)
        SNR[i] = np.mean(SNR_run)

        print('iteration: ' + str(i))


    # sorting by SNR
    ind_sort = np.argsort(SNR, axis=0)
    SNR_sorted = np.choose(ind_sort, SNR)
    RMSE_LMS_sorted = np.choose(ind_sort, RMSE_LMS)
    RMSE_PI_sorted = np.choose(ind_sort, RMSE_PI)

    bar.finish()

    return SNR_sorted, RMSE_LMS_sorted, RMSE_LMS_sorted

def old_main():
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
    num_SNR = 10
    noise_amp_vec = np.concatenate([np.arange(num_SNR), np.arange(num_SNR) * 10])
    noise_amp_vec += np.ones_like(noise_amp_vec)
    # noise_amp_vec = [1, 10, 100, 1000]

    RMSE_LMS = np.zeros((len(noise_amp_vec), 1))
    RMSE_pseudo_inv = np.zeros((len(noise_amp_vec), 1))
    RMSE_LMS_run = np.zeros((num_runs, 1))
    RMSE_pseudo_inv_run = np.zeros((num_runs, 1))
    SNR = np.zeros((len(noise_amp_vec), 1))
    MAE_LMS = np.zeros((len(noise_amp_vec), 1))
    MAE_pseudo_inv = np.zeros((len(noise_amp_vec), 1))
    MAE_LMS_run = np.zeros((num_runs, 1))
    MAE_pseudo_inv_run = np.zeros((num_runs, 1))
    SNR_run = np.zeros((num_runs, 1))

    widgets = ['Processing estimation: ', Percentage(), ' ', Bar()]
    max_val = len(noise_amp_vec) * num_runs
    bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()
    for i in range(len(noise_amp_vec)):
        noise_amp = noise_amp_vec[i]
        for l in range(num_runs):
            y_real, y_LMS, y_pseudo_inverse, SNR_current = create_n_calc_all_data(d, n, k, noise_amp)
            RMSE_pseudo_inv_run[l] = np.mean((y_real - y_pseudo_inverse) ** 2) ** 0.5
            RMSE_LMS_run[l] = np.mean((y_real - y_LMS) ** 2) ** 0.5
            MAE_LMS_run[l] = np.mean(np.abs(y_real - y_LMS))
            MAE_pseudo_inv_run[l] = np.mean(np.abs(y_real - y_pseudo_inverse))

            SNR_run[l] = SNR_current
            bar.update((i + 1) * (l + 1))

        RMSE_pseudo_inv[i] = np.mean(RMSE_pseudo_inv_run)
        RMSE_LMS[i] = np.mean(RMSE_LMS_run)
        MAE_pseudo_inv[i] = np.mean(MAE_pseudo_inv_run)
        MAE_LMS[i] = np.mean(MAE_LMS_run)
        SNR[i] = np.mean(SNR_run)
        print('iteration: ' + str(i))
    bar.finish()

    # sorting by SNR
    ind_sort = np.argsort(SNR, axis=0)
    # ploting the results SNR
    plt.figure()
    plt.plot(np.choose(ind_sort, SNR), np.choose(ind_sort, RMSE_LMS), 'g')
    plt.plot(np.choose(ind_sort, SNR), np.choose(ind_sort, RMSE_pseudo_inv), 'b')
    plt.legend(('weights LMS', 'weights pseudo inverse'),
               loc='upper right')
    plt.title('RMSEs as function of SNR')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-', linewidth='1.1')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Mean Root Mean Square RMSE')
    plt.show()

    plt.figure()
    plt.plot(np.choose(ind_sort, SNR), np.choose(ind_sort, MAE_LMS), 'g')
    plt.plot(np.choose(ind_sort, SNR), np.choose(ind_sort, MAE_pseudo_inv), 'b')
    plt.legend(('weights LMS', 'weights pseudo inverse'),
               loc='upper right')
    plt.title('MAEs as function of SNR')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-', linewidth='1.1')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Mean Mean Absolute Error MAE ')
    plt.show()

    # batch size mode
    batch_size = [1, 2, 5, 10, 50]
    #
    # RMSE_LMS = np.zeros((len(batch_size), 1))
    # RMSE_pseudo_inv = np.zeros((len(batch_size), 1))
    # SNR = np.zeros((len(batch_size), 1))
    # RMSE_LMS_run = np.zeros((num_runs, 1))
    # RMSE_pseudo_inv_run = np.zeros((num_runs, 1))
    # SNR_run = np.zeros((num_runs, 1))
    # MAE_LMS = np.zeros((len(batch_size), 1))
    # MAE_pseudo_inv = np.zeros((len(batch_size), 1))
    # MAE_LMS_run = np.zeros((num_runs, 1))
    # MAE_pseudo_inv_run = np.zeros((num_runs, 1))
    #
    # # Progressbar
    # widgets = ['Processing estimation: ', Percentage(), ' ', Bar()]
    # max_val = len(batch_size) * num_runs
    # bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()
    # for i in range(len(batch_size)):
    #     batch_size_current = batch_size[i]
    #     for l in range(num_runs):
    #         y_real, y_LMS, y_pseudo_inverse, SNR_current = create_n_calc_all_data(d, n, k, noise_amp=1,
    #                                                                               batch_size=batch_size_current)
    #         RMSE_pseudo_inv_run[l] = np.mean((y_real - y_pseudo_inverse) ** 2) ** 0.5
    #         MAE_LMS_run[l] = np.mean(np.abs(y_real - y_LMS))
    #         MAE_pseudo_inv_run[l] = np.mean(np.abs(y_real - y_pseudo_inverse))
    #         RMSE_LMS_run[l] = np.mean((y_real - y_LMS) ** 2) ** 0.5
    #         SNR_run[l] = SNR_current
    #         bar.update((i + 1) * (l + 1))
    #
    #     RMSE_pseudo_inv[i] = np.mean(RMSE_pseudo_inv_run)
    #     RMSE_LMS[i] = np.mean(RMSE_LMS_run)
    #     MAE_pseudo_inv[i] = np.mean(MAE_pseudo_inv_run)
    #     MAE_LMS[i] = np.mean(MAE_LMS_run)
    #
    #     print('iteration: ' + str(i))
    # bar.finish()
    #
    # # plotting the batch size results
    # plt.figure()
    # plt.plot(batch_size, RMSE_LMS, 'g')
    # plt.plot(batch_size, RMSE_pseudo_inv, 'b')
    # plt.legend(('weights LMS', 'weights pseudo inverse'),
    #            loc='upper right')
    # plt.title('RMSEs as function of batch size')
    # plt.minorticks_on()
    # plt.grid(b=True, which='major', color='k', linestyle='-', linewidth='1.1')
    # plt.grid(b=True, which='minor', color='k', linestyle='--')
    # plt.xlabel('Batch size')
    # plt.ylabel('Mean Root Mean Square RMSE')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(batch_size, MAE_LMS, 'g')
    # plt.plot(batch_size, MAE_pseudo_inv, 'b')
    # plt.legend(('weights LMS', 'weights pseudo inverse'),
    #            loc='upper right')
    # plt.title('MAEs as function of batch size')
    # plt.minorticks_on()
    # plt.grid(b=True, which='major', color='k', linestyle='-', linewidth='1.1')
    # plt.grid(b=True, which='minor', color='k', linestyle='--')
    # plt.xlabel('Batch size')
    # plt.ylabel('Mean Mean Absolute Error MAE ')
    # plt.show()
    #

    # change n sample mode
    n_vec = [d / 1000, d / 500, d / 250, d / 100, d / 50, d / 25, d / 10, d / 5, d / 2]

    # RMSE_LMS = np.zeros((len(n_vec), 1))
    # RMSE_pseudo_inv = np.zeros((len(n_vec), 1))
    # SNR = np.zeros((len(n_vec), 1))
    # RMSE_LMS_run = np.zeros((num_runs, 1))
    # RMSE_pseudo_inv_run = np.zeros((num_runs, 1))
    # SNR_run = np.zeros((num_runs, 1))
    # MAE_LMS = np.zeros((len(n_vec), 1))
    # MAE_pseudo_inv = np.zeros((len(n_vec), 1))
    # MAE_LMS_run = np.zeros((num_runs, 1))
    # MAE_pseudo_inv_run = np.zeros((num_runs, 1))
    #
    # # Progressbar
    # widgets = ['Processing estimation: ', Percentage(), ' ', Bar()]
    # max_val = len(n_vec) * num_runs
    # bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()
    # for i in range(len(n_vec)):
    #     n_current = int(n_vec[i])
    #     for l in range(num_runs):
    #         y_real, y_LMS, y_pseudo_inverse, SNR_current = create_n_calc_all_data(d, n_current, k, noise_amp=1)
    #         RMSE_pseudo_inv_run[l] = np.mean((y_real - y_pseudo_inverse) ** 2) ** 0.5
    #         MAE_LMS_run[l] = np.mean(np.abs(y_real - y_LMS))
    #         MAE_pseudo_inv_run[l] = np.mean(np.abs(y_real - y_pseudo_inverse))
    #         RMSE_LMS_run[l] = np.mean((y_real - y_LMS) ** 2) ** 0.5
    #         SNR_run[l] = SNR_current
    #         bar.update((i + 1) * (l + 1))
    #
    #     RMSE_pseudo_inv[i] = np.mean(RMSE_pseudo_inv_run)
    #     RMSE_LMS[i] = np.mean(RMSE_LMS_run)
    #     MAE_pseudo_inv[i] = np.mean(MAE_pseudo_inv_run)
    #     MAE_LMS[i] = np.mean(MAE_LMS_run)
    #     print('iteration: ' + str(i))
    # bar.finish()
    #
    # # plotting the n samples results
    # plt.figure()
    # plt.plot(n_vec, RMSE_LMS, 'g')
    # plt.plot(n_vec, RMSE_pseudo_inv, 'b')
    # plt.legend(('weights LMS', 'weights pseudo inverse'),
    #            loc='best')
    # plt.title('RMSEs as function of #samples')
    # plt.minorticks_on()
    # plt.grid(b=True, which='major', color='k', linestyle='-', linewidth='1.1')
    # plt.grid(b=True, which='minor', color='k', linestyle='--')
    # plt.xlabel('# samples')
    # plt.ylabel('Mean Root Mean Square RMSE')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(n_vec, MAE_LMS, 'g')
    # plt.plot(n_vec, MAE_pseudo_inv, 'b')
    # plt.legend(('weights LMS', 'weights pseudo inverse'),
    #            loc='best')
    # plt.title('MAEs as function of #samples')
    # plt.minorticks_on()
    # plt.grid(b=True, which='major', color='k', linestyle='-', linewidth='1.1')
    # plt.grid(b=True, which='minor', color='k', linestyle='--')
    # plt.xlabel('# samples')
    # plt.ylabel('Mean Mean Absolute Error MAE')
    # plt.show()

    # effective dimension mode
    # k_vec = np.arange(15) + 5
    k_vec = [d / 500, d / 100, d / 50, d / 10, d / 5, d / 2]

    RMSE_LMS = np.zeros((len(k_vec), 1))
    RMSE_pseudo_inv = np.zeros((len(k_vec), 1))
    SNR = np.zeros((len(k_vec), 1))
    RMSE_LMS_run = np.zeros((num_runs, 1))
    RMSE_pseudo_inv_run = np.zeros((num_runs, 1))
    SNR_run = np.zeros((num_runs, 1))
    MAE_LMS = np.zeros((len(k_vec), 1))
    MAE_pseudo_inv = np.zeros((len(k_vec), 1))
    MAE_LMS_run = np.zeros((num_runs, 1))
    MAE_pseudo_inv_run = np.zeros((num_runs, 1))

    # Progressbar
    widgets = ['Processing estimation: ', Percentage(), ' ', Bar()]
    max_val = len(k_vec) * num_runs
    bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()
    for i in range(len(k_vec)):
        k_current = k_vec[i]
        for l in range(num_runs):
            y_real, y_LMS, y_pseudo_inverse, SNR_current = create_n_calc_all_data(d, n, k=k_current, noise_amp=1,
                                                                                  batch_size=5)
            RMSE_pseudo_inv_run[l] = np.mean((y_real - y_pseudo_inverse) ** 2) ** 0.5
            MAE_LMS_run[l] = np.mean(np.abs(y_real - y_LMS))
            MAE_pseudo_inv_run[l] = np.mean(np.abs(y_real - y_pseudo_inverse))
            RMSE_LMS_run[l] = np.mean((y_real - y_LMS) ** 2) ** 0.5
            SNR_run[l] = SNR_current
            bar.update((i + 1) * (l + 1))

        RMSE_pseudo_inv[i] = np.mean(RMSE_pseudo_inv_run)
        RMSE_LMS[i] = np.mean(RMSE_LMS_run)
        MAE_pseudo_inv[i] = np.mean(MAE_pseudo_inv_run)
        MAE_LMS[i] = np.mean(MAE_LMS_run)

        print('iteration: ' + str(i))
    bar.finish()

    # plotting the batch size results
    plt.figure()
    plt.plot(k_vec, RMSE_LMS, 'g')
    plt.plot(k_vec, RMSE_pseudo_inv, 'b')
    plt.legend(('weights LMS', 'weights pseudo inverse'),
               loc='upper right')
    plt.title('RMSEs as function of effective dimension')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-', linewidth='1.1')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    plt.xlabel('Effective dimension')
    plt.ylabel('Mean Root Mean Square RMSE')
    plt.show()

    plt.figure()
    plt.plot(k_vec, MAE_LMS, 'g')
    plt.plot(k_vec, MAE_pseudo_inv, 'b')
    plt.legend(('weights LMS', 'weights pseudo inverse'),
               loc='upper right')
    plt.title('MAEs as function of effective dimension')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-', linewidth='1.1')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    plt.xlabel('Effective dimension')
    plt.ylabel('Mean Mean Absolute Error MAE ')
    plt.show()

    print('RMSE y LMS mean:' + str(np.mean(RMSE_LMS)))
    print('RMSE y pseudo inverse mean:' + str(np.mean(RMSE_pseudo_inv)))
    print('done')


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


if __name__ == '__main__':
    main()
