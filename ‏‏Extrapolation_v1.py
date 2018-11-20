import numpy as np
import matplotlib.pyplot as plt

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


def LMSprocess(x, y):
    """
    LMSprocess run LMS process on x,y
    :param x: training vectors
    :param y: desired answer
    :return: weights according to LMS process
    """
    eps_value = 5e-4
    theta_at = np.matmul(np.matmul(x, np.linalg.inv(np.matmul(x.T, x))), y)
    y_final = np.matmul(x.T, theta_at)
    weights = np.random.rand(x.shape[0], )*np.max(y)
    error_mean_old = np.Inf
    Emax = len(y)
    maxIter = 1000
    i = 0
    while ((i < maxIter) ):
        for ind in range(len(y)):
            x_current = x[:, ind]
            error_current = y[ind] - np.matmul(x_current.T, weights)
            # error_current = np.mean(y - np.matmul(x.T, weights))
            # error_mean_current = np.mean(y - np.matmul(x.T, weights))
            error_mean_current = np.sum((y-np.matmul(x.T, weights))**2)**0.5
            sign = np.sign(np.mean(np.sign(y - np.matmul(x.T, weights))))
            if sign == 0:
                sign = 1
            error_mean_current = sign*error_mean_current
            if (error_mean_current > Emax) or (abs(error_mean_current) > 1.1 * abs(error_mean_old)):
                    weights = np.random.rand(x.shape[0], )
                    error_mean_old = np.Inf
                    i = 0
                    print('new weights')
            else:
                weights = weights + np.dot(error_current * eps_value, x_current)

            if (abs(error_mean_current) == abs(error_mean_old)):
                i = maxIter + 5
                break
                print('Saturation')
            error_mean_old = error_mean_current
            print(error_mean_current)
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


def main():
    plt.interactive(True)
    # x dimensions
    d = 1000
    # number of samples
    n = 100
    # The "real dimension
    k = 10
    R = create_cov_matrix(d, k)
    x, y = create_random_vectors(n, R, np.zeros((d,)))
    weights = LMSprocess(x, y)
    # run PCA process
    eigenvalues_sorted, eigenvectors_sorted, S = PCAprocess(x)
    projcetion_x, eigenvectors_k = projection(x, k, eigenvectors_sorted)
    weights_projection_PCA = LMSprocess(projcetion_x, y)
    weights_PCA = np.matmul(eigenvectors_k,weights_projection_PCA)
    y_ = np.matmul(x.T, weights)
    theta_at = np.matmul(np.matmul(x, np.linalg.inv(np.matmul(x.T, x))), y)
    y_final = np.matmul(x.T, theta_at)
    plt.figure()
    plt.plot(y,'b')
    # plt.hold(True)
    plt.plot(y_,'g')
    plt.plot(y_final, 'r')
    plt.legend(('Real', 'weights LMS', 'weights pseudo inverse'),
               loc='upper right')
    plt.show()
    print(np.sum((y-y_)**2)**0.5)
    print('done')


if __name__ == '__main__':
    main()
