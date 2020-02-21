import numpy as np
import matplotlib.pyplot as plt
def kMeans(data, k, max_iter, min_improve_ratio= 1E-5):
    """

    :param data: numpy array in shape (nData, nFeatures)
    :param k: number k in kmeans
    :param max_iter: maximum iteration.
    :return: centroid location in shape (k, nFeatures)
    """
    nData=data.shape[0]

    cent_init_inds = np.random.permutation(np.arange(nData))[:k]
    cent_init = data[cent_init_inds]

    cent_x = cent_init
    clst_inds_ls = [[] for _ in range(k)]
    loss= 0
    for it in range(max_iter):
        loss_old= loss;
        loss = 0
        for i in range(nData):
            cent_dist = np.sum((cent_x - data[i]) * (cent_x - data[i]), axis= -1)
            clst_ind = np.argmin(cent_dist)
            clst_inds_ls[clst_ind].append(i)
            loss = loss + cent_dist[clst_ind]

        # update new centroid
        for ki in range(k):
            cent_x[ki] = np.mean(data[clst_inds_ls[ki]], axis= 0)

        if(abs(loss - loss_old) / loss_old < min_improve_ratio):
            break;
    return cent_x

def generate_data(mu, sigma, nSamples= 500):
    """
    生成符合多中心二维高斯分布的数据
    :param mu: list of mu of array of mu
    :param sigma: list of sigma or array of sigma
    :param nSamples: nSample for each centroid
    :return:
    """
    assert len(mu) == len(sigma), "inconsistent number of centroid"
    nCentroid= mu.shape[0]
    ret = np.zeros([nSamples * nCentroid, 2])
    for i in range(nCentroid):
        ret[i * nSamples: (i + 1) * nSamples] = np.random.randn(nSamples, 2) * sigma[i] + mu[i]
    return ret

if __name__ == '__main__':
    nSamples= 500
    mu = np.array([[1, 1], [-1, -1]]);
    sigma= np.array([[0.5, 0.2], [0.2, 0.5]])
    data = generate_data(mu, sigma, nSamples)
    centers= kMeans(data, 2, 50)
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1])
    ax.scatter(centers[:, 0], centers[:, 1])
    plt.show()
