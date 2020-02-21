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


if __name__ == '__main__':
    nSamples= 500
    rndState= np.random.RandomState(20)
    data = rndState.randn(nSamples, 2)
    centers= kMeans(data, 5, 500)
