import numpy as np
def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))

def logisticRegression(data, label, eta, max_iter= 1024, seed= 31):
    """

    :param data: data array, numpy array in shape (nSamples, nFeatures)
    :param label: label array, numpy array in shape (nSamples, )
    :param eta: learning rate for gradient descent
    :param max_iter: max iteration for gradient descent
    :param seed: seed for random state
    :return:
    """
    assert len(data) == len(label), "inconsistent data shape."
    nSamples= len(data)
    nFeatures= data.shape[1]
    rndState = np.random.RandomState(seed)
    w = rndState.randn(nFeatures)

    for i in range(max_iter):
        q = sigmoid(np.matmul(data, w))
        g = np.matmul((q - label), data)

        w = w - eta * g
        print(w[0])
    return w


if __name__ == "__main__":
    data = np.random.rand(500, 32)
    label= np.random.randint(0, 2, (500, ))
    learning_rate= 0.001
    w= logisticRegression(data, label, learning_rate)
    print(w)