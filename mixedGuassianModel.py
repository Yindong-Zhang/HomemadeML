import numpy as np
from functools import reduce, partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os
from scipy.io import loadmat
from config import THRESH, log

def load_data(filepath):
    mat = []
    for file in os.listdir(filepath):
        matData = loadmat(os.path.join(filepath, file))
        mat.append(matData)
    x1 = mat[0]['x']
    y1= mat[0]['y']
    x2 = mat[1]['x']
    y2 = mat[1]['y']
    x3 = mat[2]['x']
    y3 = mat[2]['y']
    return (x1, y1), (x2, y2), (x3, y3)



def normPdf(x,mu,sigma):
    '''
    计算均值为mu，标准差为sigma的正态分布函数的密度函数值
    :param x: x值
    :param mu: 均值
    :param sigma: 标准差
    :return: x处的密度函数值
    '''
    return (1./(2*np.pi*np.sqrt(np.linalg.det(sigma)))) * np.exp(- 1/2 * reduce(np.inner, [(x-mu), np.linalg.inv(sigma), (x - mu)]))



def em(dataArray,k,mu,sigma,step = 20):
    '''
    em算法估计高斯混合模型
    :param dataNum: 已知数据个数
    :param k: 每个高斯分布的估计系数
    :param mu: 每个高斯分布的估计均值
    :param sigma: 每个高斯分布的估计标准差
    :param step:迭代次数
    :return: em 估计迭代结束估计的参数值[k,mu,sigma]
    '''
    # 高斯分布个数
    n = len(k)
    # 数据个数
    dataNum = len(dataArray)
    # 初始化gama数组
    probArray = np.zeros((n,dataNum))
    #
    llOld= - np.inf
    for s in range(step):
        # E step:
        for j in range(dataNum):
            for i in range(n):
                probArray[i][j] = k[i]*normPdf(dataArray[j],mu[i],sigma[i])

        # 归一化概率
        gammaArray= probArray / np.sum(probArray, axis= 0)

        llNew= np.sum(np.log(np.sum(probArray, axis= 0)))

        if llNew - llOld <= THRESH:
            log.info('EM stop at step %s.' %(s, ))
            break

        # M step:
        for i in range(n):
            # 更新 mu
            mu[i] = np.dot(gammaArray[i], dataArray)/np.sum(gammaArray[i])
            # 更新 sigma
            sigma[i] = reduce(np.matmul, [dataArray.T, np.diag(gammaArray[i]), dataArray])/np.sum(gammaArray[i])
            # 更新系数k
            k[i] = np.sum(gammaArray[i])/dataNum

        llOld= llNew

        log.debug('Step %s: ' %(s, ))
        log.debug(' k: %s' %(k, ))
        log.debug(' mu: %s' %(mu, ))
        log.debug(' sigma: %s' %(sigma, ))
        log.debug(' LL: %s' %(llOld, ))
    return k,mu,sigma

def mixturePDF(k, mu, sigma, x, y):
    assert len(k) == len(mu) == len(sigma), "invalid guassian mixture model parameter."
    classCount= len(k)
    data= np.array((x, y))
    res= 0
    for i in range(classCount):
        res += k[i] * normPdf(data, mu[i], sigma[i])

    return res


def plot_surface(dx, dy, k, mu, sigma, gridCount= 50, name= 'test'):
    figPath= 'figures/'
    x= np.linspace(0, 1, gridCount)
    y= np.linspace(0, 1, gridCount)
    hx, hy= np.meshgrid(x, y)
    hf= np.vectorize(partial(mixturePDF, k, mu, sigma))(hx, hy)
    # ax= plt.subplot(projection= '3d')
    # ax.plot_surface(hx, hy, hf)
    fig, ax= plt.subplots()
    ax.pcolormesh(hx, hy, hf)
    ax.scatter(dx, dy)
    plt.show()
    fig.savefig(os.path.join(figPath, name))

if __name__ == '__main__':
    # print(normPdf(np.array((0, 1)), np.array((0, 0)), np.array((( 1, 0), (0, 1)))))
    # plot_surface([0.5, 0.5], [np.array((0, 0)), np.array((1, 1))], [np.array(((0.2, 0), (0, 0.2))), np.array(((0.2, 0), (0, 0.2)))])
    (x1, y1), _, _ = load_data(filepath= '/home/yindong/Downloads/数据')
    # initialize parameter
    nComponent= 6
    alpha= np.ones(nComponent)
    ki = np.random.dirichlet(alpha)
    mui= np.random.rand(nComponent, 2)
    sigmai=np.apply_along_axis(np.diag, 1, np.random.rand(nComponent, 2))
    k, mu, sigma= em(np.hstack((x1, y1)), ki, mui, sigmai, step= 100)

    configStr= 'test_%s' %(nComponent, )
    datafile= os.path.join('data', '%s.txt' %(configStr, ))
    with open(datafile, 'w') as f:
        f.writelines([repr(k), '\n', repr(mu), '\n', repr(sigma)])
    log.info('alpha: %s' %(k, ))
    log.info('mu: %s' %(mu, ))
    log.info('sigma: %s' %(sigma, ))

    plot_surface(x1, y1, k, mu, sigma, name= configStr)