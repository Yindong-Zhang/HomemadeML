# my implementation of cart
import math
import numpy as np

class Dataset():
    def __init__(self, x, y):
        """

        :param x: feature array, all numerical, category feature in one-hot format, in numpy array (num sample, num feature) format. no missing value considered.
        :param y: label, numpy array in shape (num sample, ), suppose a classification problem.
        :param ps: argsort feature array column wise, in shape the same as x
        :param index: index of sample in current data
        """
        self._x= x
        self._y= y
        self.n_samples= x.shape[0]
        self.n_features= x.shape[1]
        self._cat_count= self._count_category()


    def split(self, feature_index, split_value):
        inds = self._x[:, feature_index] < split_value
        return Dataset(self._x[inds], self._y[inds]), Dataset(self._x[np.logical_not(inds)], self._y[np.logical_not(inds)])

    def _count_category(self):
        count_dict= {}
        for y in self._y:
            if y not in count_dict:
                count_dict[y]= 1
            else:
                count_dict[y] += 1
        return count_dict

class SplitInfo():
    def __init__(self, feature_index, split_value, split_gain):
        self.split_feature= feature_index
        self.split_value= split_value
        self.split_gain= split_gain

class TreeNode():
    def __init__(self, lChild= None, rChild= None, splitInfo= None, category= None):
        self.l_child= lChild
        self.r_child= rChild
        self.split_info= splitInfo
        self.category= category





def calGini(count_dict, n_samples):
    """

    :param count_dict: a count dictionary with category: count pair
    :return:
    """
    tmp= 0
    for cat_count in count_dict.values():
        tmp += np.square(cat_count / n_samples)
    return 1-tmp


def majority(count_dict):
    max_value = float("-inf")
    maj_key = None
    for key, value in count_dict.items():
        if value > max_value:
            maj_key = key

    return maj_key


class CARTModel():
    def __init__(self, max_depth, min_sample_per_leaf):
        self._max_depth= max_depth
        self._min_sample_per_leaf= min_sample_per_leaf
        self._cart= None
        self._data= None

    def __findBestSplit(self, dataset, fid):
        # 非常低效的算法设计，需要修改
        n_samples = dataset.n_samples
        l_count_dict = {}
        l_count = 0
        for key in dataset._cat_count:
            l_count_dict[key] = 0
        r_count_dict = dataset._cat_count
        r_count = dataset.n_samples
        min_loss = float("inf")
        min_split = -1
        feature = dataset._x[:, fid]
        agst_feat = np.argsort(feature)
        for s_id in agst_feat[self._min_sample_per_leaf: -self._min_sample_per_leaf]:
            l_count_dict[dataset._y[s_id]] += 1
            r_count_dict[dataset._y[s_id]] -= 1
            l_count += 1
            r_count -= 1
            loss = l_count / n_samples * calGini(l_count_dict, l_count) + r_count / n_samples * calGini(
                r_count_dict, r_count)
            if loss < min_loss:
                min_split = dataset._x[s_id, fid]
                min_loss = loss
        return min_loss, min_split

    def __chooseBestFeatureSplit(self, dataset):
        min_loss = float("inf")
        best_fid = -1
        best_split = float("inf")
        for fid in range(dataset.n_features):
            print(fid)
            loss, split_value = self.__findBestSplit(dataset, fid)
            if loss < min_loss:
                best_fid, min_loss, best_split = fid, loss, split_value

        return best_fid, min_loss, best_split

    def __growTree(self, dataset, depth):
        print(depth)
        if depth < self._max_depth or dataset.n_samples > self._min_sample_per_leaf:
            fid, gain, split_value= self.__chooseBestFeatureSplit(dataset)
            cur_node= TreeNode()
            l_dataset, r_dataset= dataset.split(fid, split_value)
            cur_node.l_child= self.__growTree(l_dataset, depth + 1)
            cur_node.r_child= self.__growTree(r_dataset, depth + 1)
            cur_node.split_info= SplitInfo(fid, split_value, gain)
            cur_node.category= majority(dataset._cat_count)
            return cur_node
        else:
            return None


    def fit(self, x, y):
        """

        :param x: numpy array in shape (n_samples, n_features)
        :param y: numpy array in dtype int and shape (n_samples, )
        :return:
        """
        n_samples= x.shape[0]
        self._data = Dataset(x, y)

        self._cart = self.__growTree(self._data, 0)



if __name__ == "__main__":
    x = np.random.rand(1000, 100)
    y = np.random.randint(0, 5, (1000, ))
    m = CARTModel(100, 20)
    m.fit(x, y)


