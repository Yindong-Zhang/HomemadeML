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


    def split(self, feature_index, split_value):
        inds = self._x[:, feature_index] < split_value
        return Dataset(self._x[inds], self._y[inds]), Dataset(self._x[np.logical_not(inds)], self._y[np.logical_not(inds)])


class SplitInfo():
    def __init__(self, feature_index, split_value, split_gain):
        self.split_feature= feature_index
        self.split_value= split_value
        self.split_gain= split_gain

class TreeNode():
    def __init__(self, lChild= None, rChild= None, splitInfo= None, predict_value= None):
        self.l_child= lChild
        self.r_child= rChild
        self.split_info= splitInfo
        self.predict_value= predict_value





def calGini(count_dict, n_samples):
    """

    :param count_dict: a count dictionary with category: count pair
    :return:
    """
    tmp= 0
    for cat_count in count_dict.values():
        tmp += np.square(cat_count / n_samples)
    return 1-tmp


def majority(y):
    count_dict= count_category(y)
    max_value = float("-inf")
    maj_key = None
    for key, value in count_dict.items():
        if value > max_value:
            maj_key = key

    return maj_key


def count_category(y):
    count_dict= {}
    for label in y:
        if label not in count_dict:
            count_dict[label]= 1
        else:
            count_dict[label] += 1
    return count_dict

def mse(y):
    return np.sum(y * y) - np.square(np.mean(y))

def gini(y):
    return calGini(count_category(y), len(y))

class CARTModel():
    def __init__(self, task, max_depth, min_sample_per_leaf, min_gain = 0):
        """

        :param task: string "regression" or "classification", correspond to mse or gini loss
        :param max_depth:
        :param min_sample_per_leaf:
        :param min_gain:
        """
        if task not in ("reg", "cls"):
            raise Exception("invalid task argument")
        self._task = task
        self._max_depth= max_depth
        self._min_sample_per_leaf= min_sample_per_leaf
        self._cart= None
        self._data= None
        self._min_gain = min_gain
        if self._task == "reg":
            self._loss_func = mse
            self._pred_func = np.mean
        else:
            self._loss_func = gini
            self._pred_func = majority



    def __findBestSplitForClassification(self, dataset, fid, cat_count):
        n_samples = dataset.n_samples
        l_count_dict = {}
        l_count = 0
        for key in cat_count:
            l_count_dict[key] = 0
        r_count_dict = cat_count
        r_count = dataset.n_samples
        min_loss = float("inf")
        min_split = -1
        feature = dataset._x[:, fid]
        agst_feat = np.argsort(feature)
        for s_id in agst_feat:
            l_count_dict[dataset._y[s_id]] += 1
            r_count_dict[dataset._y[s_id]] -= 1
            l_count += 1
            r_count -= 1

            if l_count < self._min_sample_per_leaf:
                continue
            if r_count < self._min_sample_per_leaf:
                break

            loss = l_count / n_samples * calGini(l_count_dict, l_count) + r_count / n_samples * calGini(
                r_count_dict, r_count)
            if loss < min_loss:
                min_split = dataset._x[s_id, fid]
                min_loss = loss
        return min_loss, min_split



    def __findBestSplitForRegresssion(self, dataset, f_ind):
        n_samples = dataset.n_samples
        min_loss = float("inf")
        min_split = -1
        feature = dataset._x[:, f_ind]
        l_count = 0
        count = dataset.n_samples
        l_sum = 0
        sum = np.sum(feature)
        l_sqr_sum = 0
        sqr_sum = np.sum(feature * feature)
        agst_feat = np.argsort(feature)
        for s_id in agst_feat:
            l_count += 1
            r_count = count - l_count
            l_sum += feature[s_id]
            r_sum = sum - l_sum
            l_sqr_sum += feature[s_id] * feature[s_id]
            r_sqr_sum = sqr_sum - l_sqr_sum
            l_mse = l_sqr_sum / l_count - np.square(l_sum / l_count)
            r_mse = r_sqr_sum / r_count - np.square( r_sum / r_count)

            if l_count < self._min_sample_per_leaf:
                continue
            if r_count < self._min_sample_per_leaf:
                break

            loss = l_count / n_samples * l_mse + r_count / n_samples * r_mse
            if loss < min_loss:
                min_split = dataset._x[s_id, f_ind]
                min_loss = loss
        return min_loss, min_split


    def __chooseBestFeatureSplit(self, dataset):
        min_loss = float("inf")
        best_fid = -1
        best_split = float("inf")
        if self._task == "cls":
            cat_count = count_category(dataset._y)

        for fid in range(dataset.n_features):
            # print(fid)
            if self._task == "cls":
                loss, split_value = self.__findBestSplitForClassification(dataset, fid, cat_count)
            else:
                loss, split_value = self.__findBestSplitForRegresssion(dataset, fid)

            if loss < min_loss:
                best_fid, min_loss, best_split = fid, loss, split_value

        return best_fid, min_loss, best_split

    def __growTree(self, dataset, depth):
        # 为当前数据集，构建决策树
        cur_loss= self._loss_func(dataset._y)
        cur_node= TreeNode()
        cur_node.predict_value= self._pred_func(dataset._y)
        if depth < self._max_depth or dataset.n_samples > self._min_sample_per_leaf:
            fid, min_loss, split_value= self.__chooseBestFeatureSplit(dataset)
            max_gain = cur_loss - min_loss
            l_dataset, r_dataset= dataset.split(fid, split_value)
            if max_gain > self._min_gain:
                cur_node.split_info= SplitInfo(fid, split_value, max_gain)
                cur_node.l_child= self.__growTree(l_dataset, depth + 1)
                cur_node.r_child= self.__growTree(r_dataset, depth + 1)
            return cur_node
        else:
            return cur_node


    def fit(self, x, y):
        """

        :param x: numpy array in shape (n_samples, n_features)
        :param y: numpy array in dtype int and shape (n_samples, )
        :return:
        """
        self._data = Dataset(x, y)

        self._cart = self.__growTree(self._data, 0)

    def __predict(self, x, cur_node):
        """

        :param x: a feature vector in shape (n_features, )
        :param cur_node:
        :return:
        """
        if cur_node.l_child is None and cur_node.r_child is None:
            return cur_node.predict_value
        else:
            f_id = cur_node.split_info.split_feature
            split_value = cur_node.split_info.split_value
            if x[f_id] <= split_value:
                return self.__predict(x, cur_node.l_child)
            else:
                return self.__predict(x, cur_node.r_child)

    def predict(self, x):
        """

        :param x: sample features in shape (n_samples, features)
        :return:
        """
        n_samples= x.shape[0]
        p= np.zeros((n_samples, ))
        for i in range(n_samples):
            p[i] = self.__predict(x[i, :], self._cart)

        return p

if __name__ == "__main__":
    x = np.random.rand(1000, 100)
    y = np.random.randint(0, 5, (1000, ))
    m = CARTModel("cls", 5, 20)
    m.fit(x, y)
    x_p = np.random.rand(10, 100)
    print(m.predict(x) - y)
    print(m.predict(x_p))


