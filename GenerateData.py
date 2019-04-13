'''
copyright by linzhongping
'''
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import datasets
import numpy as np
# 随机生成数据集
class Random_Dataset(object):

    def __init__(self, sample_num = 2000, class_num = 2, feature_num = 20, random_state = 123):
        self.sample_num = sample_num
        self.class_num = class_num
        self.feature_num = feature_num
        self.random_state = random_state

    def generate(self):
        return make_classification(n_samples=self.sample_num, n_features=self.feature_num, n_classes= self.class_num , random_state= self.random_state)

    def show_data_set(self, X, y):
        tsne = TSNE(init = 'pca')
        two_dimension_X = tsne.fit_transform(X,y)

        #show
        plt.scatter(two_dimension_X[:,0], two_dimension_X[:,1],c = y, s=10, marker = 'o')
        plt.show()


class Load_Dataset(object):
    def __init__(self):
        pass

    def load_iris(self):
        data = datasets.load_iris()
        X = data['data'][:,(2,3)]
        y = (data['target']==2).astype(np.float64)
        return X,y

    def load_breast(self):
        data = datasets.load_breast_cancer()
        X = data['data']
        y = data['target']
        return X,y

    def show_data_set(self, X, y):
        tsne = TSNE(init = 'pca')
        two_dimension_X = tsne.fit_transform(X,y)

        #show
        plt.scatter(two_dimension_X[:,0], two_dimension_X[:,1],c = y, s=10, marker = 'o')
        plt.show()


