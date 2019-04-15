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
        X = data['data']
        y = data['target']
        return X,y

    def load_breast(self):
        data = datasets.load_breast_cancer()
        X = data['data']
        y = data['target']
        return X,y

    def load_digits(self):
        data = datasets.load_digits()
        X = data['data']
        y = data['target']
        return X, y



    def show_data_set(self, X, y):
        tsne = TSNE(init = 'pca')
        two_dimension_X = tsne.fit_transform(X,y)

        #show
        plt.scatter(two_dimension_X[:,0], two_dimension_X[:,1],c = y, s=10, marker = 'o')
        plt.show()

    def load_KEEL_dataset(self, path):
        '''
        the datasets from website KEEL
        :return:
        '''
        X = []
        y = []
        with open(path,'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                l = l.split(',')
                X.append([float(i) for i in l[:-1]])
                y.append(int(l[-1]))

        X = np.array(X)
        y = np.array(y)
        print(X,y)
        return X,y
    def load_UCI_dataset(self, path):
        '''
        the data sets from website UCI
        :param path:
        :return:
        '''
