'''
copyright by linzhongping
'''
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import datasets
import numpy as np


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
                y.append(int(l[-1]=='1.0'))

        X = np.array(X)
        y = np.array(y)
        # print(X,y)
        return X,y

    def load_UCI_dataset(self, path):
        '''
        the data sets from website UCI
        :param path:
        :return:
        '''
        X = []
        y = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                l = l.split(',')
                X.append([float(i) for i in l[0:-1]])
                y.append(int(l[-1]))

        X = np.array(X)
        y = np.array(y)
        # print(X,y)
        return X, y


    def show_data_set(self, X, y):
        tsne = TSNE(init = 'pca')
        two_dimension_X = tsne.fit_transform(X,y)

        #show
        plt.scatter(two_dimension_X[:,0], two_dimension_X[:,1],c = y, s=10, marker = 'o')
        plt.show()