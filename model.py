from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import random

# all copyrights by linzhongping
class Bagging(object):
    def __init__(self, n_classifier = 10, classifier= 'SVM'):
        self.n_classifier = n_classifier
        if classifier == 'SVM':
            self.classifier = LinearSVC(max_iter=10000)
        if classifier == 'NB':
            self.classifier = GaussianNB()
        if classifier == 'DT':
            self.classifier = DecisionTreeClassifier()

    def random_sample(self, X ,y, sample_num = 200, random_state = 0):
        train_X = []
        train_y = []
        while sample_num > 0:

            index = random.randint(0, len(X) - 1)
            train_X.append(X[index])
            train_y.append(y[index])
            sample_num -= 1
        return train_X, train_y

    def train_base_classifier(self, train_X, train_y, test_X):
        predictor = self.classifier
        if not predictor:
            return 'parameter error'
        predictor.fit(train_X,train_y)

        preds = predictor.predict(test_X)
        return preds



    def integer_results(self, method = 'majority_voting', init_results = None, weight_vector = None):
        if method == 'majority_voting':
            return self.majority_voting(init_results)
        if method == 'plurality_voting':
            return self.plurality_voting(init_results)
        if method == 'weighted_voting':
            return self.weighted_voting(weight_vector, init_results)


    def majority_voting(self, init_results):

        pred_result = []
        init_results = np.array(init_results)

        for j in range(len(init_results[0])):
            pos = 0
            neg = 0
            for i in range(self.n_classifier):
                if init_results[i][j] == 1:  #预测为正
                    pos += 1
                else:
                    neg += 1
            print(pos,neg)
            if pos > neg and pos / (pos+neg) >= 0.5:
                pred_result.append(1)
            else:
                pred_result.append(0) # reject
        return pred_result

    def plurality_voting(self, init_results):
        pred_result = []
        arr = np.array(init_results)
        for j in range(arr.shape[1]):
            a = np.array(arr[:,j])
            b = np.bincount(a)
            pred_result.append(np.argmax(b))
            # pos = 0
            # neg = 0
            # for i in range(self.n_classifier):
            #     if init_results[i][j] == 1:  # 预测为正
            #         pos += 1
            #     else:
            #         neg += 1
            # if pos >= neg:
            #     pred_result.append(1)
            # else:
            #     pred_result.append(0)
        return pred_result

    def weighted_voting(self, weight_vector, init_results):
        pred_result = []
        for j in range(len(init_results[0])):
            prob = 0.0
            for i in range(self.n_classifier):
                prob += init_results[i][j] * weight_vector[i]  # 预测为正
            if prob >= 0.5:
                pred_result.append(1)
            else:
                pred_result.append(0)
        return pred_result

    def compute_accuracy(self,pred_y, test_y):
        return accuracy_score(test_y, pred_y)