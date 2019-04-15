from GenerateData import Random_Dataset, Load_Dataset
from model import Bagging
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
NUM_CLASSIFIER = 10
# for i in range(2002,2002 + 15 ):
# 生成一个随机数据集

# rd = Random_Dataset(random_state = 0 )
# X,y = rd.generate()

# 载入sklearn带的数据集

X,y = Load_Dataset().load_UCI_dataset('dataset/KEEL/hayes-roth.dat')

# 分隔数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.362, random_state=1996)
for classifier in ['DT','SVM','NB']:
    print('——分类器：', classifier)
# 初始化bagging
    bagging = Bagging(n_classifier= NUM_CLASSIFIER, classifier= classifier)
    results = []
    average_acc = []
    for i in range(NUM_CLASSIFIER):
        x_sample, y_sample =bagging.random_sample(X_train,y_train, sample_num=len(X_train), random_state=i)
        pred = bagging.train_base_classifier(x_sample,y_sample, X_test)
        average_acc.append(bagging.compute_accuracy(pred, y_test))
        # rd.show_data_set(X_test, pred)
        # print(pred)
        results.append(pred)
    print('——————T个弱分类器平均准确率结果:',np.mean(average_acc))
    pred_y = bagging.integer_results(method = 'plurality_voting', init_results=results)

    print('——————集成结果——————', bagging.compute_accuracy(pred_y,y_test))
    print(np.mean(average_acc),'\t',bagging.compute_accuracy(pred_y,y_test))





# 算ROC
# fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_y, pos_label= 1)
#
# import matplotlib.pyplot as plt
#
# plt.plot(fpr,tpr,marker = 'o')
# plt.show()
#
# AUC = metrics.auc(fpr,tpr)
# print('——AUC=', AUC)

