from GenerateData import Load_Dataset
from model import Bagging
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

NUM_CLASSIFIER = 10

# 载入数据集
# X,y = Load_Dataset().load_UCI_dataset('dataset/UCI/data_banknote_authentication.data') #
X,y = Load_Dataset().load_digits() # 载入手写体数字识别数据集

# 分割数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.368, random_state=1996)


# 初始化bagging
bagging = Bagging(n_classifier= NUM_CLASSIFIER, classifier= 'DT')
results = []
average_acc = []
for i in range(NUM_CLASSIFIER):
    # 自助采样 Bootstrap
    x_sample, y_sample =bagging.random_sample(X_train,y_train, sample_num=len(X_train), random_state=i)

    # 单分类器训练以及预测过程
    pred = bagging.train_base_classifier(x_sample,y_sample, X_test)

    average_acc.append(bagging.compute_accuracy(pred,y_test))

    results.append(pred)

print('T个弱分类器平均准确率结果:',np.mean(average_acc))

# 集成
pred_y = bagging.integer_results(method = 'plurality_voting', init_results=results)
print('集成结果:', bagging.compute_accuracy(pred_y,y_test))

# 可视化部分
# Load_Dataset().show_data_set(X_test,pred_y)  #预测结果可视化
# Load_Dataset().show_data_set(X_test,y_test)  #ground truth可视化


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

