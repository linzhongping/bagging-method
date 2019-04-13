from GenerateData import Random_Dataset, Load_Dataset
from model import Bagging
from sklearn.model_selection import train_test_split
from sklearn import metrics
NUM_CLASSIFIER = 10

# 生成一个随机数据集
rd = Random_Dataset()
X,y = rd.generate()

# 载入sklearn带的数据集

# X,y = Load_Dataset().load_breast()

# 分隔数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=1996)

# print(X_train,X_test,y_train,y_test)
print(len(X_test))
# 初始化bagging
bagging = Bagging(n_classifier= NUM_CLASSIFIER, classifier= 'DT')
results = []
print('——————弱分类器结果——————')
for i in range(NUM_CLASSIFIER):
    x_sample, y_sample =bagging.random_sample(X_train,y_train, sample_num=200, random_state=i)

    pred = bagging.train_base_classifier(x_sample,y_sample, X_test)
    # 输出单分类器精确率

    # print(bagging.compute_accuracy(pred, y_test))
    # rd.show_data_set(X_test, pred)

    results.append(pred)

pred_y = bagging.integer_results(method = 'plurality_voting', init_results=results)
print('——————集成结果——————')
print(bagging.compute_accuracy(pred_y,y_test))

# 算ROC
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_y, pos_label= 1)

import matplotlib.pyplot as plt

plt.plot(fpr,tpr,marker = 'o')
plt.show()

AUC = metrics.auc(fpr,tpr)
print('——AUC=', AUC)

