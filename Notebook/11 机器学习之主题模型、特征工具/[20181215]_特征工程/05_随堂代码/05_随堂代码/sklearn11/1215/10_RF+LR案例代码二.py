# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/15
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings('ignore')

# 1. 产生模拟数据
X, y = make_classification(n_samples=80000, random_state=28)

# 2. 数据划分
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=28)
x_train1, x_train2, y_train1, y_train2 = train_test_split(x_train, y_train, test_size=0.5, random_state=28)

# 3. 直接模型训练
# a. 构建随机森林模型
rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=28)
rf.fit(x_train1, y_train1)
# b. 构建LR模型需要的训练数据
x_train_lr = rf.apply(x_train2)
# c. 对叶子节点的数值做一个哑编码操作
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(rf.apply(x_train1))
x_train_lr = one_hot_encoder.transform(x_train_lr).toarray()
# 合并原始数据和扩展的数据
x_train_lr = np.hstack((np.array(x_train2), np.array(x_train_lr)))
# d. 构建LR模型
print("用于最终模型训练的数据形状为:{}".format(x_train_lr.shape))
algo = LogisticRegression()
algo.fit(x_train_lr, y_train2)

# 4. 模型效果评估
tmp = np.hstack((np.array(x_train), np.array(one_hot_encoder.transform(rf.apply(x_train)).toarray())))
y_train_pred = algo.predict(tmp)
tmp = np.hstack((np.array(x_test), np.array(one_hot_encoder.transform(rf.apply(x_test)).toarray())))
y_test_pred = algo.predict(tmp)
tmp = np.hstack((np.array(X), np.array(one_hot_encoder.transform(rf.apply(X)).toarray())))
y_pred = algo.predict(tmp)
print("训练数据上的准确率:{}".format(accuracy_score(y_train, y_train_pred)))
print("测试数据上的准确率:{}".format(accuracy_score(y_test, y_test_pred)))
print("所有数据上的准确率:{}".format(accuracy_score(y, y_pred)))
print("训练数据上的混淆矩阵:\n{}".format(confusion_matrix(y_train, y_train_pred)))
print("测试数据上的混淆矩阵:\n{}".format(confusion_matrix(y_test, y_test_pred)))
print("所有数据上的混淆矩阵:\n{}".format(confusion_matrix(y, y_pred)))
print("训练数据上的分类报告:\n{}".format(classification_report(y_train, y_train_pred)))
print("测试数据上的分类报告:\n{}".format(classification_report(y_test, y_test_pred)))
print("所有数据上的分类报告:\n{}".format(classification_report(y, y_pred)))
