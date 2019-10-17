# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/15
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

# 1. 产生模拟数据
X, y = make_classification(n_samples=80000, random_state=28)

# 2. 数据划分
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=28)

# 3. 直接模型训练
print("用于最终模型训练的数据形状为:{}".format(x_train.shape))
algo = LogisticRegression()
algo.fit(x_train, y_train)

# 4. 模型效果评估
y_train_pred = algo.predict(x_train)
y_test_pred = algo.predict(x_test)
y_pred = algo.predict(X)
print("训练数据上的准确率:{}".format(accuracy_score(y_train, y_train_pred)))
print("测试数据上的准确率:{}".format(accuracy_score(y_test, y_test_pred)))
print("所有数据上的准确率:{}".format(accuracy_score(y, y_pred)))
print("训练数据上的混淆矩阵:\n{}".format(confusion_matrix(y_train, y_train_pred)))
print("测试数据上的混淆矩阵:\n{}".format(confusion_matrix(y_test, y_test_pred)))
print("所有数据上的混淆矩阵:\n{}".format(confusion_matrix(y, y_pred)))
print("训练数据上的分类报告:\n{}".format(classification_report(y_train, y_train_pred)))
print("测试数据上的分类报告:\n{}".format(classification_report(y_test, y_test_pred)))
print("所有数据上的分类报告:\n{}".format(classification_report(y, y_pred)))
