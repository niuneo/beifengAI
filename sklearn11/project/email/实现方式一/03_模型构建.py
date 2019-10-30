# -- encoding:utf-8 --
"""
@File : 03_模型构建
@Author: Octal_H
@Date : 2019/10/30
@Desc : 
"""
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib

# 1. 加载数据
train_datas = np.load('./data/traim_email_datas.npy')
test_datas = np.load('./data/test_email_datas.npy')
train_labels = np.load('./data/traim_email_labels.npy')
test_labels = np.load('./data/test_email_labels.npy')

# 2. 模型训练
print("用于模型训练的数据格式:{}".format(train_datas.shape))
algo = BernoulliNB(alpha=1.0, binarize=0.0005)
# algo = LogisticRegression()
algo.fit(train_datas, train_labels)

#  3. 模型效果评估
print("训练数据上的准确率:", end='')
print(algo.score(train_datas, train_labels))
print("测试数据上的准确率:", end='')
print(algo.score(test_datas, test_labels))
print("训练数据上的混淆矩阵:")
print(metrics.confusion_matrix(train_labels, algo.predict(train_datas)))
print("测试数据上的混淆矩阵:")
print(metrics.confusion_matrix(test_labels, algo.predict(test_datas)))
print("训练数据上的分类评估报告:")
print(metrics.classification_report(train_labels, algo.predict(train_datas)))
print("测试数据上的分类评估报告:")
print(metrics.classification_report(test_labels, algo.predict(test_datas)))

# 4. 模型持久化
joblib.dump(algo, './model/algo.pkl')