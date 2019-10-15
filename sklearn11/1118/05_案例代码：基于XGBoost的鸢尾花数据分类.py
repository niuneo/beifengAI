# -- encoding:utf-8 --
"""
@File : 05_案例代码：基于XGBoost的鸢尾花数据分类
@Author: Octal_H
@Date : 2019/10/15
@Desc : 
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

mpl.rcParams['font.sans-serif'] = [u'simHei']

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = '../datas/iris.data'
names = ['A', 'B', 'C', 'D', 'cla']
df = pd.read_csv(path, header=None, names=names)

# 2. 数据清洗

# # 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = df[names[0:-1]]
Y = df[names[-1]]
# print(Y)
label_encoder = LabelEncoder()
label_encoder.fit(Y)
Y = label_encoder.transform(Y)
# 这里得到的序号其实就是classes_这个集合中对应数据的下标
# print(label_encoder.classes_)
# true_label = label_encoder.inverse_transform([0, 1, 2, 0])
# print(true_label)
# print(Y)

# 4. 数据分割
# train_size: 给定划分之后的训练数据的占比是多少，默认0.75
# random_state：给定在数据划分过程中，使用到的随机数种子，默认为None，使用当前的时间戳；给定非None的值，可以保证多次运行的结果是一致的。
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 5. 特征工程的操作
# NOTE: 不做特征工程

# 6. 模型对象的构建
"""
max_depth=3, 树的深度
learning_rate=0.1, 学习率，相当于每个子模型的可信度
n_estimators=100, 子模型的数目
objective="binary:logistic", 损失函数
booster='gbtree', 求解方式
"""
algo = XGBClassifier(n_estimators=10, objective="binary:logistic", max_depth=3)

# 7. 模型的训练
algo.fit(x_train, y_train)

# 8. 模型效果评估
train_predict = algo.predict(x_train)
test_predict = algo.predict(x_test)
print("测试集上的效果(准确率):{}".format(algo.score(x_test, y_test)))
print("训练集上的效果(准确率):{}".format(algo.score(x_train, y_train)))
print("测试集上的效果(分类评估报告):\n{}".format(classification_report(y_test, test_predict)))
print("训练集上的效果(分类评估报告):\n{}".format(classification_report(y_train, train_predict)))

# 9. 其它
print("返回的预测概率值:\n{}".format(algo.predict_proba(x_test)))

# 10. 其他特殊的API
print("各个特征属性的重要性权重:\n{}".format(algo.feature_importances_))

# 返回叶子节点下标
print("*" * 100)
x_test2 = x_test.iloc[:2, :]
print(x_test2)
# apply方法返回的是叶子节点下标
print(algo.apply(x_test2))
