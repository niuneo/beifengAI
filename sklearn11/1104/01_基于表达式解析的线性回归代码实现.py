# -- encoding:utf-8 --
"""
@File : 01_基于表达式解析的线性回归代码实现.py
@Author: Octal_H
@Date : 2019/9/17
@Desc : 现有一批描述家庭用电情况的数据，对数据进行算法模型预测，并最终得到预测
模型（每天各个时间段和功率之间的关系、功率与电流之间的关系等）
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1.数据加载
path = '../datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';')
# 查看数据信息
# df.info()
# print(df.head(2))

# 2.获取特征属性X和目标属性Y
X = df.iloc[:, 2:4]
# X['b'] = pd.Series(data=np.ones(shape=X.shape[0]))
Y = df.iloc[:, 5]
# print(X.head(5))
# print(Y[:5])

# 前900条数据
# print(X.iloc[:900, :].shape)
# 后100条数据
# print(X.iloc[900:, :].shape)

# 3.划分训练集和测试数据集
# train_size：给定划分之后的训练数据的占比是多少，默认0.75
# random_state：给定在数据划分过程中，使用到的随机数种子，默认为None，使用当前的时间戳；给定非None的值，可以保证多次运行的结果是一样的
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练数据X的格式：{}，以及类型：{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式：{}，以及类型：{}".format(x_test.shape, type(x_test)))
print("训练数据Y的格式：{}，以及类型：{}".format(y_train.shape, type(y_train)))
print("测试数据Y的格式：{}，以及类型：{}".format(y_test.shape, type(y_test)))

# 4.模型的构建
# a.将DateFrame转化为numpy中的矩阵形式
x = np.mat(x_train)
y = np.mat(y_train).reshape(-1, 1)

# b.直接解析式求解theta值
theta = (x.T * x).I * x.T * y
print('theta值：{}'.format(theta))
"""
theta值：[[ 4.2000866 ]
 [ 1.37131883]]
 """

# 5.使用训练出来的模型参数theta对测试数据做一个预测
predict_y = np.mat(x_test) * theta

# 6.可以考虑一下画图看一下效果
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', label=u'真实值')
plt.plot(t, predict_y, 'b-', label=u'预测值')
plt.legend(loc='lower right')
plt.show()