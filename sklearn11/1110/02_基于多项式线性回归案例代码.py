# -- encoding:utf-8 --
"""
@File : 02_基于多项式线性回归案例代码.py
@Author: Octal_H
@Date : 2019/9/30
@Desc : 
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
import time

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1.加载数据(数据一般存在于磁盘或者数据库)
path = '../datas/household_power_consumption_1000_2.txt'
df = pd.read_csv(path, sep=';')


# 2.数据清洗
df.replace('?', np.nan, inplace=True)
df = df.dropna(axis=0, how='any')
# df.info()


# 3.根据需求获取最原始的特征属性矩阵X和目标属性Y
def date_format(t):
    date_str = time.strptime(' '.join(t), '%d/%m/%Y %H:%M:%S')
    return (date_str.tm_year, date_str.tm_mon, date_str.tm_mday, date_str.tm_hour, date_str.tm_min, date_str.tm_sec)


X = df.iloc[:, 0:2]
X = X.apply(lambda row: pd.Series(date_format(row)), axis=1)
Y = df.iloc[:, 4].astype(np.float32)
# print(X)


# 4.数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练数据X的格式：{}，以及类型：{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式：{}，以及类型：{}".format(x_test.shape, type(x_test)))
print("训练数据Y的格式：{}，以及类型：{}".format(y_train.shape, type(y_train)))
print("测试数据Y的格式：{}，以及类型：{}".format(y_test.shape, type(y_test)))


# 5.特征工程的操作
"""
degree:给定做几阶的扩展，默认是2阶扩展
"""
ploy = PolynomialFeatures(degree=4)
x_train = ploy.fit_transform(x_train)
x_test = ploy.transform(x_test)


# 6.模型对象的构建
print('用于模型训练的数据形状：{}'.format(np.shape(x_train)))
algo = LinearRegression(fit_intercept=True)


# 7.模型的训练
algo.fit(x_train, y_train)


# 8.模型效果评估
print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(algo.coef_))
print("截距项值:{}".format(algo.intercept_))
# b. 直接通过评估相关的API查看效果
print("模型在训练数据上的效果(R2)：{}".format(algo.score(x_train, y_train)))
# 在测试的时候对特征属性数据必须做和训练数据完全一样的操作
print("模型在测试数据上的效果(R2)：{}".format(algo.score(x_test, y_test)))


# 9.模型保存\模型持久化
"""
方式一：直接保存预测结果
方式二：将模型持久化为磁盘文件
方式三：将模型参数保存数据库
"""

# 10.可以考虑一下画图看一下效果
predict_y = algo.predict(x_test)
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', label=u'真实值')
plt.plot(t, predict_y, 'b-', label=u'预测值')
plt.legend(loc='lower right')
plt.show()