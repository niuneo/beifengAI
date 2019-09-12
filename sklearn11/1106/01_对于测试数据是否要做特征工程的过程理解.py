# -- encoding:utf-8 --
'''
@File : 01_对于测试数据是否要做特征工程的过程理解.py
@Author: Octal_H
@Date : 2019/9/11
@Desc : 
'''
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

mpl.rcParams['font.sans-serif'] = [u'simHei']

# 2.获取特征属性X和目标属性Y
x_train = np.array([
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6]
]).astype(np.float32)

# 模拟一个数据转换的操作
x_train = x_train - [1, 4]
print(x_train)

y_train = np.array([0.1, 0.2, 0.31, 0.4])

x_test = np.array([[5, 7], [6, 8]]).astype(np.float32)
x_test = x_test - [1, 4]
print(x_test)

print("**"*8)
# 4.模型构建
# a.将DataFrame转为numpy中的矩阵形式
x = np.mat(x_train)
y = np.mat(y_train).reshape(-1, 1)

# b. 直接解析式求解theta值
theta = (x.T * x).I * x.T * y
print(theta)

# 5. 使用训练出来的模型参数theta对测试数据做一个预测
predict_y = np.mat(x_test) * theta
print(predict_y)

