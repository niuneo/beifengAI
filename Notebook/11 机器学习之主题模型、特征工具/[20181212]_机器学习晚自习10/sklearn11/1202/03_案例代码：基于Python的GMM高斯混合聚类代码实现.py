# -- encoding:utf-8 --
"""
# multivariate_normal API参考链接：http://scipy.github.io/devdocs/generated/scipy.stats.multivariate_normal.html#scipy.stats.multivariate_normal

Create by ibf on 2018/12/2
"""

import numpy as np
from scipy.stats import multivariate_normal

# 1. 产生模拟数据
np.random.seed(28)
N1 = 400
N2 = 100
# 类别1的数据
mean1 = (0, 0, 0)
cov1 = np.diag((1, 2, 3))
data1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=N1)
# 类别2的数据
mean2 = (5, 6, 7)
cov2 = np.array([
    [3, 1, 0],
    [1, 1, 0],
    [0, 0, 5]
])
data2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=N2)
# 合并两个数据
data = np.vstack((data1, data2))

# 2. 构建模型
max_iter = 1000
m, d = data.shape
print("样本数目:{}, 每个样本的维度值:{}".format(m, d))

# 给定初值(模型参数的初始值)
mu1 = data.min(axis=0)
mu2 = data.max(axis=0)
sigma1 = np.identity(d)
sigma2 = np.identity(d)
pi = 0.5
print("初始化的π值:{}".format([pi, 1 - pi]))
print("初始化的均值:\n{}\n{}".format(mu1, mu2))
print("初始化的协方差矩阵:\n{}\n{}".format(sigma1, sigma2))

# 实现EM算法
for idx in range(max_iter):
    # 1. E step：计算在当前模型参数下，各个样本的条件概率
    # a. 根据均值和方差构建对应的多元高斯概率密度函数
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    # b. 获取多元高斯概率密度函数计算样本的概率密度值
    pdf1 = pi * norm1.pdf(data)
    pdf2 = (1 - pi) * norm2.pdf(data)
    # c. 做一个归一化的操作，计算出w
    w1 = pdf1 / (pdf1 + pdf2)
    w2 = 1.0 - w1

    # 2. M step: 基于计算出来的条件概率w，更新模型参数
    # 均值更新
    mu1 = np.dot(w1, data) / np.sum(w1)
    mu2 = np.dot(w2, data) / np.sum(w2)
    # 方差更新
    sigma1 = np.dot(w1 * (data - mu1).T, data - mu1) / np.sum(w1)
    sigma2 = np.dot(w2 * (data - mu2).T, data - mu2) / np.sum(w2)
    # pi的更新
    pi = np.sum(w1) / m
print("最终的π值:{}".format([pi, 1 - pi]))
print("最终的均值:\n{}\n{}".format(mu1, mu2))
print("最终的协方差矩阵:\n{}\n{}".format(sigma1, sigma2))

# 3. 基于训练好的模型产生预测值
x_test = np.array([
    [0.0, 0.0, 0.0],
    [5.0, 6.0, 7.0],
    [2.5, 1.5, 1.5],
    [6.0, 8.0, 4.0]
])
# a. 根据均值和方差构建对应的多元高斯概率密度函数
norm1 = multivariate_normal(mu1, sigma1)
norm2 = multivariate_normal(mu2, sigma2)
# b. 获取多元高斯概率密度函数计算样本的概率密度值
pdf1 = pi * norm1.pdf(x_test)
pdf2 = (1 - pi) * norm2.pdf(x_test)
# c. 做一个归一化的操作，计算出w
w1 = pdf1 / (pdf1 + pdf2)
w2 = 1.0 - w1
print("预测为类别1的概率为:{}".format(w1))
print("预测为类别2的概率为:{}".format(w2))
w = np.vstack((w1, w2))
y_hat = np.argmax(w, axis=0)
print("预测的类别为:{}".format(y_hat))
