# -- encoding:utf-8 --
"""
@File : 04_案例代码：基于sklearn的GMM高斯混合聚类代码实现
@Author: Octal_H
@Date : 2019/10/29
@Desc : 
"""
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

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

"""
n_components: 给定GMM中到底有几个高斯分布
covariance_type: 给定的是方差属性的类型是什么， 可选值： {‘full’, ‘tied’, ‘diag’, ‘spherical’}
-- full: 针对每个高斯分布的特征属性给定一个协方差矩阵
-- tied：所有的高斯分布公用一个协方差矩阵
-- diag：针对每个高斯分布的特征属性给定一个方差矩阵，也就是只有对角线上有值的协方差矩阵
-- spherical：每个高斯分布的所有特征属性公用一个方差
max_iter: 给定的最大允许的迭代次数
"""
algo = GaussianMixture(n_components=2, covariance_type='full', max_iter=100)
algo.fit(data)

print("最终的π值:{}".format(algo.weights_))
print("最终的均值:\n{}".format(algo.means_))
print("最终的协方差矩阵:\n{}".format(algo.covariances_))

# 3. 基于训练好的模型产生预测值
x_test = np.array([
    [0.0, 0.0, 0.0],
    [5.0, 6.0, 7.0],
    [2.5, 1.5, 1.5],
    [6.0, 8.0, 4.0]
])
print("预测的类别为:{}".format(algo.predict(x_test)))
print("预测为各个类别的概率:\n{}".format(algo.predict_proba(x_test)))