# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/24
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

"""
功能：产生一个服从给定均值和方差的高斯分布数据，最终返回样本数据组成的x矩阵，以及对应的类别Y矩阵
make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)
n_samples：产生多少样本
n_features: 每个样本产生多少个特征
centers: 中心点数目或者中心点坐标
cluster_std：每个中心/簇/类型对应的所有样的标准差
center_box: 中心点坐标的取值范围, 随机中心点的时候
shuffle: 是否对产生的数据打乱顺序
"""
x, y = make_blobs(n_samples=100, n_features=2, centers=2, center_box=(0, 10))
# print(x.ravel())
# print(y)

plt.scatter(x[:, 0], x[:, 1], c=y, s=30)
plt.show()
