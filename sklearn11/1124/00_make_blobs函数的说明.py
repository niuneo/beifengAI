# -- encoding:utf-8 --
"""
@File : 00_make_blobs函数的说明
@Author: Octal_H
@Date : 2019/10/16
@Desc : 
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


"""
功能：产生一个服从给定均值和方差的高斯分布数据，最终返回样本数据组成的x矩阵，以及对应的类别Y矩阵
make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0,
               center_box=(-10.0, 10.0), shuffle=True, random_state=None):
n_samples：产生多少样本
n_features：每个样本产生多少特征
centers：中心点数目或者中心点坐标
cluster_std：每个中心/簇/类别对应的所有样本的标准差
center_box：中心点坐标的取值范围，随机中心点的时候
shuffle：是否对产生的数据打乱顺序

"""
x, y = make_blobs(n_samples=100, n_features=2, centers=2, center_box=(0, 10))
print(x.ravel())
print(y)

"""
c：表示的是色彩或颜色序列，可选，默认蓝色’b’。
但是c不应该是一个单一的RGB数字，也不应该是一个RGBA的序列，因为不便区分。c可以是一个RGB或RGBA二维行数组。
"""
plt.scatter(x[:, 0], x[:, 1], c=y, s=30)
plt.show()