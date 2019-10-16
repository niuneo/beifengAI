# -- encoding:utf-8 --
"""
@File : 03_MiniBatchKMeans算法案例代码
@Author: Octal_H
@Date : 2019/10/16
@Desc : 
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans

# 1.产生数据
N = 1000
n_centers = 4
X, Y = make_blobs(n_samples=N, n_features=2, centers=n_centers, cluster_std=2.0, random_state=14)

# 2.模型构建
algo = MiniBatchKMeans(n_clusters=n_centers, random_state=28, batch_size=50)
algo.fit(X)

# 3.对数据做一个预测
x_test = [
    [-4, 8],
    [-3, 7],
    [0, 5],
    [0, -5],
    [8, -8],
    [9, 9]
]

print("预测值：{}".format(algo.predict(x_test)))

# 获取属性参数
print("簇中心点坐标:")
print(algo.cluster_centers_)
print("目标函数的损失值:")
print(algo.inertia_)
print(algo.score(X))
print("训练数据对应的簇的id:\n{}".format(algo.labels_))

