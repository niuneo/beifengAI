# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/24
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

# 1. 产生数据
N = 1000
n_centers = 4
X, Y = make_blobs(n_samples=N, n_features=2, centers=n_centers, cluster_std=2.0, random_state=14)

# 2. 模型构建
kmeans = KMeans(random_state=28)
algo = GridSearchCV(estimator=kmeans, param_grid={"n_clusters": [2, 3, 4, 5, 6, 7, 8]})
algo.fit(X)

# 3. 模型对数据做一个预测
x_test = [
    [-4, 8],
    [-3, 7],
    [0, 5],
    [0, -5],
    [8, -8],
    [9, 9]
]
print("预测值:{}".format(algo.predict(x_test)))

# 获取属性参数
print("簇中心点坐标:")
print(algo.best_estimator_.cluster_centers_)
print("目标函数的损失值:")
print(algo.best_estimator_.inertia_)
print(algo.best_estimator_.score(X))
