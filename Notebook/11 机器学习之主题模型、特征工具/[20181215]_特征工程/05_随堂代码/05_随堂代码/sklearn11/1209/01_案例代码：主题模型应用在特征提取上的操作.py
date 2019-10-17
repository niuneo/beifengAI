# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/9
"""

import numpy as np
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

# 1. 加载保存好的经过词袋法处理的数据
# NOTE: 要求代码所在的文件夹中有一个data.npy文件
data = np.load('data.npy')
print("原始数据格式:{}".format(data.shape))

# 2. 基于主题模型做一个降维的操作
nmf = NMF(n_components=2, solver='cd')
data = nmf.fit_transform(data)
print("处理后的数据格式:{}".format(data.shape))

# 3. 基于降维后的数据做聚类操作
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

# 4. 打印相关信息
print(kmeans.predict(data))
print(kmeans.cluster_centers_)
