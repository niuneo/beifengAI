# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/9
"""
import numpy as np
from sklearn.decomposition import NMF
from sklearn.neighbors import KDTree

# 1. 加载保存好的经过词袋法处理的数据
# NOTE: 要求代码所在的文件夹中有data.npy和word_name.npy文件
data = np.load('data.npy')
feature_names = np.load('word_name.npy')
print(feature_names)
print("原始数据格式:{}".format(data.shape))

# 2. 基于主题模型做一个降维的操作
nmf = NMF(n_components=2, solver='cd')
doc_2_topic_data = nmf.fit_transform(data)
print("文档和主题之间相关性的数据格式:{}".format(doc_2_topic_data.shape))
topic_2_word_data = nmf.components_
print("主题和单词之间的相关性的数据格式:{}".format(topic_2_word_data.shape))

"""
eg：
现在有一个文档和主题之间的相关性是:[0.9,0.1];
那么在单词和主题之间相关的中找一个和这个相关最高是，eg：
w1: [0,1.0],
w2: [0.92,0.08],
w3: [0.5,0.5]
w4: [0.8,0.2]
"""
train_data = topic_2_word_data.T
kdtree = KDTree(train_data, metric='euclidean')
result = kdtree.query(doc_2_topic_data, k=10, return_distance=False)
print("\n最终提取出来的关键词:")
print(feature_names[result])
