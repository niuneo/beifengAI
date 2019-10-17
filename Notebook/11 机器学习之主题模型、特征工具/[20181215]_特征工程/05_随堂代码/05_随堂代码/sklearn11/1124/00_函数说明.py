# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/24
"""

from sklearn.metrics.pairwise import pairwise_distances_argmin

"""
pairwise_distances_argmin: 主要用于两个集合中的元素的距离计算，并且返回距离最近的样本的下标
pairwise_distances_argmin(A,B): 返回集合A中每个样本对应的最近样本(B中)所对应的下标
"""

print(pairwise_distances_argmin([[5], [3], [2]], [[1.5], [3.5], [4.5]]))
print(pairwise_distances_argmin([[5], [3], [2]], [[1.5], [4.0], [6.5]]))
print(pairwise_distances_argmin([[-3.95500247, 9.33901235],
                                 [7.26918456, -9.77745055],
                                 [0.31837485, -3.71616684],
                                 [0.13539809, 5.09169931]],
                                [[-0.20485586, 5.36198502],
                                 [7.41375408, - 9.56115236],
                                 [0.43372079, - 3.50088682],
                                 [-4.02864373, 9.99786527]]))
