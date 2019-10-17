# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/1
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.externals import joblib

# 1. 加载数据
digits = datasets.load_digits()
# 总样本数目：1797张0~9的数字图像，每个图像的大小: 8*8, 也就是使用64个像素点来表示图像信息
print(digits.data.shape)
print(digits.images.shape)
print(digits.target.shape)
# print(digits)

# 随便看几个图片
# idx = 3
# print(digits.images[idx])
# plt.imshow(digits.images[idx], cmap=plt.cm.gray_r)
# plt.title(digits.target[idx])
# plt.show()

# 2. 数据的划分
n_samples = digits.images.shape[0]
X = digits.data
Y = digits.target
split_index = 1 * n_samples // 2
x_train, x_test = X[:split_index], X[split_index:]
y_train, y_test = Y[:split_index], Y[split_index:]
print("训练数据格式:{}".format(np.shape(x_train)))
print("测试数据格式:{}".format(np.shape(x_test)))

# 3. 模型构建
algo = SVC(kernel='rbf', C=1.0, gamma=0.001)
algo.fit(x_train, y_train)

# 4. 模型效果查看
print("训练数据上的效果:{}".format(algo.score(x_train, y_train)))
print("测试数据上的效果:{}".format(algo.score(x_test, y_test)))

# 模型持久化
joblib.dump(algo, './svm_digits.pkl')
