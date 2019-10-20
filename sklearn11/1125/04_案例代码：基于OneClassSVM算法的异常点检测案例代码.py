# -- encoding:utf-8 --
"""
@File : 04_案例代码：基于OneClassSVM算法的异常点检测案例代码
@Author: Octal_H
@Date : 2019/10/20
@Desc : 
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

# 设置一下，防止乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(25)

# 产生模拟数据
x = 0.3 * np.random.rand(100, 2)
x_train = np.vstack([x + 2, x - 2])
x = 0.3 * np.random.rand(20, 2)
x_test = np.vstack([x + 2, x - 2])
x_outliers = np.random.uniform(low=-2.5, high=2.5, size=(20, 2))

# 模型构建
algo = OneClassSVM(kernel='rbf', nu=0.01)
algo.fit(x_train)

# 模型预测（1表示正常样本，-1表示异常样本）
y_pred_train = algo.predict(x_train)
print(y_pred_train)
y_pred_test = algo.predict(x_test)
print(y_pred_test)
y_pred_outliers = algo.predict(x_outliers)
print(y_pred_outliers)

# 看一下decision_function
print(algo.decision_function(x_test).ravel())
print(algo.decision_function(x_train).ravel())

# 画图可视化
x1_min = -3
x1_max = 3
x2_min = -3
x2_max = 3

# 等距离的从最小值到最大值之间产生50点
t1 = np.linspace(x1_min, x1_max, 50)
t2 = np.linspace(x2_min, x2_max, 50)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
x_show = np.dstack((x1.flat, x2.flat))[0]  # 测试点
z = algo.decision_function(x_show)
z = z.reshape(x1.shape)

plt.contourf(x1, x2, z, cmap=plt.cm.Blues_r)
plt.scatter(x_train[:, 0], x_train[:, 1], c='b')
plt.scatter(x_test[:, 0], x_test[:, 1], c='g')
plt.scatter(x_outliers[:, 0], x_outliers[:, 1], c='r')
plt.show()
