# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/6
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.array([
    [10, 1],
    [15, 1],
    [20, 1],
    [30, 1],
    [50, 1],
    [60, 1],
    [60, 1],
    [70, 1]
]).astype(np.float32)
y = np.array([0.8, 1.0, 1.8, 2.0, 3.2, 3.0, 3.1, 3.5]).reshape((-1, 1))

x = np.mat(x)
y = np.mat(y).reshape(-1, 1)

theta = (x.T * x).I * x.T * y
print(theta)

a = float(theta[0][0])
b = float(theta[1][0])
# 画图可视化
t = np.array(y).reshape(-1)
z = np.array(x[:, 0]).reshape(-1)
plt.plot(z, t, 'bo')
plt.plot([10, 70], [a * 10 + b, a * 70 + b], 'r--', linewidth=2)
plt.plot([55, 55], [0, 4.0], 'g--', linewidth=2)
plt.plot([10,70],[a * 55 + b, a * 55 + b], 'g--', linewidth=2)
plt.title("a:%.4f; b:%.4f" % (a, b))
plt.grid()
plt.show()
