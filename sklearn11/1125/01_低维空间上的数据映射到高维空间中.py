# -- encoding:utf-8 --
"""
@File : 01_低维空间上的数据映射到高维空间中
@Author: Octal_H
@Date : 2019/10/18
@Desc : 
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(0, 10, 81)
y = np.power(x, 2)

# plt.plot(x, y, 'ro')
# plt.show()

x = x.reshape((-1, 1))
x2 = np.power(x, 2)
y = y.reshape((1, -1))

fig = plt.figure(facecolor='w')
ax = Axes3D(fig)
ax.plot_surface(x, x2, y, rstride=1, cstride=1, cmap=plt.cm.jet)
ax.set_xlabel('x1')
ax.set_zlabel('y')
ax.set_ylabel('x2')
plt.show()

