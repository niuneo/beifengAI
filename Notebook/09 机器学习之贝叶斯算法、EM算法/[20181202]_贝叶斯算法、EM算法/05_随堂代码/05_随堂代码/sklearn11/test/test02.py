# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/10
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.figure()

x = np.arange(0, 10)


def f1(x):
    print("{}={}".format(x, type(x)))
    a = 2 ** x
    print("a={}".format(type(a)))
    return a


def f2(x):
    print("{}={}".format(x, type(x)))
    a = pow(x, 2)
    print("a={}".format(type(a)))
    return a


# b = pd.Series(list(map(f2, x))).plot(label='yyyy')
a = pd.Series(x).apply(f1).plot(label='xxx')
plt.legend(loc='lower right')
plt.show()

print(np.__version__)
print(matplotlib.__version__)