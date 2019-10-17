# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/15
"""

import numpy as np

a = [[1, 2], [3, 4]]
b = [[5], [6]]
print(np.concatenate((a, b), axis=1))
