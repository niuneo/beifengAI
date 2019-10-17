# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/6
"""

import pandas as pd
import numpy as np

# a = pd.read_csv('province.csv')
# b = pd.read_csv('user.csv')
# b['phone'] = pd.Series(map(lambda t: t // 10000, b['phone']))
#
# # join数据
# c = pd.merge(a, b, on='phone')
# c['id'] = c['id'].astype(np.str)
# # 截取出生年份，计算年龄
# c['c1'] = pd.Series(data=map(lambda t: 2018 - int(t[6:10]), c['id']))
# # 获取用户年龄
# r1 = c[['name', 'c1']]
# print(r1.head())
# # 获取用户省份
# r2 = c[['name', 'province']]
# print(r2.head())

"""
用户年龄获取：
select name, substr(cast(id as char), 7, 4) as age from user
用户省份获取:
select name, province from user join province on substr(cast(user.phone as char), 0, 7) = province.phone

"""

a = np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0])
b = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4])
a = np.array([0.1, 0.1, 0.1, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.3, 0.4])
a = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4])
a = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
print(np.sum(a))
print(np.sum(b))
c = np.power(a - b, 2)
c = np.sum(c)
d = np.sqrt(c)
s = 1 / (d + 1)
print(s)
