# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/12
"""

import numpy as np
import pandas as pd

df = pd.DataFrame(data=[['1', 10, 1], ['2', 20, 2], ['1', 20, 2], ['2', 10, 2], ['1', 20, 1], ['2', 10, 2]],
                  columns=['name', 'age', 'count'])
a = df.groupby(['name', 'age'])['count'].sum().to_frame()
print(a)
print(a.reset_index())
# pd.DataFrame().reset_index()
