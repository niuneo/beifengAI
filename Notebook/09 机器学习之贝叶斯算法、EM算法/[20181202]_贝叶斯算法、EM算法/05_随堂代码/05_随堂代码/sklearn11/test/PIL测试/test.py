# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/29
"""

import re

pattern = re.compile("(^[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+\.[\.a-zA-Z0-9_-]+$)")  # 预编译能增加匹配的速度   #前闭后开的区间次数
res = re.findall(pattern, 'cc_ge-006d@163.com.cn')  # \d默认都是匹配一次
print(res)
