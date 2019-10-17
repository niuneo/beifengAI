# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/12
"""

import numpy as np


def log_sum_exp(arr):
    """
    可以参考numpy中的log sum exp的API
    scipy.misc.logsumexp
    :param arr:
    :return:
    """
    arr = np.asarray(arr)
    # a. 获取列表arr中的最大值
    max_v = max(arr)
    # b. 计算列表中所有值和最大值的差值的指数函数的值
    tmp = np.sum(np.exp(arr - max_v))
    # c. 对和做一个对数转换后加上最大值返回
    return max_v + np.log(tmp)


def convert_obs_seq_2_index(Q, index=None):
    if index is not None:
        cht = Q[index]
        if cht == '黑':
            return 1
        else:
            return 0
    else:
        result = []
        for cht in Q:
            if cht == '黑':
                result.append(1)
            else:
                result.append(0)
        return result
