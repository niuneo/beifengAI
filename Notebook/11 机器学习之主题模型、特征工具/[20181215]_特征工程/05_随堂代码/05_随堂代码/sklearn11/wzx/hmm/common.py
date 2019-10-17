# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/12
"""


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
