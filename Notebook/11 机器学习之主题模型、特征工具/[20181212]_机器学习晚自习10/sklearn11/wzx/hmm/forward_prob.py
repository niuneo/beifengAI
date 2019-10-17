# -- encoding:utf-8 --
"""
前向概率alpha的计算
Create by ibf on 2018/12/12
"""

import numpy as np
import common


def calc_alpha(pi, A, B, Q, alpha=None, fetch_index_by_obs_seq=None):
    """
    根据传入的参数计算前向概率alpha
    :param pi: 隐状态序列中初始状态的概率值
    :param A:  状态与状态之间的转移概率矩阵
    :param B:  状态与观测值之间的转移概率矩阵
    :param Q:  观测值序列
    :param alpha:  前向概率矩阵
    :return:  返回计算后的结果
    """
    # 1. 参数初始化
    n = np.shape(A)[0]
    T = np.shape(Q)[0]
    if alpha is None:
        alpha = np.zeros(shape=(T, n))
    if fetch_index_by_obs_seq is not None:
        Q = fetch_index_by_obs_seq(Q)

    # 2. 定义t=0时刻的对应的前向概率的值
    for i in range(n):
        alpha[0][i] = pi[i] * B[i][Q[0]]

    # 3. 更新t=1到t=T-1时刻对应的前向概率值
    for t in range(1, T):
        for i in range(n):
            # a. 计算累加值
            tmp_prob = 0.0
            for j in range(n):
                tmp_prob += alpha[t - 1][j] * A[j][i]
            # b. 基于累加值和B值求出当前的前向概率值
            alpha[t][i] = tmp_prob * B[i][Q[t]]

    # 4. 返回最终结果
    return alpha


if __name__ == '__main__':
    pi = np.array([0.2, 0.5, 0.3])
    A = np.array([
        [0.5, 0.4, 0.1],
        [0.2, 0.2, 0.6],
        [0.2, 0.5, 0.3]
    ])
    B = np.array([
        [0.4, 0.6],
        [0.8, 0.2],
        [0.5, 0.5]
    ])
    Q_str = "白黑白白黑"
    Q = common.convert_obs_seq_2_index(Q_str)

    # 计算alpha的值
    alpha = calc_alpha(pi, A, B, Q)
    print("计算出来的alpha值为:")
    print(alpha)

    # 计算序列Q出现的可能性
    p = 0
    for i in alpha[-1]:
        p += i
    print("序列{}出现的可能性为:{}".format(Q_str, p))
