# -- encoding:utf-8 --
"""
计算γ概率值
Create by ibf on 2018/12/12
"""

import numpy as np
import common, forward_prob, backward_prob


def calc_gamma(alpha, beta, gamma=None):
    """
    根据传入的参数计算概率矩阵gamma
    :param alpha:  前向概率矩阵
    :param beta:  后向概率矩阵
    :param gamma:  gamma概率矩阵
    :return:  返回计算后的结果
    """
    # 1. 参数初始化
    T, n = np.shape(alpha)
    if gamma is None:
        gamma = np.zeros(shape=(T, n))

    # 2. 更新gamma矩阵的值
    tmp_prob = np.zeros(n)
    for t in range(T):
        # a. 分别计算当前时刻t，状态为j的前向概率和后向概率的乘积
        for j in range(n):
            tmp_prob[j] = alpha[t][j] * beta[t][j]
        tmp_prob_sum = np.sum(tmp_prob)

        # b. 更新gamma概率值
        for i in range(n):
            gamma[t][i] = tmp_prob[i] / tmp_prob_sum

    # 3. 返回最终结果
    return gamma


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

    # 计算gamma的值
    alpha = forward_prob.calc_alpha(pi, A, B, Q)
    beta = backward_prob.calc_beta(pi, A, B, Q)
    gamma = calc_gamma(alpha, beta)
    print("计算出来的gamma值为:")
    print(gamma)
