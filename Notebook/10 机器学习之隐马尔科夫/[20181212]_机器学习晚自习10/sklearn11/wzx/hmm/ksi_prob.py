# -- encoding:utf-8 --
"""
计算ksi的概率值
Create by ibf on 2018/12/12
"""

import numpy as np
import common, forward_prob, backward_prob


def calc_ksi(alpha, beta, A, B, Q, ksi=None, fetch_index_by_obs_seq=None):
    """
    根据传入的参数计算ksi矩阵'
    :param alpha:  前向概率矩阵
    :param beta: 后向概率矩阵
    :param pi: 隐状态序列中初始状态的概率值
    :param A:  状态与状态之间的转移概率矩阵
    :param B:  状态与观测值之间的转移概率矩阵
    :param Q:  观测值序列
    :param ksi: ksi概率矩阵
    :return:  返回计算后的结果
    """
    # 1. 参数初始化
    T, n = np.shape(alpha)
    if ksi is None:
        ksi = np.zeros(shape=(T - 1, n, n))
    if fetch_index_by_obs_seq is not None:
        Q = fetch_index_by_obs_seq(Q)

    # 2. 更新ksi矩阵的值
    tmp_prob = np.zeros((n, n))
    for t in range(T - 1):
        # a. 分别计算当前时刻t，状态为j的前向概率、后向概率、A以及B概率的乘积
        for i in range(n):
            for j in range(n):
                tmp_prob[i][j] = alpha[t][i] * A[i][j] * B[j][Q[t + 1]] * beta[t + 1][j]
        tmp_prob_sum = np.sum(tmp_prob)

        # b. 更新gamma概率值
        for i in range(n):
            for j in range(n):
                ksi[t][i][j] = tmp_prob[i][j] / tmp_prob_sum

    # 3. 返回最终结果
    return ksi


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

    # 计算ksi的值
    alpha = forward_prob.calc_alpha(pi, A, B, Q)
    beta = backward_prob.calc_beta(pi, A, B, Q)
    ksi = calc_ksi(alpha, beta, A, B, Q)
    print("计算出来的ksi值为:")
    print(ksi)
