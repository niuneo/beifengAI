# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/12
"""

import numpy as np
import common, forward_prob, backward_prob, ksi_prob, gamma_prob


def baum_welch(pi, A, B, Q, max_iter=3, fetch_index_by_obs_seq=None):
    # 1. 参数初始化
    if fetch_index_by_obs_seq is not None:
        Q = fetch_index_by_obs_seq(Q)
    T = np.shape(Q)[0]
    n = np.shape(A)[0]
    m = np.shape(B)[1]
    alpha = np.zeros((T, n))
    beta = np.zeros((T, n))
    gamma = np.zeros((T, n))
    ksi = np.zeros((T - 1, n, n))

    # 2. 迭代更新
    for time in range(max_iter):
        # a. 计算在当前模型参数下的序列Q的相关概率的值
        forward_prob.calc_alpha(pi, A, B, Q, alpha)
        backward_prob.calc_beta(pi, A, B, Q, beta)
        gamma_prob.calc_gamma(alpha, beta, gamma)
        ksi_prob.calc_ksi(alpha, beta, A, B, Q, ksi)

        # b. 在当前的最优的概率情况下修改模型参数
        # b1. 更新pi的值
        for i in range(n):
            pi[i] = gamma[0][i]

        # b2. 更新A的值
        for i in range(n):
            for j in range(n):
                # 1. 分别求解分子和分母的值
                numerator = 0.0
                denominator = 0.0
                for t in range(T - 1):
                    numerator += ksi[t][i][j]
                    denominator += gamma[t][i]

                # 2. 基于分子和分母计算概率值
                if denominator == 0.0:
                    A[i][j] = 0.0
                else:
                    A[i][j] = numerator / denominator

        # b3. 更新B的值
        for i in range(n):
            for j in range(m):
                # 1. 分别求解分子和分母的值
                numerator = 0.0
                denominator = 0.0
                for t in range(T - 1):
                    if j == Q[t]:
                        numerator += ksi[t][i][j]
                    denominator += gamma[t][i]

                # 2. 基于分子和分母计算概率值
                if denominator == 0.0:
                    B[i][j] = 0.0
                else:
                    B[i][j] = numerator / denominator

    return pi, A, B


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

    Q = common.convert_obs_seq_2_index(np.array(['白', '黑', '白', '白', '黑']))
    Q1 = common.convert_obs_seq_2_index(np.array(['白', '黑', '黑', '白', '黑']))
    print("初始的随机状态矩阵:")
    print("初始状态概率向量：")
    print(pi)
    print("\n初始的状态之间的转移概率矩阵：")
    print(A)
    print("\n初始的状态和观测值之间的转移概率矩阵：")
    print(B)

    # 计算结果
    baum_welch(pi, A, B, Q)
    baum_welch(pi, A, B, Q1)

    # 输出最终结果
    print("\n\n最终计算出来的状态矩阵:")
    print("状态概率向量：")
    print(pi)
    print("\n状态之间的转移概率矩阵：")
    print(A)
    print("\n状态和观测值之间的转移概率矩阵：")
    print(B)
