# -- encoding:utf-8 --
"""
加入log对数化后的HMM实现
Create by ibf on 2018/12/12
"""

import numpy as np
import common


def calc_alpha(pi, A, B, Q, alpha=None, fetch_index_by_obs_seq=None):
    """
    根据传入的参数计算前向概率alpha
    :param pi: 隐状态序列中初始状态的概率值，是经过log对数化后的值
    :param A:  状态与状态之间的转移概率矩阵，是经过log对数化后的值
    :param B:  状态与观测值之间的转移概率矩阵，是经过log对数化后的值
    :param Q:  观测值序列
    :param alpha:  前向概率矩阵，是经过log对数化后的值
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
        alpha[0][i] = pi[i] + B[i][Q[0]]

    # 3. 更新t=1到t=T-1时刻对应的前向概率值
    for t in range(1, T):
        for i in range(n):
            # a. 计算累加值
            tmp_prob = np.zeros(n)
            for j in range(n):
                tmp_prob[j] = alpha[t - 1][j] + A[j][i]
            # b. 基于累加值和B值求出当前的前向概率值
            alpha[t][i] = common.log_sum_exp(tmp_prob) + B[i][Q[t]]

    # 4. 返回最终结果
    return alpha


def calc_beta(pi, A, B, Q, beta=None, fetch_index_by_obs_seq=None):
    """
    根据传入的参数计算后向概率beta
    :param pi: 隐状态序列中初始状态的概率值
    :param A:  状态与状态之间的转移概率矩阵
    :param B:  状态与观测值之间的转移概率矩阵
    :param Q:  观测值序列
    :param beta:  后向概率矩阵
    :return:  返回计算后的结果
    """
    # 1. 参数初始化
    n = np.shape(A)[0]
    T = np.shape(Q)[0]
    if beta is None:
        beta = np.zeros(shape=(T, n))
    if fetch_index_by_obs_seq is not None:
        Q = fetch_index_by_obs_seq(Q)

    # 2. 定义t=T-1时刻的对应的后向概率的值
    for i in range(n):
        beta[T - 1][i] = 0

    # 3. 更新t=T-2到t=0时刻对应的后向概率值
    for t in range(T - 2, -1, -1):
        for i in range(n):
            # a. 计算累加值
            tmp_prob = np.zeros(n)
            for j in range(n):
                tmp_prob[j] = A[i][j] + beta[t + 1][j] + B[j][Q[t + 1]]

            # b. 基于累加值求出当前的后向概率值
            beta[t][i] = common.log_sum_exp(tmp_prob)

    # 4. 返回最终结果
    return beta


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
            tmp_prob[j] = alpha[t][j] + beta[t][j]
        tmp_prob_sum = common.log_sum_exp(tmp_prob)

        # b. 更新gamma概率值
        for i in range(n):
            gamma[t][i] = tmp_prob[i] - tmp_prob_sum

    # 3. 返回最终结果
    return gamma


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
                tmp_prob[i][j] = alpha[t][i] + A[i][j] + B[j][Q[t + 1]] + beta[t + 1][j]
        tmp_prob_sum = common.log_sum_exp(tmp_prob.flat)

        # b. 更新gamma概率值
        for i in range(n):
            for j in range(n):
                ksi[t][i][j] = tmp_prob[i][j] - tmp_prob_sum

    # 3. 返回最终结果
    return ksi


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
        calc_alpha(pi, A, B, Q, alpha)
        calc_beta(pi, A, B, Q, beta)
        calc_gamma(alpha, beta, gamma)
        calc_ksi(alpha, beta, A, B, Q, ksi)

        # b. 在当前的最优的概率情况下修改模型参数
        # b1. 更新pi的值
        for i in range(n):
            pi[i] = gamma[0][i]

        # b2. 更新A的值
        for i in range(n):
            for j in range(n):
                # 1. 分别求解分子和分母的值
                numerator = np.zeros(T - 1)
                denominator = np.zeros(T - 1)
                for t in range(T - 1):
                    numerator[t] = ksi[t][i][j]
                    denominator[t] = gamma[t][i]

                # 2. 基于分子和分母计算概率值
                A[i][j] = common.log_sum_exp(numerator) - common.log_sum_exp(denominator)

        # b3. 更新B的值
        for i in range(n):
            for j in range(m):
                # 1. 分别求解分子和分母的值
                numerator = np.zeros(T)
                denominator = np.zeros(T)
                number = 0
                for t in range(T):
                    if j == Q[t]:
                        numerator[number] = gamma[t][i]
                        number += 1
                    denominator[t] = gamma[t][i]

                # 2. 基于分子和分母计算概率值
                if number == 0:
                    B[i][j] = float(-2 ** 31)
                else:
                    B[i][j] = common.log_sum_exp(numerator[:number]) - common.log_sum_exp(denominator)

    return pi, A, B


def viterbi(pi, A, B, Q, delta=None, fetch_index_by_obs_seq=None):
    """
    根据传入的参数计算前向概率alpha
    :param pi: 隐状态序列中初始状态的概率值
    :param A:  状态与状态之间的转移概率矩阵
    :param B:  状态与观测值之间的转移概率矩阵
    :param Q:  观测值序列
    :param delta:  viterbi的概率矩阵
    :return:  返回计算后的结果
    """
    # 1. 参数初始化
    n = np.shape(A)[0]
    if fetch_index_by_obs_seq is not None:
        Q = fetch_index_by_obs_seq(Q)
    T = np.shape(Q)[0]
    pre_index = np.zeros((T, n), dtype=np.int32)
    if delta is None:
        delta = np.zeros(shape=(T, n))

    # 2. 定义t=0时刻的对应的deta值
    for i in range(n):
        delta[0][i] = pi[i] + B[i][Q[0]]

    # 3. 更新t=1到t=T-1时刻对应的delta值
    for t in range(1, T):
        for i in range(n):
            # a. 获取最大值
            max_delta = delta[t - 1][0] + A[0][i]
            for j in range(1, n):
                tmp = delta[t - 1][j] + A[j][i]
                if tmp > max_delta:
                    max_delta = tmp
                    pre_index[t][i] = j

            # b. 基于最大概率值和B值求出当前的delta值
            delta[t][i] = max_delta + B[i][Q[t]]

    # 4. 做一个解码操作，获取最有可能的状态链
    decode = [-1 for i in range(T)]
    # 首先要找到最后一个时刻对应的最优状态，也就是概率最大的那个状态
    max_delta_index = 0
    for i in range(1, n):
        if delta[T - 1][i] > delta[T - 1][max_delta_index]:
            max_delta_index = i
    decode[T - 1] = max_delta_index
    # 然后基于最后一个时刻的最优状态，找之前最优可能的状态，也就是最大的
    for t in range(T - 2, -1, -1):
        max_delta_index = pre_index[t + 1][max_delta_index]
        decode[t] = max_delta_index

    # 4. 返回最终结果
    return delta, decode


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
    pi = np.log(pi)
    A = np.log(A)
    B = np.log(B)
    Q_str = "白黑白白黑"
    Q = common.convert_obs_seq_2_index(Q_str)

    # 计算alpha的值
    alpha = calc_alpha(pi, A, B, Q)
    print("计算出来的alpha值为:")
    print(alpha)
    print(np.exp(alpha))

    # 计算序列Q出现的可能性
    p = 0
    for i in alpha[-1]:
        p += np.exp(i)
    print("序列{}出现的可能性为:{}".format(Q_str, p))
