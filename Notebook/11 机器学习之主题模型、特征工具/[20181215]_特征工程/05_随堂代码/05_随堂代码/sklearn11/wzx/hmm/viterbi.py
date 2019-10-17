# -- encoding:utf-8 --
"""
Viterbi算法实现
Create by ibf on 2018/12/12
"""

import numpy as np
import common


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
    T = np.shape(Q)[0]
    pre_index = np.zeros((T, n), dtype=np.int32)
    if delta is None:
        delta = np.zeros(shape=(T, n))
    if fetch_index_by_obs_seq is not None:
        Q = fetch_index_by_obs_seq(Q)

    # 2. 定义t=0时刻的对应的deta值
    for i in range(n):
        delta[0][i] = pi[i] * B[i][Q[0]]

    # 3. 更新t=1到t=T-1时刻对应的delta值
    for t in range(1, T):
        for i in range(n):
            # a. 获取最大值
            max_delta = -1.0
            for j in range(n):
                tmp = delta[t - 1][j] * A[j][i]
                if tmp > max_delta:
                    max_delta = tmp
                    pre_index[t][i] = j

            # b. 基于最大概率值和B值求出当前的delta值
            delta[t][i] = max_delta * B[i][Q[t]]

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
    state = np.asarray(['盒子1', '盒子2', '盒子3'])
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

    # 计算delta的值
    delta, decode = viterbi(pi, A, B, Q)
    print("计算出来的delta值为:")
    print(delta)
    print("序列[{}]最有可能的状态序列为:".format(Q_str))
    print(decode)
    print(state[decode])
