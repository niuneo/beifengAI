# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/13
"""

import numpy as np
import hmm_learn
import common

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
    # 对A\B\pi进行log转换、
    pi = np.log(pi)
    A = np.log(A)
    B = np.log(B)
    n = np.shape(A)[0]

    Q_str = "白黑白白黑"
    Q = common.convert_obs_seq_2_index(Q_str)
    T = np.shape(Q)[0]

    print("测试前向概率计算....................")
    alpha = hmm_learn.calc_alpha(pi, A, B, Q)
    print(alpha)
    print(np.exp(alpha))
    # 计算序列Q出现的可能性
    print("序列{}出现的可能性为:{}".format(Q_str, np.exp(common.log_sum_exp(alpha[T - 1]))))

    print("测试后向概率计算....................")
    beta = hmm_learn.calc_beta(pi, A, B, Q)
    print(beta)
    print(np.exp(beta))

    # 计算最终概率值：
    tmp_p = np.zeros(n)
    for i in range(n):
        tmp_p[i] = pi[i] + B[i][Q[0]] + beta[0][i]
    p = common.log_sum_exp(tmp_p)
    print(Q_str, end="->出现的概率为:")
    print(np.exp(p))

    # print("baum welch算法应用.......................")
    # hmm_learn.baum_welch(pi, A, B, Q)
    # print("最终的pi矩阵：", end='')
    # print(np.exp(pi))
    # print("最终的状态转移矩阵：")
    # print(np.exp(A))
    # print("最终的状态-观测值转移矩阵：")
    # print(np.exp(B))

    print("viterbi算法应用.....................")
    delta, state_seq = hmm_learn.viterbi(pi, A, B, Q)
    print("最终结果为:", end='')
    print(state_seq)
    state = ['盒子1', '盒子2', '盒子3']
    for i in state_seq:
        print(state[i], end='\t')