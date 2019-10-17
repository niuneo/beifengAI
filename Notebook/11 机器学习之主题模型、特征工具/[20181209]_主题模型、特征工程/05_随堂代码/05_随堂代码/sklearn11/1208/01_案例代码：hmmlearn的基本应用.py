# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/8
"""

import numpy as np
import hmmlearn.hmm as hmm

# 定义变量
states = ['盒子1', '盒子2', '盒子3']
obs = ['白球', '黑球']
n = 3
m = 2

# 定义模型参数
pi = np.array([0.2, 0.5, 0.3])
a = np.array([
    [0.5, 0.4, 0.1],
    [0.2, 0.2, 0.6],
    [0.2, 0.5, 0.3]
])
b = np.array([
    [0.4, 0.6],
    [0.8, 0.2],
    [0.5, 0.5]
])

# 定义模型
"""
hmm.MultinomialHMM: 观测值是离散值的HMM模型，参数说明：
n_components：给定隐状态的数目
"""
model = hmm.MultinomialHMM(n_components=n)

# 直接给定模型参数
model.startprob_ = pi
model.transmat_ = a
model.emissionprob_ = b

# 做一个viterbi算法的预测
test = np.array([
    [0, 1, 0, 0, 1]  # 白，黑，白，白，黑
]).T
print("需要预测的观测序列为:\n{}".format(test))
print("预测值为:\n{}".format(model.predict(test)))
print("预测值为（盒子id，从1开始）:\n{}".format(model.predict(test) + 1))
print("概率值（viterbi算法计算出来的结果做了一个归一化操作）:\n{}".format(model.predict_proba(test)))
logprod, box_index = model.decode(test, algorithm='viterbi')
print("预测的盒子序号:{}".format(box_index))
print("预测的概率值(hmm底层对概率做了一个log转换，目的是为了防止概率为0的情况出现):\n{}".format(logprod))
print("预测的概率值:\n{}".format(np.exp(logprod)))
