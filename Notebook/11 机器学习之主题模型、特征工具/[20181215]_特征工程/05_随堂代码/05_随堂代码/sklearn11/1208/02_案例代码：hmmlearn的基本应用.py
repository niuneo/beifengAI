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

# 定义训练数据
# 第一个序列: 0,1,0,0,1
# 第二个序列: 0,1,1,0,1
# 第三个序列: 0,1,1
# 第四个序列: 0,1,0,0,0,0
train = np.array([
    [0], [1], [0], [1], [1],
    [0], [1], [1], [0], [1],
    [0], [1], [0],
    [0], [1], [0], [1], [1], [1]
])

# 定义模型
"""
hmm.MultinomialHMM: 观测值是离散值的HMM模型，参数说明：
n_components：给定隐状态的数目
startprob_prior=1.0, 随机产生初始的初始概率π的模型参数，一般不需要改动
transmat_prior=1.0, 随机产生状态转移矩阵概率A的模型参数，一般不需要改动
algorithm="viterbi", 预测过程中采用的算法方式，可选值: viterbi和map
random_state=None, 随机数种子
n_iter=10, 无监督的模型训练最大允许迭代次数
tol=1e-2, 模型训练的收敛阈值
verbose=False: 是否打印日志
"""
model = hmm.MultinomialHMM(n_components=n, n_iter=10, tol=0.01, random_state=28, verbose=False)

# 模型训练
# lengths： 给定训练数据train中，各个子序列的长度是多少，如果给定为None，那么表示train是一个子序列
model.fit(train, lengths=[5, 5, 3, 6])

# 输出最终模型参数
print("最终的参数π:")
print(model.startprob_)
print("最终的参数A:")
print(model.transmat_)
print("最终的参数B:")
print(model.emissionprob_)

# 做一个viterbi算法的预测
test = np.array([
    [0, 1, 0, 0, 1]  # 白，黑，白，白，黑
]).T
print("需要预测的观测序列为:\n{}".format(test))
print("预测值为:\n{}".format(model.predict(test)))
print("概率值（viterbi算法计算出来的结果做了一个归一化操作）:\n{}".format(model.predict_proba(test)))
# 这里的盒子id只是人为的给定一个盒子的含义而已
logprod, box_index = model.decode(test, algorithm='viterbi')
print("预测的盒子序号:{}".format(box_index))
print("预测的概率值(hmm底层对概率做了一个log转换，目的是为了防止概率为0的情况出现):\n{}".format(logprod))
print("预测的概率值:\n{}".format(np.exp(logprod)))
