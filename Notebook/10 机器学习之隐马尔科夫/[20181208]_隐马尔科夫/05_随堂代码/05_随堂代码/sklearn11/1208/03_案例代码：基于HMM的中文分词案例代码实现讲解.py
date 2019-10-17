# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/8
"""

import numpy as np
import hmmlearn.hmm as hmm
import warnings

warnings.filterwarnings('ignore')


def normalize(a):
    a = np.asarray(a)
    a = a / np.sum(a)
    return a


def fit(train_file_path, encoding='utf-8'):
    """
    基于有监督的HMM模型训练算法，计算π、A、B的值，并返回
    :param train_file_path:
    :param encoding:
    :return:
    """
    # 1. 读取数据
    with open(train_file_path, mode='r', encoding=encoding) as reader:
        sentence = reader.read()[1:]

    # 2. 初始化相关的概率值
    # 隐状态4个，观测值65536个
    # B:0, M:1, E: 2, S:3
    pi = np.zeros(4)
    A = np.zeros((4, 4))
    B = np.zeros((4, 65536))

    # 3. 模型训练
    # 初始化隐状态为2
    last_state = 2
    # a. 基于空格划分数据
    tokens = sentence.split(" ")
    # b. 迭代处理所有的文本数据，统计训练数据中各个情况的样本数目是多少
    for token in tokens:
        # 1. 去掉单词的前后空格
        token = token.strip()
        # 2. 获取单词的长度
        length = len(token)
        # 3. 过滤异常的单词，如果单词长度小于1，那么直接过滤
        if length < 1:
            continue

        # 4. 单独处理长度为1的特殊单词
        if length == 1:
            pi[3] += 1
            A[last_state][3] += 1
            # ord的函数作用是讲字符状态为ACSII码
            B[3][ord(token[0])] += 1
            last_state = 3
        else:
            # 如果长度大于1，那么表示这个词语至少有两个单词，那么初始概率中为0的增加1
            pi[0] += 1

            # 更新状态转移概率矩阵
            A[last_state][0] += 1
            last_state = 2
            if length == 2:
                A[0][2] += 1
            else:
                A[0][1] += 1
                A[1][2] += 1
                A[1][1] += (length - 3)

            # 更新隐状态到观测值之间的转移概率矩阵
            B[0][ord(token[0])] += 1
            B[2][ord(token[-1])] += 1
            for i in range(1, length - 1):
                B[1][ord(token[i])] += 1

    # 4. 计算概率值
    pi = normalize(pi)
    for i in range(4):
        A[i] = normalize(A[i])
        B[i] = normalize(B[i])

    return pi, A, B


def dump(pi, A, B):
    n, m = np.shape(B)
    # 1. pi的输出
    with open('pi.txt', 'w') as w1:
        w1.write(str(n))
        w1.write('\n')
        w1.writelines(' '.join(map(str, pi)))
    # 2. A的输出
    with open('A.txt', 'w') as w2:
        w2.write(str(n))
        w2.write('\n')
        for a in A:
            w2.writelines(' '.join(map(str, a)))
            w2.writelines('\n')
    # 3. B的输出
    with open('B.txt', 'w') as w3:
        w3.write(str(n))
        w3.write('\n')
        w3.write(str(m))
        w3.write('\n')
        for b in B:
            w3.writelines(' '.join(map(str, b)))
            w3.writelines('\n')


def load():
    # 1. 加载pi的数据
    with open('pi.txt', 'r', encoding='utf-8') as reader:
        reader.readline()
        line = reader.readline()
        pi = list(map(float, line.strip().split(" ")))
    # 2. 加载A的数据
    with open('A.txt', 'r', encoding='utf-8') as reader:
        n = int(reader.readline())
        A = np.zeros((n, n))
        i = 0
        for line in reader:
            j = 0
            for v in map(float, line.strip().split(" ")):
                A[i][j] = v
                j += 1
            i += 1
    # 3. 加载B的数据
    with open('B.txt', 'r', encoding='utf-8') as reader:
        n = int(reader.readline())
        m = int(reader.readline())
        B = np.zeros((n, m))
        i = 0
        for line in reader:
            j = 0
            for v in map(float, line.strip().split(" ")):
                B[i][j] = v
                j += 1
            i += 1

    return pi, A, B


def cut(model, sentence):
    # 1. 将文本数据做一个转换的操作
    x_test = []
    for token in sentence:
        x_test.append(ord(token))
    x_test = np.asarray(x_test).reshape((-1, 1))
    # 2. 基于模型对输出数据做一个预测
    decode = model.predict(x_test)
    # 3. 基于隐状态的信息对数据做一个分词的操作
    print("隐状态为:\n{}".format(decode))
    T = len(decode)
    t = 0
    while t < T:
        state = decode[t]
        if state == 0 or state == 1:
            # 表示t时刻对应的单词是一个词语的开始或者中间位置，那么后面还有字属于同一个词语
            j = t + 1
            while j < T:
                if decode[j] == 2:
                    break
                j += 1
            yield sentence[t:j + 1]
            t = j
        elif state == 3 or state == 2:
            yield sentence[t:t + 1]
        else:
            yield "ERROR"
        t += 1


if __name__ == '__main__':
    # TODO: 大家自己想一下，能不能简化一下模型参数的保存工作，提示：使用numpy相关的API进行存储+加载
    flag = False
    if flag:
        pi, A, B = fit('./pku_training.utf8')
        dump(pi, A, B)
    else:
        # 1. 加载模型参数
        pi, A, B = load()
        # 2. 基于模型参数构建模型对象
        model = hmm.MultinomialHMM(n_components=4)
        # 3. 直接给定模型参数
        model.startprob_ = pi
        model.transmat_ = A
        model.emissionprob_ = B
        # 4. 做一个分词的操作
        cut_result = cut(model, "因为在我们对文本数据进行理解的时候，机器不可能完整的对文本进行处理，只能把每个单词作为特征属性来进行处理，所以在所有的文本处理中，第一步就是分词操作，中共中央总书记")
        print("分词结果为:\n{}".format(list(cut_result)))
