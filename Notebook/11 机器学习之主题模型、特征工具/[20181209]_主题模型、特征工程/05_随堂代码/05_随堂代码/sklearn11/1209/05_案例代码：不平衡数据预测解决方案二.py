# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/9
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

if __name__ == '__main__':
    # 1. 产生模拟数据
    N1 = 10000
    N2 = 500
    data1, label1 = datasets.make_blobs(n_samples=N1, n_features=2, centers=[(0, 0)], cluster_std=2.0)
    data2, label2 = datasets.make_blobs(n_samples=N2, n_features=2, centers=[(-2, -2.5)], cluster_std=[(2.0, 1.0)])
    label2[label2 == 0] = 1

    label1 = label1.reshape((-1, 1))
    label2 = label2.reshape((-1, 1))
    data1_ = np.concatenate((data1, label1), axis=1)
    # 下采样：随机抽取data1_中的N2条数据, 减少多数类别样本数目
    data1_ = data1_[np.random.permutation(N1)[:N2]]
    data2_ = np.concatenate((data2, label2), axis=1)
    data = np.concatenate((data1_, data2_), axis=0)
    print(data.shape)

    df = pd.DataFrame(data=data, columns=['A', 'B', 'label'])

    X = df.drop('label', axis=1)
    Y = df['label']
    algo = LogisticRegression()
    algo.fit(X, Y)

    w1, w2 = algo.coef_[0]
    b = algo.intercept_

    print("在训练数据上的评估报告:\n{}".format(metrics.classification_report(Y, algo.predict(X))))

    plt.plot(data1[:, 0], data1[:, 1], 'yo', markersize=3)
    plt.plot(data2[:, 0], data2[:, 1], 'bo', markersize=3)
    plt.plot([-8, 8], [8 * w1 / w2 - b / w2, -8 * w1 / w2 - b / w2], 'r-')
    plt.show()
