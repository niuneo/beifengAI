# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/9
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

warnings.filterwarnings('ignore')

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
    data2_ = np.concatenate((data2, label2), axis=1)
    # 产生一个range(N1)的序列，然后将这个序列中的值打乱顺序
    random_idx = np.random.permutation(N1)

    # 获取总数据
    t_data = np.concatenate((data1_, data2_), axis=0)
    t_df = pd.DataFrame(data=t_data, columns=['A', 'B', 'label'])
    x_test = t_df.drop('label', axis=1)
    y_test = t_df['label']

    # 下采样：随机抽取data1_中的N2条数据，做一个模型融合的方式
    y_pred = None
    total_model = N1 // N2
    for i in range(total_model):
        print("训练第{}个模型".format(i + 1))
        start_index = int(max(0, i * N2))
        end_index = int(min(N1, (i + 1) * N2))
        new_data1_ = data1_[random_idx[start_index:end_index]]
        data = np.concatenate((new_data1_, data2_), axis=0)
        df = pd.DataFrame(data=data, columns=['A', 'B', 'label'])

        X = df.drop('label', axis=1)
        Y = df['label']
        algo = LogisticRegression()
        algo.fit(X, Y)

        y_pred_ = algo.predict(x_test)
        if y_pred is None:
            y_pred = y_pred_
        else:
            y_pred += y_pred_

    # 如果有超过一半的模型认为是类别1，那么就设置为类别1，否则设置为类别0
    y_pred[y_pred < total_model // 2] = 0
    y_pred[y_pred >= total_model // 2] = 1
    print("在训练数据上的评估报告:\n{}".format(metrics.classification_report(y_test, y_pred)))
