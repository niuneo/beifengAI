# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/9
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 给定随机数种子
np.random.seed(28)


def upper_sample_data(df, sample_number, label_name):
    """
    从df中抽取出sample_number条数据，df中的标签列为label_name
    :param df:
    :param sample_number:
    :param label_name:
    :return:
    """
    # 1. 获取DataFrame的数量
    df_column_size = df.columns.size
    df_row_size = len(df)

    # 2. 做抽样操作
    sample_df = pd.DataFrame(columns=df.columns)
    for i in range(sample_number):
        # a. 随机选择一个样本下标
        index = np.random.randint(0, df_row_size, 1)[0]
        # b. 获取下标对应的样本, 以及对应的标签值
        item = df.iloc[index]
        label_value = item[label_name]
        # c. 对数据做一个偏移
        item = item + [np.random.normal(loc=0, scale=0.1) for j in range(df_column_size)]
        # d. 对label做一个恢复
        item[label_name] = label_value
        # e. 将数据添加到DataFrame中
        sample_df.loc[i] = item
    return sample_df


if __name__ == '__main__':
    # 1. 产生模拟数据
    N1 = 10000
    N2 = 100
    features = 2
    feature_names = ['A', 'B', 'Label']
    category1 = np.random.randint(low=0, high=10, size=[N1, features]).astype(np.float)
    label1 = np.array([1] * N1).reshape((-1, 1))
    category2 = np.random.randint(low=8, high=18, size=[N2, features]).astype(np.float)
    label2 = np.array([0] * N2).reshape((-1, 1))

    data1 = np.concatenate((category1, label1), axis=1)
    data2 = np.concatenate((category2, label2), axis=1)
    data = np.concatenate((data1, data2), axis=0)
    df = pd.DataFrame(data=data, columns=feature_names)
    print(df.head(5))

    # 查看一下各个类别的数量
    print("=" * 100)
    print(df.Label.value_counts())

    # 3. 获取小众样本的数据，然后产生更多的数据出来
    small_category = df[df.Label == 0.0]
    sample_number = 1000
    sample_category_data = upper_sample_data(small_category, sample_number, label_name='Label')
    print("原始的小众样本数据:")
    print(small_category.head(5))
    print("抽样出来的小众样本数据:")
    print(sample_category_data.head(5))

    # 4. 合并数据
    df = pd.concat([df, sample_category_data], ignore_index=True)
    print("=" * 100)
    print(df.Label.value_counts())

    plt.plot(data1[:, 0], data1[:, 1], 'yo', markersize=3)
    plt.plot(data2[:, 0], data2[:, 1], 'bo', markersize=3)
    plt.plot(sample_category_data.iloc[:, 0], sample_category_data.iloc[:, 1], 'ro', markersize=3)
    plt.show()
