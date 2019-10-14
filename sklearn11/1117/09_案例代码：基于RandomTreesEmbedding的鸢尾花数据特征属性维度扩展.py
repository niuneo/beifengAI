# -- encoding:utf-8 --
"""
@File : 09_案例代码：基于RandomTreesEmbedding的鸢尾花数据特征属性维度扩展
@Author: Octal_H
@Date : 2019/10/14
@Desc : 
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


mpl.rcParams['font.sans-serif'] = [u'simHei']


# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = '../datas/iris.data'
names = ['A', 'B', 'C', 'D', 'cla']
df = pd.read_csv(path, header=None, names=names)


# 2. 数据清洗


# # 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = df[names[0:-1]]
Y = df[names[-1]]
# print(Y)
label_encoder = LabelEncoder()
label_encoder.fit(Y)
Y = label_encoder.transform(Y)
# 这里得到的序号其实就是classes_这个集合中对应数据的下标
# print(label_encoder.classes_)
# true_label = label_encoder.inverse_transform([0, 1, 2, 0])
# print(true_label)
# print(Y)


# 4. 数据分割
# train_size: 给定划分之后的训练数据的占比是多少，默认0.75
# random_state：给定在数据划分过程中，使用到的随机数种子，默认为None，使用当前的时间戳；给定非None的值，可以保证多次运行的结果是一致的。
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))


# 5. 特征工程的操作
# NOTE: 不做特征工程


# 6. 模型对象的构建
algo = RandomTreesEmbedding(n_estimators=10, max_depth=2, sparse_output=False)


# 7. 模型的训练
algo.fit(x_train)


# 10. 其他特殊的API
print("子模型列表:\n{}".format(algo.estimators_))


from sklearn import tree
import pydotplus

k = 0
for algo1 in algo.estimators_:
    dot_data = tree.export_graphviz(decision_tree=algo1, out_file=None,
                                    feature_names=['A', 'B', 'C', 'D'],
                                    class_names=['1', '2', '3'],
                                    filled=True, rounded=True,
                                    special_characters=True
                                    )

    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('images/trte_{}.png'.format(k))
    k += 1
    if k > 3:
        break

# 做一个维度扩展
print("*" * 100)
x_test2 = x_test.iloc[:2, :]
print(x_test2)
# apply方法返回的是叶子节点下标
print(algo.apply(x_test2))
# transform转换数据（其实就是apply方法+哑编码）
print(algo.transform(x_test2))