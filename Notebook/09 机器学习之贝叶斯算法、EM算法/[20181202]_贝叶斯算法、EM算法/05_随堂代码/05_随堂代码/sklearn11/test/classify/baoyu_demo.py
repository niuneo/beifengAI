# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/8
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

np.random.seed(28)


class Smote:
    def __init__(self, samples, N=10, k=5):
        """
        samples是DataFrame对象，是原始的小众数据样本
        N: 每个样本需要扩展的几个其它样本
        k: 计算近邻的时候，邻居的数量
        """
        self.n_samples, self.n_attrs = samples.shape
        self.N = N
        self.k = k
        self.samples = samples

    def over_sampling(self):
        # 每个类别至少合成的样本数量
        self.synthetic = pd.DataFrame(columns=self.samples.columns)
        self.new_index = 0

        # 模型训练
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)

        # 对所有样本进行一个遍历，每个样本都需要产生N个随机的新样本
        for i in range(len(self.samples)):
            # 针对于当前样本，获取对应的k个邻居的索引下标
            nnarray = neighbors.kneighbors([self.samples.iloc[i]], return_distance=False)
            # 存储具体的数据
            self.__populate(self.N, i, nnarray.flatten())

        return self.synthetic

    # 从k个邻居中随机选取N次，生产N个合成的样本
    def __populate(self, N, i, nnarray):
        for j in range(N):
            # 获取随机一个索引值（为了获取相似样本的坐标值）
            nn = np.random.randint(0, self.k)
            # 随机一个每个维度上的随机数量, 最后一个是标签，不进行偏置操作
            gap = np.random.rand(data.shape[1])
            gap[-1] = 0.0
            # 计算新样本的位置信息
            new_sample = zip(self.samples.iloc[i], self.samples.iloc[nnarray[nn]],
                             gap, self.samples.dtypes)

            def f(t):
                if t[-1] == 'int64':
                    idx = np.random.randint(0, 2)
                    result = t[idx]
                else:
                    result = t[0] + (t[1] - t[0]) * t[2]
                return result

            new_sample = np.asarray(list(map(lambda t: f(t), new_sample)))

            # 进行添加操作操作
            self.synthetic.loc[self.new_index] = new_sample
            self.new_index += 1


# 1. 加载数据
DATA_SET_LOCATION = "./baoyu.csv"
data = pd.read_csv(DATA_SET_LOCATION,
                   names=['Sex', 'Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings'])
# data.info()

"""
M    1528
I    1342
F    1307
"""
# print(data.Sex.value_counts())

# a = data.Rings.value_counts()
# print("Mean：{}".format(np.mean(a)))
# a = a.sort_index(axis=0)
# a.plot(kind='bar')
# plt.show()

# 删除类别比较少的数据
data = data.loc[~ data.Rings.isin([1, 2, 24, 25, 26, 27, 29])]
print(data.shape)

# 对字符串进行编码
le = LabelEncoder()
data.Sex = le.fit_transform(data.Sex)

# 对数据做一个划分
train, test = train_test_split(data, test_size=0.25, random_state=0)
train_old, test_old = train, test

# 针对特定的类别做一个增加数据的操作
values = [(3, 9, 3), (4, 2, 5), (16, 2, 5), (17, 2, 5), (18, 2, 5),
          (19, 3, 5), (20, 4, 5), (21, 9, 3), (22, 20, 3), (23, 20, 3)]
datas = []
for rings, n, k in values:
    #  直接对原始数据做一个扩展
    tmp = data[data.Rings == rings]
    tmp = Smote(samples=tmp, N=n, k=k).over_sampling()
    datas.append(tmp)

# 合并数据
data0 = pd.concat(datas, ignore_index=True)
train0, test0 = train_test_split(data0, test_size=0.25, random_state=0)
train = pd.concat([train, train0], ignore_index=True)
test = pd.concat([test, test0], ignore_index=True)

# a = data0.Rings.value_counts()
# a = a.sort_index(axis=0)
# print(a)
# print(data0.Sex.value_counts())

# 1. 模型构建
xTrain = train.drop('Rings', axis=1)
yTrain = train['Rings']
xTest = test.drop('Rings', axis=1)
yTest = test['Rings']
print("训练数据形状:{}".format(xTrain.shape))
print("测试数据形状:{}".format(xTest.shape))

algo = Pipeline(steps=[
    ('one', OneHotEncoder(categorical_features=[0])),
    ('ft', FunctionTransformer(func=lambda t: t.toarray(), accept_sparse='csr')),
    ('poly', PolynomialFeatures(degree=4)),
    # ('scaler', StandardScaler()),
    # ('algo', KNeighborsClassifier(n_neighbors=5))
    # ('algo', DecisionTreeClassifier(max_depth=30))
    # ('algo', RandomForestClassifier(n_estimators=100, max_depth=10))
    ('algo', GradientBoostingClassifier(n_estimators=100, max_depth=3))
])
algo.fit(xTrain, yTrain)

print("训练数据效果:{}".format(algo.score(xTrain, yTrain)))
print("测试数据效果:{}".format(algo.score(xTest, yTest)))

x_old = data.drop('Rings', axis=1)
y_old = data['Rings']
print("在原始数据集上的效果:{}".format(algo.score(x_old, y_old)))

x_train_old = train_old.drop('Rings', axis=1)
y_train_old = train_old['Rings']
print("在原始训练集上的效果:{}".format(algo.score(x_train_old, y_train_old)))

x_test_old = test_old.drop('Rings', axis=1)
y_test_old = test_old['Rings']
print("在原始测试集上的效果:{}".format(algo.score(x_test_old, y_test_old)))
