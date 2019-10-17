# coding=utf-8
import pandas as pd
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
import warnings

warnings.filterwarnings("ignore")
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

DATA_SET_LOCATION = "./baoyu.csv"

DATA = pd.read_csv(DATA_SET_LOCATION,
                   names=['Sex', 'Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings'])

# 变量转换
# 先直接做变量转换
# def sex2Num(x):
#     if x == 'I':
#         return 1
#     elif x == 'M':
#         return 2
#     else:
#         return 3
# DATA['Sex'] = DATA['Sex'].apply(sex2Num)

# 使用哑变量进行变量转换
# 对效果并没有多大影响
sexDummies = pd.get_dummies(DATA['Sex'])
DATA = DATA.join(sexDummies)
DATA.drop(['Sex'], axis=1, inplace=True)

print(DATA.info())

X = DATA.drop('Rings', axis=1)
Y = DATA['Rings']

# print(X.shape)
# print(Y.shape)
# print(type(X))
# print(type(Y))

# print(Y.apply(pd.Series.velue_counts()))

# 数据分割
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.25, random_state=0)
print(len(Y.value_counts()))
# 问题一怎样选择标准化
# 先不使用标准化

# 1.使用数据归一化处理
ss = MinMaxScaler()
xTrain = ss.fit_transform(xTrain)
xTest = ss.transform(xTest)
# 使用这个标准化并没有什么卵用

# 选择模型
# 1.使用KNN
knn = KNeighborsClassifier(n_neighbors=400, algorithm='kd_tree')
knnAfter = knn.fit(xTrain, yTrain)
print("KNN训练数据集:{}".format(knn.score(xTrain, yTrain)))
print("KNN:", knnAfter.score(xTest, yTest))

# 2.使用逻辑回归
# 这里面的参数并不是很清楚
lr = LogisticRegressionCV(multi_class='ovr', fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, penalty='l2',
                          solver='lbfgs', tol=0.1)
lr.fit(xTrain, yTrain)
print("Logistic训练数据集:{}".format(lr.score(xTrain, yTrain)))
print("Logistic:", lr.score(xTest, yTest))


# 使用SoftMax
# 分类不能多于特征数量吗?
# 运行有错误
# lr = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-5, 1, 100),
#                           multi_class='multinomial', penalty='l2', solver='lbfgs')
# lr.fit(xTrain, yTrain)
# print("SoftMax:", lr.score(xTest, yTest))

# 使用决策树，随机森林
deci = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=None, min_samples_split=10, min_samples_leaf=2,
                              max_leaf_nodes=40)
deci.fit(xTrain, yTrain)
print("DECI训练数据集:{}".format(deci.score(xTrain, yTrain)))
print("DECI:", deci.score(xTest, yTest))

# ID3和CART效果并不明显

# 使用RF
rf = RandomForestClassifier(n_estimators=80, max_leaf_nodes=40)
rf.fit(xTrain, yTrain)
print("RF训练数据集:{}".format(rf.score(xTrain, yTrain)))
print("RF:", rf.score(xTest, yTest))

# 试试Extra tree
extTree = ExtraTreesClassifier(n_estimators=80,max_depth=12, min_samples_leaf=7)
extTree.fit(xTrain, yTrain)
print("EXT训练数据集:{}".format(extTree.score(xTrain, yTrain)))
print("EXT:", extTree.score(xTest, yTest))

# AdaBoosting
ada = AdaBoostClassifier(
    DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=10, min_samples_split=10,
                           min_samples_leaf=2, max_leaf_nodes=40), n_estimators=60, learning_rate=0.5, random_state=0)
ada.fit(xTrain, yTrain)
print("ADA训练数据集:{}".format(ada.score(xTrain, yTrain)))
print("ADA:", ada.score(xTest, yTest))
