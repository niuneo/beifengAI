# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/10
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings

import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn import metrics

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
## 拦截异常
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

## 数据加载
path = "../datas/iris.data"
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'cla']
df = pd.read_csv(path, header=None, names=names)
df['cla'].value_counts()


# df = df[df.cla != 'Iris-virginica']


def parseRecord(record):
    result = []
    r = zip(names, record)
    for name, v in r:
        if name == 'cla':
            if v == 'Iris-setosa':
                result.append(1)
            elif v == 'Iris-versicolor':
                result.append(2)
            elif v == 'Iris-virginica':
                result.append(3)
            else:
                result.append(np.nan)
        else:
            result.append(float(v))
    return result


### 1. 数据转换为数字以及分割
## 数据转换
datas = df.apply(lambda r: pd.Series(parseRecord(r), index=names), axis=1)
## 异常数据删除
datas = datas.dropna(how='any')
## 数据分割
X = datas[names[0:1]]
Y = datas[names[-1]]
## 数据抽样(训练数据和测试数据分割)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

### 5. 模型构建
lr = LogisticRegressionCV(Cs=np.logspace(-4, 1, 50), multi_class='multinomial')
lr.fit(X_train, Y_train)

# 画图预测
theta10, theta20, theta30 = lr.intercept_
theta11, theta21, theta31 = lr.coef_[:, 0]
print(theta10)
print(theta20)
print(theta30)
print(theta11)
print(theta21)
print(theta31)
print(lr.coef_)
print(lr.intercept_)

## 画图2：预测结果画图
X_train = np.array(X_train).reshape((-1, 1))
x_min = np.min(X_train).astype(np.float32) - 0.5
x_max = np.max(X_train).astype(np.float32) + 0.5

Y_train = np.array(Y_train).reshape((-1, 1))
test = np.concatenate([X_train, Y_train], axis=1)
test.sort(axis=0)
y_predict = lr.predict(test[:, 0].reshape(-1, 1))
plt.figure(figsize=(12, 9), facecolor='w')
plt.plot(test[:, 0], test[:, 1], 'ro', markersize=6, zorder=3, label=u'真实值')
plt.plot(test[:, 0], y_predict, 'go', markersize=10, zorder=2,
         label=u'Logis算法预测值,准确率=%.3f' % lr.score(X_test, Y_test))

# 画第一条线
plt.plot([x_min, x_max], [theta11 * x_min + theta10, theta11 * x_max + theta10], 'r-', label=u'第一条线')
plt.plot([x_min, x_max], [theta21 * x_min + theta20, theta21 * x_max + theta20], 'b-', label=u'第二条线')
plt.plot([x_min, x_max], [theta31 * x_min + theta30, theta31 * x_max + theta30], 'g-', label=u'第三条线')
plt.legend(loc='lower right')
plt.xlabel(u'{}'.format(names[0:1]), fontsize=18)
plt.ylabel(u'种类', fontsize=18)
plt.title(u'鸢尾花数据分类', fontsize=20)
plt.grid()
plt.show()

x_t = test[-10:, 0].reshape(-1,1)
y_t = lr.decision_function(x_t)
print(y_t)
print(x_t.reshape(-1))
print(lr.predict(x_t))
print(test[-10:, 1])
print([theta31 * x_min + theta30, theta31 * 7.1 + theta30])
