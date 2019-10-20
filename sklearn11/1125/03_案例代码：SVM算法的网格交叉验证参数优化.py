# -- encoding:utf-8 --
"""
@File : 03_案例代码：SVM算法的网格交叉验证参数优化
@Author: Octal_H
@Date : 2019/10/20
@Desc : 
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

mpl.rcParams['font.sans-serif'] = [u'simHei']

np.random.seed(0)

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = '../datas/iris.data'
names = ['A', 'B', 'C', 'D', 'cla']
df = pd.read_csv(path, header=None, names=names)
# 2. 数据清洗

# # 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = df[names[0:-1]]
Y = df[names[-1]]
label_encoder = LabelEncoder()
label_encoder.fit(Y)
Y = label_encoder.transform(Y)

# 4. 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 6. 模型对象的构建
svc = SVC(C=1.0, kernel='linear', probability=True)
parameters = {
    "C": [0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
    "kernel": ['linear', 'rbf', 'poly'],
    "gamma": ['auto', 0.01, 0.1, 0.2, 0.5, 1.0],
    "degree": [2, 3]
}
algo = GridSearchCV(estimator=svc, param_grid=parameters, cv=5, verbose=1)

# 7. 模型的训练
algo.fit(x_train, y_train)

print("最优模型参数:{}".format(algo.best_params_))

# 8. 模型效果评估
train_predict = algo.predict(x_train)
test_predict = algo.predict(x_test)
print("测试集上的效果(准确率):{}".format(algo.score(x_test, y_test)))
print("训练集上的效果(准确率):{}".format(algo.score(x_train, y_train)))
print("测试集上的效果(分类评估报告):\n{}".format(classification_report(y_test, test_predict)))
print("训练集上的效果(分类评估报告):\n{}".format(classification_report(y_train, train_predict)))
