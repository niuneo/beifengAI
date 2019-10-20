# -- encoding:utf-8 --
"""
@File : 02_案例代码：基于SVM算法的鸢尾花分类
@Author: Octal_H
@Date : 2019/10/19
@Desc : 
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

mpl.rcParams['font.sans-serif'] = [u'simHei']

np.random.seed(0)

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = '../datas/iris.data'
names = ['A', 'B', 'C', 'D', 'cla']
df = pd.read_csv(path, header=None, names=names)
# 看一下二分类的话，就把下面这行代码注释去掉
df = df[df.cla != 'Iris-virginica']

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
algo = SVC(C=1.0, kernel='linear', probability=True)

# 7. 模型的训练
algo.fit(x_train, y_train)

# 8. 模型效果评估
train_predict = algo.predict(x_train)
test_predict = algo.predict(x_test)
print("测试集上的效果(准确率):{}".format(algo.score(x_test, y_test)))
print("训练集上的效果(准确率):{}".format(algo.score(x_train, y_train)))
print("测试集上的效果(分类评估报告):\n{}".format(classification_report(y_test, test_predict)))
print("训练集上的效果(分类评估报告):\n{}".format(classification_report(y_train, train_predict)))

# 9. 其它
print("=" * 100)
print("预测值:\n{}".format(test_predict))
print("预测的Y值(类别):\n{}".format(label_encoder.inverse_transform(test_predict)))
print("实际值:\n{}".format(y_test))
print("返回的预测概率值（要求参数probability必须为True）:\n{}".format(algo.predict_proba(x_test)))

print("支持向量在训练数据中的下标:\n{}".format(algo.support_))
print(x_train.iloc[22, :].ravel())
print("具体的支持向量坐标值:\n{}".format(algo.support_vectors_))
print("SVM算法中训练出来的权重系数w:(仅支持LinearSVM)\n{}".format(algo.coef_))
print("SVM算法中训练出来的偏置项b:(仅支持LinearSVM)\n{}".format(algo.intercept_))
print("具体的决策函数值:\n{}".format(algo.decision_function(x_test)))
print("预测值:\n{}".format(test_predict))
