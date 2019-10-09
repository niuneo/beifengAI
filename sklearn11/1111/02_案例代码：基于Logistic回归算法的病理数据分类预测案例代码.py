# -- encoding:utf-8 --
"""
@File : 02_案例代码：基于Logistic回归算法的病理数据分类预测案例代码.py
@Author: Octal_H
@Date : 2019/10/9
@Desc : 
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix

mpl.rcParams['font.sans-serif'] = [u'simHei']

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = '../datas/breast-cancer-wisconsin.data'
df = pd.read_csv(path, sep=',', header=None)
df.info()

# 2. 数据清洗
df.replace("?", 0, inplace=True)

# # 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = df.iloc[:, 1:-1]
Y = df.iloc[:, -1]
print("实际的等级:{}".format(np.unique(Y)))
print(Y.value_counts())

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
"""
penalty='l2', 指定惩罚项/正则项使用L1还是L2正则
C=1.0, 指定惩罚项系数
fit_intercept=True, 是否训练截距项
solver='liblinear', 给定求解方式，可选参数:'newton-cg', 'lbfgs', 'liblinear', 'sag'
max_iter=100, 修改模型训练的最大允许迭代次数
class_weight: 只在分类API中存在，作用是给定各个类别的权重因子，权重越大的样本在模型训练的时候会着重考虑
multi_class: 给定多分类的求解方式,对于二分类应用，该参数不起任何作用，可选参数：ovr和multinomial，multinomial相当于softmax； 默认是ovr； 当该参数给定为multinomial， solver参数值最好设置为lbfgs
"""
algo = LogisticRegression()

# 7. 模型的训练
algo.fit(x_train, y_train)

# 8. 模型效果评估
train_predict = algo.predict(x_train)
test_predict = algo.predict(x_test)
print("测试集上的效果(准确率):{}".format(algo.score(x_test, y_test)))
print("训练集上的效果(准确率):{}".format(algo.score(x_train, y_train)))
print("测试集上的效果(准确率):{}".format(accuracy_score(y_test, test_predict)))
print("训练集上的效果(准确率):{}".format(accuracy_score(y_train, train_predict)))
print("测试集上的效果(混淆矩阵):\n{}".format(confusion_matrix(y_test, test_predict)))
print("训练集上的效果(混淆矩阵):\n{}".format(confusion_matrix(y_train, train_predict)))
print("测试集上的效果(召回率y=4):{}".format(recall_score(y_test, test_predict, pos_label=4)))
print("训练集上的效果(召回率y=4):{}".format(recall_score(y_train, train_predict, pos_label=4)))
print("测试集上的效果(精确率y=4):{}".format(precision_score(y_test, test_predict, pos_label=4)))
print("训练集上的效果(精确率y=4):{}".format(precision_score(y_train, train_predict, pos_label=4)))
print("测试集上的效果(分类评估报告):\n{}".format(classification_report(y_test, test_predict)))
print("训练集上的效果(分类评估报告):\n{}".format(classification_report(y_train, train_predict)))

# 9. 调整阈值
print("参数值theta:\n{}".format(algo.coef_))
print("参数值截距项:\n{}".format(algo.intercept_))
print(test_predict)
print("decision_function API返回值:\n{}".format(algo.decision_function(x_test)))
print("sigmoid函数返回的概率值:\n{}".format(algo.predict_proba(x_test)))
print("sigmoid函数返回的经过对数转换之后的概率值:\n{}".format(algo.predict_log_proba(x_test)))
# 更改阈值，看模型效果
print("更改阈值前模型效果")
print("测试集上的效果(分类评估报告):\n{}".format(classification_report(y_test, test_predict)))
print("更改阈值后模型效果")
a = 0.2
y_predict_proba = algo.predict_proba(x_test)[:, 1]
y_predict_proba[y_predict_proba > a] = 4
y_predict_proba[y_predict_proba < a] = 2
print("测试集上的效果(分类评估报告):\n{}".format(classification_report(y_test, y_predict_proba)))


