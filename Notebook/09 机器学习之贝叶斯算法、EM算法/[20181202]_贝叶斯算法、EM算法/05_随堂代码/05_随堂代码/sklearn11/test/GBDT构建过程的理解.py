# -- encoding:utf-8 --
"""
Create by ibf on 2018/9/2
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

print("回归效果")
np.random.seed(28)
x = np.random.randn(10, 2) * 5
y = np.random.randn(10, 1) * 3
y_true = y

algo = DecisionTreeRegressor(max_depth=1)
algo.fit(x, y)
print("训练数据上的效果:{}".format(algo.score(x, y)))
print("实际y值:\n{}".format(y.reshape(-1)))
print("预测y值:\n{}".format(algo.predict(x).reshape(-1)))

# GBDT的构建过程（回归， 使用平方和损失函数的执行过程）
models = []
# 构建第一个模型: 在回归中，第一个模型预测为均值
m1 = np.mean(y)
models.append(m1)
# 构建后面的模型（全部为回归决策树）
learn_rate = 0.01
pre_m = m1
n = 10000
for i in range(n):
    # 更改y值
    if i == 0:
        y = y - learn_rate * pre_m
    else:
        y = y - learn_rate * pre_m.predict(x).reshape(y.shape)
    # 模型训练
    model = DecisionTreeRegressor(max_depth=1)
    model.fit(x, y)
    models.append(model)
    pre_m = model
print("模型构建完成")
print("开始预测")
y_hat = np.zeros_like(y)
print(y_hat.shape)
for i in range(n + 1):
    # 获取第i个模型
    model = models[i]
    # 使用模型得到预测值
    if i == 0:
        y_hat = y_hat + learn_rate * model
    else:
        y_hat = y_hat + learn_rate * model.predict(x).reshape(y.shape)
print("实际值为:\n{}".format(y_true.reshape(-1)))
print("预测值为:\n{}".format(y_hat.reshape(-1)))
print("单个简单的决策树的效果:{}".format(r2_score(y_true, algo.predict(x))))
print("GBDT的效果:{}".format(r2_score(y_true, y_hat)))

#
print("\n\n分类效果(二分类)")
x = np.random.randn(10, 2) * 5
y = np.array([1] * 6 + [0] * 4).astype(np.float)
y_true = y
models = []
# 构建第一个模型: 在回归中，第一个模型预测为均值
m1 = np.log(6.0 / 4)
models.append(m1)
# 构建后面的模型（全部为回归决策树）
learn_rate = 0.01
pre_m = m1
n = 10000
for i in range(n):
    # 更改y值
    if i == 0:
        y = y - learn_rate * pre_m
    else:
        y = y - learn_rate * pre_m.predict(x).reshape(y.shape)
    # 模型训练
    model = DecisionTreeRegressor(max_depth=1)
    model.fit(x, y)
    models.append(model)
    pre_m = model
print("模型构建完成")
print("开始预测")
y_hat = np.zeros_like(y)
print(y_hat.shape)
for i in range(n + 1):
    # 获取第i个模型
    model = models[i]
    # 使用模型得到预测值
    if i == 0:
        y_hat = y_hat + learn_rate * model
    else:
        y_hat = y_hat + learn_rate * model.predict(x).reshape(y.shape)
print("初步预测值为:\n{}".format(y_hat.T))
y_hat = np.asarray(y_hat)
y_hat[y_hat < 0.5] = 0
y_hat[y_hat >= 0.5] = 1.0
print("最终预测值为:\n{}".format(y_hat))
print("实际值为:\n{}".format(y_true))
algo = DecisionTreeClassifier(max_depth=1)
algo.fit(x, y_true)
print("单个简单的决策树的效果:{}".format(accuracy_score(y_true, algo.predict(x))))
print("GBDT的效果:{}".format(accuracy_score(y_true, y_hat)))

# https://www.cnblogs.com/ModifyRong/p/7744987.html
print("\n\n分类效果（多分类）")
x = np.random.randn(10, 2) * 5
y = np.array([0] * 2 + [1] * 4 + [2] * 4).astype(np.float)
y_true = y
# y编程哑编码的形式
y1 = np.array([1] * 2 + [0] * 8).astype(np.float)
y2 = np.array([0] * 2 + [1] * 4 + [0] * 4).astype(np.float)
y3 = np.array([0] * 6 + [1] * 4).astype(np.float)
ys = [y1, y2, y3]
# GBDT的构建过程（分类， 使用平方和损失函数的执行过程）
models = []
# 构建第一个模型: 在分类中，第一个预测为0
m1 = np.array([0, 0, 0])
models.append(m1)
# 构建后面的模型（全部为回归决策树）
learn_rate = 0.01
pre_m = m1
n = 100
for i in range(n):
    # 更改y值
    tmp_algo = []
    for k in range(3):
        if i == 0:
            p = np.exp(pre_m[k]) / np.sum(np.exp(pre_m))
            ys[k] = ys[k] - p
        else:
            pred_y = np.array(list(map(lambda algo: algo.predict(x), pre_m)))
            p = np.exp(pred_y[k].reshape(y.shape)) / np.sum(np.exp(pred_y), axis=0)
            ys[k] = ys[k] - p
        # 模型训练
        model = DecisionTreeRegressor(max_depth=1)
        model.fit(x, ys[k])
        tmp_algo.append(model)

    models.append(tmp_algo)
    pre_m = tmp_algo

print("模型构建完成")
print("总模型数目:{}".format(np.size(models)))
print("开始预测")
y_hat = np.zeros(shape=(3, y.shape[0]))
print(y_hat.shape)
for i in range(n + 1):
    # 获取第i个模型
    model = models[i]
    # 使用模型得到预测值
    for k in range(3):
        if i == 0:
            y_hat[k] = model[k]
        else:
            y_hat[k] += model[k].predict(x).reshape(y.shape)
print("初步预测值为:\n{}".format(y_hat.T))
print("最终预测值为:\n{}".format(np.argmax(y_hat, axis=0)))
print("实际值为:\n{}".format(y_true))
algo = DecisionTreeClassifier(max_depth=1)
algo.fit(x, y_true)
print("单个简单的决策树的效果:{}".format(accuracy_score(y_true, algo.predict(x))))
print("GBDT的效果:{}".format(accuracy_score(y_true, np.argmax(y_hat, axis=0))))
