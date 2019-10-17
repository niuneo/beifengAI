# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/22
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(28)
warnings.filterwarnings('ignore')

# 1. 读取数据形成DataFrame
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
file_path = '../datas/boston_housing.data'
df = pd.read_csv(file_path, header=None, sep='\\s+', names=names)
# print(df.head())
# df.info()

# 2. 数据的分割
x = df[names[:13]]
y = df[names[13]]
y = y.ravel()
print("样本数据量:%d, 特征个数:%d" % x.shape)
print("目标属性样本数据量:%d" % y.shape[0])

# 3. 数据的划分
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=28)
print("训练数据特征属性形状:{}, 测试数据特征形状:{}".format(x_train.shape, x_test.shape))

# 4. 特征工程

# 5. 构建管道流对象
poly = PolynomialFeatures(degree=2)
lasso = Lasso(fit_intercept=True, alpha=0.048329302385717518)

# 6. 模型训练
x_train = poly.fit_transform(x_train, y_train)
x_test = poly.transform(x_test)
# x_train = np.asarray(x_train)
# x_test = np.asarray(x_test)
lasso.fit(x_train, y_train)

# 7. 看一下参数项
threshold = 0.0
print("参数项数目:{}, 参数项为零的数目:{}, 参数项的绝对值小于等于阈值的数目:{}, 具体参数项值:\n{}".format(
    np.size(lasso.coef_),
    np.sum(lasso.coef_ == 0),
    np.sum(np.abs(lasso.coef_) <= threshold),
    lasso.coef_))
print("截距项:\n{}".format(lasso.intercept_))

# 8. 做一个特征选择，选择大于阈值的那些特征
print("原始特征形状:{}".format(x_train.shape))
coef_ = np.abs(lasso.coef_)
x_train = x_train[:, coef_ > threshold]
x_test = x_test[:, coef_ > threshold]
print("提取特征后的数据形状:{}".format(x_train.shape))

# 9. 基于特征选择后的数据做一个Ridge模型
ridge = Pipeline([
    # ('poly', PolynomialFeatures(degree=2)),
    ('linear', Ridge(alpha=10.0, fit_intercept=False))
])
ridge.fit(x_train, y_train)

# 模型效果评估
train_pred = ridge.predict(x_train)
test_pred = ridge.predict(x_test)
print("训练数据的MSE评估指标:{}".format(mean_squared_error(y_train, train_pred)))
print("测试数据的MSE评估指标:{}".format(mean_squared_error(y_test, test_pred)))
print("训练数据的R2评估指标:{}".format(r2_score(y_train, train_pred)))
print("测试数据的R2评估指标:{}".format(r2_score(y_test, test_pred)))
print("训练数据的RMSE评估指标:{}".format(np.sqrt(mean_squared_error(y_train, train_pred))))
print("测试数据的RMSE评估指标:{}".format(np.sqrt(mean_squared_error(y_test, test_pred))))
