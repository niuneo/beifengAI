import pandas as pd
import numpy as np
import os,sys
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

data = pd.read_csv('../datas/code_0_datas/t_data.csv')

def feature_extraction(data):
    def get_days(date):  # date = '2016-08'
        days_dict = {'2016-01': {'working_days': 20, 'weekend_days': 10, 'holiday_days': 1},
                     '2016-02': {'working_days': 18, 'weekend_days': 8, 'holiday_days': 3},
                     '2016-03': {'working_days': 23, 'weekend_days': 8, 'holiday_days': 0},
                     '2016-04': {'working_days': 20, 'weekend_days': 9, 'holiday_days': 1},
                     '2016-05': {'working_days': 21, 'weekend_days': 9, 'holiday_days': 1},
                     '2016-06': {'working_days': 21, 'weekend_days': 8, 'holiday_days': 1},
                     '2016-07': {'working_days': 21, 'weekend_days': 10, 'holiday_days': 0},
                     '2016-08': {'working_days': 23, 'weekend_days': 8, 'holiday_days': 0},
                     '2016-09': {'working_days': 21, 'weekend_days': 8, 'holiday_days': 1},
                     '2016-10': {'working_days': 18, 'weekend_days': 10, 'holiday_days': 3},
                     '2016-11': {'working_days': 22, 'weekend_days': 8, 'holiday_days': 0},
                     '2016-12': {'working_days': 22, 'weekend_days': 9, 'holiday_days': 0},

                     '2017-01': {'working_days': 18, 'weekend_days': 9, 'holiday_days': 4},
                     '2017-02': {'working_days': 20, 'weekend_days': 8, 'holiday_days': 0},
                     '2017-03': {'working_days': 23, 'weekend_days': 8, 'holiday_days': 0},
                     '2017-04': {'working_days': 19, 'weekend_days': 10, 'holiday_days': 1},
                     '2017-05': {'working_days': 21, 'weekend_days': 8, 'holiday_days': 2},
                     '2017-06': {'working_days': 22, 'weekend_days': 8, 'holiday_days': 0},
                     '2017-07': {'working_days': 21, 'weekend_days': 10, 'holiday_days': 0},
                     '2017-08': {'working_days': 23, 'weekend_days': 8, 'holiday_days': 0},
                     '2017-09': {'working_days': 21, 'weekend_days': 9, 'holiday_days': 0},
                     '2017-10': {'working_days': 18, 'weekend_days': 9, 'holiday_days': 4},
                     '2017-11': {'working_days': 22, 'weekend_days': 8, 'holiday_days': 0},
                     '2017-12': {'working_days': 21, 'weekend_days': 10, 'holiday_days': 0},

                     '2018-01': {'working_days': 22, 'weekend_days': 8, 'holiday_days': 1},
                     '2018-02': {'working_days': 17, 'weekend_days': 8, 'holiday_days': 3},
                     '2018-03': {'working_days': 22, 'weekend_days': 9, 'holiday_days': 0},
                     '2018-04': {'working_days': 20, 'weekend_days': 9, 'holiday_days': 1},
                     '2018-05': {'working_days': 22, 'weekend_days': 8, 'holiday_days': 1},
                     '2018-06': {'working_days': 20, 'weekend_days': 9, 'holiday_days': 1},
                     '2018-07': {'working_days': 22, 'weekend_days': 9, 'holiday_days': 0},
                     '2018-08': {'working_days': 23, 'weekend_days': 8, 'holiday_days': 0},
                     '2018-09': {'working_days': 21, 'weekend_days': 9, 'holiday_days': 1},
                     '2018-10': {'working_days': 18, 'weekend_days': 10, 'holiday_days': 3},
                     '2018-11': {'working_days': 22, 'weekend_days': 8, 'holiday_days': 0},
                     '2018-12': {'working_days': 21, 'weekend_days': 10, 'holiday_days': 0},
                     }
        working_days = days_dict[date]['working_days']
        weekend_days = days_dict[date]['weekend_days']
        holiday_days = days_dict[date]['holiday_days']
        result_dict = {'working_days': working_days, 'weekend_days': weekend_days, 'holiday_days': holiday_days}
        return result_dict
    def get_season(date):
        season = ''
        season_dict = {'spring':[1, 2, 3], 'summer': [4, 5, 6], 'autumn': [7, 8, 9], 'winter':[10, 11, 12] }
        month = int(date[5:7])
        if month in season_dict['spring']:
            season = 'spring'
        elif month in season_dict['summer']:
            season = 'summer'
        elif month in season_dict['autumn']:
            season = 'autumn'
        elif month in season_dict['winter']:
            season = 'winter'
        return season

    data['year'] = pd.Series(list(map(lambda x: x[0:4], data['date'])))
    data['month'] = pd.Series(list(map(lambda x: x[5:7], data['date'])))
    data['season'] = pd.Series(list(map(lambda x: get_season(x), data['date'])))
    data['working_days'] = pd.Series(list(map(lambda x: get_days(x)['working_days'], data['date'])))
    data['weekend_days'] = pd.Series(list(map(lambda x: get_days(x)['weekend_days'], data['date'])))
    data['holiday_days'] = pd.Series(list(map(lambda x: get_days(x)['holiday_days'], data['date'])))
    return data
datas = feature_extraction(data)

X = datas[['year', 'month', 'season', 'working_days', 'weekend_days', 'holiday_days']]
print(X)
Y = datas['num']
print(Y)
oh = OneHotEncoder(categorical_features=[0, 1, 2])
le = LabelEncoder()
X['year'] = le.fit_transform(X['year'])
X['month'] = le.fit_transform(X['month'])
X['season'] = le.fit_transform(X['season'])
X = oh.fit_transform(X).toarray()
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 2, test_size = 0.2)

#线性回归
LR = LinearRegression()
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
# #决策树
# DT = DecisionTreeRegressor()
# DT.fit(x_train, y_train)
# score = DT.score(x_test, y_test)
# y_pred = DT.predict(x_test)
# r2_score = metrics.r2_score(y_test, y_pred)
# print(score)
# print(r2_score)
# print(y_pred)
# print(y_test)

#随机森林
# RFR = RandomForestRegressor()
# RFR.fit(X, Y)
# y_pred = RFR.predict(x_test)
# r2_score = metrics.r2_score(y_test, y_pred)
# print("随机森林回归的均方误差为:", metrics.mean_squared_error(y_test, y_pred))
# print(r2_score)
a = [1,2,3,4,5,6]
b = [list(y_pred), y_test]
c = ['red', 'blue']
for i in range(2):
    plt.plot(a, b[i], 'o-', color = c[i])
plt.show()

dt = pd.DataFrame([['2018-10', 48093]], columns = ['date', 'num'], index = None)
dts = feature_extraction(dt)
a = dts[['year', 'month', 'season', 'working_days', 'weekend_days', 'holiday_days']]
b = dts['num']
a['year'] = le.transform(a['year'])
a['month'] = le.transform(a['month'])
a['season'] = le.transform(a['season'])
print(a)
# a = oh.fit_transform(a).toarray()
# print(a)
# b_pred = RFR.predict(a)
# print(b_pred)



