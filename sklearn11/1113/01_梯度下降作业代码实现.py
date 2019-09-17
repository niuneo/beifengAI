# -- encoding:utf-8 --
'''
@File : 01_梯度下降作业代码实现.py
@Author: Octal_H
@Date : 2019/9/16
@Desc : 
'''
'''
梯度下降作业(代码)：
  目标函数：
    y = x**2 + b * x + c
  需求：求解最小值对应的x和y(这里的最小值指的是函数y在所有数据上的一个累计最小值)
  要去：写代码
    数据：
		b: 服从均值为-1，方差为10的随机数
		c：服从均值为0，方差为1的随机数
	假定b、c这样的数据组合总共1、2、10、100、10w、100w条数据,求解在现在的数据情况下，目标函数的取最小值的时候，x和y分别对应多少？
'''

import  numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1.随机数产生
np.random.seed(28)
n = 10000
b_values = np.random.normal(loc=-1.0, scale=10.0, size=n)
c_values = np.random.normal(loc=0.0, scale=1.0, size=n)


# # 随机数可视化
# plt.figure(facecolor='w')
# # 一行两列第一个子图
# plt.subplot(1, 2, 1)
# # b_values是一个数组，这里的1000相当于把这个数组中的数字分成1000等份，然后分别计算每个区间数字出现的次数，
# # 以区间作为横轴，以出现的次数作为纵轴
# plt.hist(b_values, 1000, color='#FF0000')
#
# plt.subplot(1, 2, 2)
# plt.hist(c_values, 1000, color='#0000FF')
# plt.suptitle('可视化随机数列')
# plt.show()



def calc_min_value_one_sample(b, c, max_iter=1000, tol=0.00001, alpha=0.01):
    '''
    计算 y=x**2 + b*x + c只有一个(b,c)组合样本的时候该函数的最小值是多少
    :param b:
    :param c:
    :param max_iter: 使用梯度下降的时候，最大迭代次数
    :param tol: 使用梯度下降的时候，收敛的限制条件：指的是两次迭代之间变化小于该值的时候结束参数的更新
    :param alpha: 使用梯度下降的时候，学习率
    :return:
    '''
    def f(x, b, c):
        '''
        原函数
        :param x:
        :param b:
        :param c:
        :return:
        '''
        return x ** 2 + b * x + c

    def h(x, b, c):
        '''
        导函数
        :param x:
        :param b:
        :param c:
        :return:
        '''
        return 2 * x + b

    # 1.定义一些相关的变量
    step_change = 1.0 + tol
    step = 0
    # 随机的给顶一个初始值
    current_x = np.random.randint(low=-10, high=10)
    current_y = f(current_x, b, c)
    print('当前的样本数据为：')
    print('b={}'.format(b))
    print('c={}'.format(c))

    # 2.迭代求解
    while step_change > tol and step < max_iter:
        # a.计算函数的梯度值
        current_df = h(current_x, b, c)
        # b.基于梯度值更新模型参数x
        current_x = current_x - alpha * current_df
        # c.基于更新好的x计算y值
        pre_y = current_y
        current_y = f(current_x, b, c)
        # d.记录两次更新的y的变化大小
        step_change = np.abs(pre_y - current_y)
        # e.更新参数
        step += 1
    print('最终更新的次数为：{}，最终的变化率为：{}'.format(step, step_change))
    print('最终的结果为：{}----->{}'.format(current_x, current_y))




def calc_min_value_two_sample(b1, c1, b2, c2, max_iter=1000, tol=0.00001, alpha=0.01):
    '''
    计算 y=x**2 + b*x + c只有两个(b,c)组合样本的时候，该函数在所有组合上的最小值是多少
    :param b:
    :param c:
    :param max_iter: 使用梯度下降的时候，最大迭代次数
    :param tol: 使用梯度下降的时候，收敛的限制条件：指的是两次迭代之间变化小于该值的时候结束参数的更新
    :param alpha: 使用梯度下降的时候，学习率
    :return:
    '''

    def f1(x, b, c):
        '''
        原函数 针对单个样本的
        :param x:
        :param b:
        :param c:
        :return:
        '''
        return x ** 2 + b * x + c

    def f(x, b1, c1, b2, c2):
        y1 = f1(x, b1, c2)
        y2 = f1(x, b2, c2)
        return y1 + y2

    def h1(x, b, c):
        '''
        导函数 针对于单个样本的导函数
        :param x:
        :param b:
        :param c:
        :return:
        '''
        return 2 * x + b

    def h(x, b1, c1, b2, c2):
        y1 = h1(x, b1, c1)
        y2 = h1(x, b2, c2)
        return y1 + y2

    # 1.定义一些相关的变量
    step_change = 1.0 + tol
    step = 0
    # 随机的给顶一个初始值
    current_x = np.random.randint(low=-10, high=10)
    current_y = f(current_x, b1, c1, b2, c2)
    print('当前的样本数据为：')
    print('b={}\nb均值为：{}'.format([b1, b2], np.mean([b1, b2])))
    print('c={}'.format([c1, c2]))

    # 2.迭代求解
    while step_change > tol and step < max_iter:
        # a.计算函数的梯度值
        current_df = h(current_x, b1, c1, b2, c2)
        # b.基于梯度值更新模型参数x
        current_x = current_x - alpha * current_df
        # c.基于更新好的x计算y值
        pre_y = current_y
        current_y = f(current_x, b1, c1, b2, c2)
        # d.记录两次更新的y的变化大小
        step_change = np.abs(pre_y - current_y)
        # e.更新参数
        step += 1
    print('最终更新的次数为：{}，最终的变化率为：{}'.format(step, step_change))
    print('最终的结果为：{}----->{}'.format(current_x, current_y))



def calc_min_value_ten_sample(b_values, c_values, max_iter=1000, tol=0.00001, alpha=0.01):
    '''
    计算 y=x**2 + b*x + c只有十个(b,c)组合样本的时候，该函数在所有组合上的最小值是多少
    :param b:
    :param c:
    :param max_iter: 使用梯度下降的时候，最大迭代次数
    :param tol: 使用梯度下降的时候，收敛的限制条件：指的是两次迭代之间变化小于该值的时候结束参数的更新
    :param alpha: 使用梯度下降的时候，学习率
    :return:
    '''

    # 断言：如果assert后的表达式执行为False，那么报错；否则通过判断继续执行
    assert  len(b_values) == 10 and len(c_values) == 10

    def f1(x, b, c):
        '''
        原函数 针对单个样本的
        :param x:
        :param b:
        :param c:
        :return:
        '''
        return x ** 2 + b * x + c

    def f(x, b_values, c_values):
        y1 = f1(x, b_values[0], c_values[0])
        y2 = f1(x, b_values[1], c_values[1])
        y3 = f1(x, b_values[2], c_values[2])
        y4 = f1(x, b_values[3], c_values[3])
        y5 = f1(x, b_values[4], c_values[4])
        y6 = f1(x, b_values[5], c_values[5])
        y7 = f1(x, b_values[6], c_values[6])
        y8 = f1(x, b_values[7], c_values[7])
        y9 = f1(x, b_values[8], c_values[8])
        y10 = f1(x, b_values[9], c_values[9])
        return y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10

    def h1(x, b, c):
        '''
        导函数 针对于单个样本的导函数
        :param x:
        :param b:
        :param c:
        :return:
        '''
        return 2 * x + b

    def h(x, b_values, c_values):
        y = 0.0
        for b, c in zip(b_values, c_values):
            y += h1(x, b, c)
        return y

    # 1.定义一些相关的变量
    step_change = 1.0 + tol
    step = 0
    # 随机的给顶一个初始值
    current_x = np.random.randint(low=-10, high=10)
    current_y = f(current_x, b_values, c_values)
    print('当前的样本数据为：')
    print('b={}\nb均值为：{}'.format(b_values, np.mean(b_values)))
    print('c={}'.format(c_values))

    # 2.迭代求解
    while step_change > tol and step < max_iter:
        # a.计算函数的梯度值
        current_df = h(current_x, b_values, c_values)
        # b.基于梯度值更新模型参数x
        current_x = current_x - alpha * current_df
        # c.基于更新好的x计算y值
        pre_y = current_y
        current_y = f(current_x, b_values, c_values)
        # d.记录两次更新的y的变化大小
        step_change = np.abs(pre_y - current_y)
        # e.更新参数
        step += 1
    print('最终更新的次数为：{}，最终的变化率为：{}'.format(step, step_change))
    print('最终的结果为：{}----->{}'.format(current_x, current_y))


def calc_min_value_n_sample_BGD(b_values, c_values, n, max_iter=1000, tol=0.00001, alpha=0.01):
    '''
    计算 y=x**2 + b*x + c只有n个(b,c)组合样本的时候，该函数在所有组合上的最小值是多少
    :param b:
    :param c:
    :param max_iter: 使用梯度下降的时候，最大迭代次数
    :param tol: 使用梯度下降的时候，收敛的限制条件：指的是两次迭代之间变化小于该值的时候结束参数的更新
    :param alpha: 使用梯度下降的时候，学习率
    :return:
    '''



    def f1(x, b, c):
        '''
        原函数 针对单个样本的
        :param x:
        :param b:
        :param c:
        :return:
        '''
        return x ** 2 + b * x + c

    def f(x, b_values, c_values):
        y = 0.0
        for b, c in zip(b_values, c_values):
            # 在这里求均值的目的：1：为了梯度更新的时候，梯度值比较小，这样x更新的时候才会收敛
            # 2：为了防止数据量太大，溢出越界
            y += f1(x, b, c) / n
        return y

    def h1(x, b, c):
        '''
        导函数 针对于单个样本的导函数
        :param x:
        :param b:
        :param c:
        :return:
        '''
        return 2 * x + b

    def h(x, b_values, c_values):
        y = 0.0
        for b, c in zip(b_values, c_values):
            y += h1(x, b, c) / n
        return y

    # 1.定义一些相关的变量
    step_change = 1.0 + tol
    step = 0
    # 随机的给顶一个初始值
    current_x = np.random.randint(low=-10, high=10)
    current_y = f(current_x, b_values, c_values)
    print('当前的样本数据为：')
    print('b={}\nb均值为：{}'.format(b_values, np.mean(b_values)))
    print('c={}'.format(c_values))

    # 画图相关的变量
    y_values = []
    y_values.append(current_y)
    y_change_values = []

    # 2.迭代求解
    while step_change > tol and step < max_iter:
        # a.计算函数的梯度值
        current_df = h(current_x, b_values, c_values)
        # b.基于梯度值更新模型参数x
        current_x = current_x - alpha * current_df
        # c.基于更新好的x计算y值
        pre_y = current_y
        current_y = f(current_x, b_values, c_values)
        # d.记录两次更新的y的变化大小
        step_change = np.abs(pre_y - current_y)
        # e.更新参数
        step += 1
        # f.添加画图相关的值
        y_values.append(current_y)
        y_change_values.append(step_change)

    print('最终更新的次数为：{}，最终的变化率为：{}'.format(step, step_change))
    print('最终的结果为：{}----->{}'.format(current_x, current_y))

    # 画图
    plt.figure(facecolor='w')
    plt.subplot(1, 2, 1)
    plt.plot(range(step), y_change_values, 'r-')
    plt.xlabel('迭代次数')
    plt.ylabel('每次迭代值的y值变化大小')

    plt.subplot(1, 2, 2)
    plt.plot(range(step+1), y_values, 'g-')
    plt.xlabel('迭代次数')
    plt.ylabel('函数值')
    plt.suptitle('BGD的变化情况')
    plt.show()



def calc_min_value_n_sample_SGD(b_values, c_values, n, max_iter=1000, tol=0.00001, alpha=0.01):
    '''
    计算 y=x**2 + b*x + c只有n个(b,c)组合样本的时候，该函数在所有组合上的最小值是多少
    :param b:
    :param c:
    :param max_iter: 使用梯度下降的时候，最大迭代次数
    :param tol: 使用梯度下降的时候，收敛的限制条件：指的是两次迭代之间变化小于该值的时候结束参数的更新
    :param alpha: 使用梯度下降的时候，学习率
    :return:
    '''



    def f1(x, b, c):
        '''
        原函数 针对单个样本的
        :param x:
        :param b:
        :param c:
        :return:
        '''
        return x ** 2 + b * x + c

    def f(x, b_values, c_values):
        y = 0.0
        for b, c in zip(b_values, c_values):
            # 在这里求均值的目的：1：为了梯度更新的时候，梯度值比较小，这样x更新的时候才会收敛
            # 2：为了防止数据量太大，溢出越界
            y += f1(x, b, c) / n
        return y

    def h1(x, b, c):
        '''
        导函数 针对于单个样本的导函数
        :param x:
        :param b:
        :param c:
        :return:
        '''
        return 2 * x + b

    def h(x, b_values, c_values):
        y = 0.0
        for b, c in zip(b_values, c_values):
            y += h1(x, b, c) / n
        return y

    # 1.定义一些相关的变量
    step_change = 1.0 + tol
    epochs = 0
    change_number = 0

    # 随机的给顶一个初始值
    current_x = np.random.randint(low=-10, high=10)
    current_y = f(current_x, b_values, c_values)
    print('当前的样本数据为：')
    print('b={}\nb均值为：{}'.format(b_values, np.mean(b_values)))
    print('c={}'.format(c_values))

    # 画图相关的变量
    y_values = []
    y_values.append(current_y)
    y_change_values = []

    # 2.迭代求解
    while step_change > tol and epochs < max_iter:
        # a.将更新的样本顺序打乱(产生n个乱序随机数)
        random_index = np.random.permutation(n)
        # b.遍历所有样本，更新模型参数
        for index in random_index:
            # a.计算函数的梯度值(只计算一个样本的梯度值)
            current_df = h1(current_x, b_values[index], c_values[index])
            # b.基于梯度值更新模型参数x
            current_x = current_x - alpha * current_df
            # c.基于更新好的x计算y值
            pre_y = current_y
            current_y = f(current_x, b_values, c_values)
            # d.记录两次更新的y的变化大小
            step_change = np.abs(pre_y - current_y)
            # e.更新参数
            change_number += 1
            # f.添加画图相关的值
            y_values.append(current_y)
            y_change_values.append(step_change)
            # g.做一个跳出操作
            if step_change < tol:
                break

        # 每从头到尾遍历一次数据，称为更新了一个epoch
        epochs += 1


    print('最终更新的次数为：{}，参数的更新次数：{}, 最终的变化率为：{}'.format(epochs, change_number, step_change))
    print('最终的结果为：{}----->{}'.format(current_x, current_y))

    # 画图
    plt.figure(facecolor='w')
    plt.subplot(1, 2, 1)
    plt.plot(range(change_number), y_change_values, 'r-')
    plt.xlabel('迭代次数')
    plt.ylabel('每次迭代值的y值变化大小')

    plt.subplot(1, 2, 2)
    plt.plot(range(change_number+1), y_values, 'g-')
    plt.xlabel('迭代次数')
    plt.ylabel('函数值')
    plt.suptitle('SGD的变化情况')
    plt.show()







# calc_min_value_one_sample(b_values[0], c_values[0])
# calc_min_value_two_sample(b_values[0], c_values[0], b_values[1], c_values[1])
# calc_min_value_ten_sample(b_values, c_values)
# calc_min_value_n_sample_BGD(b_values, c_values, n)
calc_min_value_n_sample_SGD(b_values, c_values, n)