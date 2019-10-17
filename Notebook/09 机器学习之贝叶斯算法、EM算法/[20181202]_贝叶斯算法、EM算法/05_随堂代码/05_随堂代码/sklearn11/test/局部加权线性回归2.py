# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/8
"""

from __future__ import print_function

import numpy as np
from numpy import *
import matplotlib.pylab as plt

def standRegres(xArr, yArr):
    '''
    Description：
        线性回归
    Args:
        xArr ：输入的样本数据，包含每个样本数据的 feature
        yArr ：对应于输入数据的类别标签，也就是每个样本对应的目标变量
    Returns:
        ws：回归系数
    '''

    # mat()函数将xArr，yArr转换为矩阵 mat().T 代表的是对矩阵进行转置操作
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 矩阵乘法的条件是左矩阵的列数等于右矩阵的行数
    xTx = xMat.T * xMat
    xTx = xTx + eye(shape(xMat)[1]) * 0.2
    # 因为要用到xTx的逆矩阵，所以事先需要确定计算得到的xTx是否可逆，条件是矩阵的行列式不为0
    # linalg.det() 函数是用来求得矩阵的行列式的，如果矩阵的行列式为0，则这个矩阵是不可逆的，就无法进行接下来的运算
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 最小二乘法
    ws = xTx.I * (xMat.T * yMat)
    return ws


# 局部加权线性回归
def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
        Description：
            局部加权线性回归，在待预测点附近的每个点赋予一定的权重，在子集上基于最小均方差来进行普通的回归。
        Args：
            testPoint：样本点
            xArr：样本的特征数据，即 feature
            yArr：每个样本对应的类别标签，即目标变量
            k:关于赋予权重矩阵的核的一个参数，与权重的衰减速率有关
        Returns:
            testPoint * ws：数据点与具有权重的系数相乘得到的预测点
        Notes:
            这其中会用到计算权重的公式，w = e^((x^((i))-x) / -2k^2)
            理解：x为某个预测点，x^((i))为样本点，样本点距离预测点越近，贡献的误差越大（权值越大），越远则贡献的误差越小（权值越小）。
            关于预测点的选取，在我的代码中取的是样本点。其中k是带宽参数，控制w（钟形函数）的宽窄程度，类似于高斯函数的标准差。
            算法思路：假设预测点取样本点中的第i个样本点（共m个样本点），遍历1到m个样本点（含第i个），算出每一个样本点与预测点的距离，
            也就可以计算出每个样本贡献误差的权值，可以看出w是一个有m个元素的向量（写成对角阵形式）。
    '''
    # mat() 函数是将array转换为矩阵的函数， mat().T 是转换为矩阵之后，再进行转置操作
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 获得xMat矩阵的行数
    m = shape(xMat)[0]
    # eye()返回一个对角线元素为1，其他元素为0的二维数组，创建权重矩阵weights，该矩阵为每个样本点初始化了一个权重
    weights = mat(eye((m)))
    for j in range(m):
        # testPoint 的形式是 一个行向量的形式
        # 计算 testPoint 与输入样本点之间的距离，然后下面计算出每个样本贡献误差的权值
        diffMat = testPoint - xMat[j, :]
        # k控制衰减的速度
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    # 根据矩阵乘法计算 xTx ，其中的 weights 矩阵是样本点对应的权重矩阵
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 计算出回归系数的一个估计
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''
        Description：
            测试局部加权线性回归，对数据集中每个点调用 lwlr() 函数
        Args：
            testArr：测试所用的所有样本点
            xArr：样本的特征数据，即 feature
            yArr：每个样本对应的类别标签，即目标变量
            k：控制核函数的衰减速率
        Returns：
            yHat：预测点的估计值
    '''
    # 得到样本点的总数
    m = shape(testArr)[0]
    # 构建一个全部都是 0 的 1 * m 的矩阵
    yHat = zeros(m)
    # 循环所有的数据点，并将lwlr运用于所有的数据点
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    # 返回估计值
    return yHat


if __name__ == '__main__':
    # xArr, yArr = loadDataSet('./data.txt')
    x = np.arange(0, 1.0, step=0.005)
    y = 3.0 + 1.7 * x + 0.1 * np.sin(60 * x) + 0.02 * np.random.normal(0.0, 1.0, len(x))
    x = x.reshape((-1, 1))
    y = y.reshape(-1)
    ones = np.ones_like(x)
    x = np.concatenate([ones, x], 1)
    xArr = x
    yArr = y
    print(np.shape(xArr))
    print(np.shape(yArr))

    flag = False
    if flag:
        xMat = mat(xArr)
        yMat = mat(yArr)
        ws = standRegres(xArr, yArr)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter([xMat[:, 1].flatten()], [yMat.T[:, 0].flatten().A[0]], s=20, c='r')
        xCopy = xMat.copy()
        xCopy.sort(0)
        yHat = xCopy * ws
        ax.plot(xCopy[:, 1], yHat, linewidth=3)
        plt.show()
    else:
        yHat = lwlrTest(xArr, xArr, yArr, 0.01)
        xMat = mat(xArr)
        srtInd = xMat[:, 1].argsort(0)
        xSort = xMat[srtInd][:, 0, :]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xSort[:, 1], yHat[srtInd], linewidth=3)
        ax.scatter(
            [xMat[:, 1].flatten().A[0]], [mat(yArr).T.flatten().A[0]],
            s=20,
            c='red')
        plt.show()
