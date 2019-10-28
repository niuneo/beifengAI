# -- encoding:utf-8 --
"""
@File : 02_案例代码：手写中文汉字识别代码实现02
@Author: Octal_H
@Date : 2019/10/28
@Desc : 
"""
import os
import glob
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(28)

_new_img_width = 128
_new_img_height = 128


def read_image(img_file_path, flag=True, size=10):
    """
    基于给定的文件路径，读取图像数据，并转换为numpy的数组形式返回
    :param img_file_path:
    :param flag: 是否进行图片数据增加的操作，True表示增加
    :param size: 对图像增加多少张
    :return:
    """
    # 1. 加载数据
    img1 = Image.open(img_file_path)
    img = img1.copy()
    # 2. 对图片的大小做一个重置操作
    img = img.resize((_new_img_width, _new_img_height))
    # 3. 将图像转换为灰度图像
    img = img.convert("L")
    # 4. 对图像做一个过滤
    img = img.filter(ImageFilter.CONTOUR)
    img = img.filter(ImageFilter.SMOOTH)
    img = img.filter(ImageFilter.MedianFilter)
    # 4. 将图像转换为数组并返回
    imgs = np.array(img).reshape((1, -1))
    # 5. 进行数据增强
    if flag:
        for i in range(size):
            img2 = img1.copy()
            # a. 对图像做一个旋转的操作
            rotate_p = 20 * np.random.random() - 10
            img2 = img2.rotate(rotate_p, expand=True, fillcolor=(255, 255, 255))
            # b. 做一个数据增强的操作
            contrast_factor = 9 * np.random.random() + 0.5
            enhance = ImageEnhance.Contrast(img2)
            img2 = enhance.enhance(factor=contrast_factor)
            # c. 对图片的大小做一个重置操作
            img2 = img2.resize((_new_img_width, _new_img_height))
            # d. 将图像转换为灰度图像
            img2 = img2.convert("L")
            # e. 对图像做一个过滤
            img2 = img2.filter(ImageFilter.CONTOUR)
            img2 = img2.filter(ImageFilter.SMOOTH)
            img2 = img2.filter(ImageFilter.MedianFilter)
            # 4. 将图像转换为数组并返回
            img2 = np.array(img2).reshape((1, -1))
            # 5. 保存
            imgs = np.append(imgs, img2, axis=0)
    return imgs


def read_images(parent_file_path, flag=True, size=10):
    """
    从给定的文件夹中读取图像数据以及对应的标签数据， 要求给定的parent_file_path下面必须是文件夹，文件夹下必须是图片，文件夹名称作为标签数据
    :param parent_file_path: 给定的是一个文件夹路径。eg: “./中文字符识别/训练数据”或者“./中文字符识别/验证数据”
    :return:
    """
    X = None
    Y = []
    count = 0
    child_list1 = os.listdir(parent_file_path)
    for child_name in child_list1:
        # 将文件夹名称转换为标签的值
        label = int(child_name)
        # 构建该子文件夹的路径
        dir_file_path = os.path.join(parent_file_path, child_name)
        print("处理文件夹‘{}’中的数据!!!".format(dir_file_path))
        # 获取该子文件夹中png格式的数据文件路径列表
        img_file_paths = glob.glob(os.path.join(dir_file_path, '*png'))
        # 遍历所有的文件读取数据形成最终的特征属性X和目标属性Y
        for img_file_path in img_file_paths:
            count += 1
            # 1. 读取图片的特征属性x
            imgs = read_image(img_file_path, flag, size)
            for x in imgs:
                x = x.reshape((1, -1))
                # 2. 将特征属性x和目标属性label添加到X和Y中
                if X is None:
                    X = x
                else:
                    X = np.append(X, x, axis=0)
                Y.append(label)
    Y = np.array(Y).reshape(-1)
    print("总样本数目:{}".format(count))
    print("特征属性矩阵X形状:{}".format(X.shape))
    print("目标属性矩阵Y形状:{}".format(Y.shape))
    return X, Y


if __name__ == '__main__':
    # TODO: 作业 -> 下面这个代码只有模型训练，没有预测的过程，自己加如模型预测的代码，并且让代码支持通过python的命令行执行，并且所有的参数支持命令行参数输入
    is_train = True
    if is_train:
        # 1. 加载数据
        x, y = read_images('./中文字符识别/训练数据', flag=True, size=1)
        x_valid, y_valid = read_images('./中文字符识别/验证数据', flag=False)

        # 2. 数据的划分
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)
        print("训练数据格式:{}".format(np.shape(x_train)))
        print("测试数据格式:{}".format(np.shape(x_test)))

        # 3. 特征工程
        # TODO: 大家自己考虑一下，把下面的代码换成管道流的实现方式。
        # 降维：因为图片中使用像素点作为特征属性，值比较大
        pca = PCA(n_components=int(min(25, 0.9 * x_train.shape[1])), random_state=28)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
        # 标准化：防止图片的像素点的取值对于模型的影响
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # 3. 模型构建
        print("用于SVM模型训练的数据形状:{}".format(x_train.shape))
        algo = SVC(kernel='rbf', C=2.0, gamma=0.1, random_state=28)
        algo.fit(x_train, y_train)

        # 4. 模型效果查看
        y_train_pred = algo.predict(x_train)
        y_test_pred = algo.predict(x_test)
        y_valid_pred = algo.predict(scaler.transform(pca.transform(x_valid)))
        print("训练数据上的准确率:{}".format(accuracy_score(y_train, y_train_pred)))
        print("测试数据上的准确率:{}".format(accuracy_score(y_test, y_test_pred)))
        print("训练数据上的混淆矩阵:\n{}".format(confusion_matrix(y_train, y_train_pred)))
        print("测试数据上的混淆矩阵:\n{}".format(confusion_matrix(y_test, y_test_pred)))
        print("验证数据上的准确率:{}".format(accuracy_score(y_valid, y_valid_pred)))
        print("验证数据上的混淆矩阵:\n{}".format(confusion_matrix(y_valid, y_valid_pred)))

        # 5. 模型持久化
        joblib.dump(pca, './model/pca.pkl')
        joblib.dump(scaler, './model/scaler.pkl')
        joblib.dump(algo, './model/svm.pkl')
        print("Done!!!")
