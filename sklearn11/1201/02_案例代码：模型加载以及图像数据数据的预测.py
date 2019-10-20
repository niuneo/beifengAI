# -- encoding:utf-8 --
"""
@File : 02_案例代码：模型加载以及图像数据数据的预测
@Author: Octal_H
@Date : 2019/10/20
@Desc : 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.externals import joblib
from PIL import Image

class FetchPredict(object):
    def __init__(self, model_path):
        self.algo = joblib.load(model_path)

    def predict(self, imgs):
        return self.algo.predict(imgs)

if __name__ == '__main__':
    # 1.将部分数据输出
    digits = datasets.load_digits()
    idxs = [0, 1, 3, 5, 8, 9]
    for idx in idxs:
        img = digits.images[idx]
        misc.imsave('digits/{}.png'.format(idx), img)

    algo = FetchPredict('./models/svm_digits.pkl')
    file_path = 'digits/0.png'
    img = Image.open(file_path)
    img = np.array(img).reshape((1, -1))
    print("当前图像的预测值为:{}".format(algo.predict(img)))

