# -- encoding:utf-8 --
"""
@File : 04_基于PIL的噪音数据过滤以及Filter API作用展示
@Author: Octal_H
@Date : 2019/10/19
@Desc : 
"""
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance


def add_salt_noise(img):
    """
    添加噪音
    :param img:
    :return:
    """
    rows, cols = img.shape
    Grey_sp = img
    snr = 0.9
    noise_num = int((1 - snr) * rows * cols)

    for i in range(noise_num):
        rand_x = random.randint(0, rows - 1)
        rand_y = random.randint(0, cols - 1)
        if random.randint(0, 1) == 1:
            Grey_sp[rand_x, rand_y] = 255
    return Grey_sp


im = Image.open('a.bmp')
im = Image.fromarray(add_salt_noise(np.array(im.convert("L"))), "L")
# im.show()

# 使用中值过滤来过滤噪音
im = im.filter(ImageFilter.MedianFilter)

# 提取边缘特征
im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
im = im.filter(ImageFilter.EMBOSS)
im = im.filter(ImageFilter.SMOOTH_MORE)
im.show()