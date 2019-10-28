# -- encoding:utf-8 --
"""
@File : test.py
@Author: Octal_H
@Date : 2019/10/28
@Desc :
"""
import numpy as np
from PIL import Image, ImageFilter

img = Image.open('c.png')

img = img.resize((128, 128))
# 3. 将图像转换为灰度图像
img = img.convert("L")
# 4. 对图像做一个过滤
img = img.filter(ImageFilter.CONTOUR)
img = img.filter(ImageFilter.SMOOTH)
img = img.filter(ImageFilter.MedianFilter)

rotate_p = 90
img2 = img.copy()
img2 = img2.rotate(rotate_p, expand=True, fillcolor=(255, 255, 255))
img2.show()
