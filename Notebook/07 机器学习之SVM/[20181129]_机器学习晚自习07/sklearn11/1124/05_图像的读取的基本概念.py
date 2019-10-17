# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/24
"""

import numpy as np
from PIL import Image

# 1. 加载图像
# 计算机中，对于图像而言，使用像素点来进行描述，每个像素点就是一个颜色
# RGB: 使用红、绿、蓝三原色进行体现；灰度图像：使用一个灰度值(0~255)来体现一个像素点的颜色。
# 在机器学习/深度学习，直接将图像的像素点的值作为特征属性
img = Image.open('./gray.png')
# img.show()
print(img)

# 2. 将Image转换为数组的形式
img_arr = np.array(img)
print(img_arr.shape)
if len(img_arr.shape) == 2:
    print(img_arr[0, :10])
else:
    print(img_arr[0, :10, :])
