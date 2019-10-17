# -- encoding:utf-8 --
"""
可以考虑将图像的Image对象转换为numpy的数组，然后对数组中的值做处理，这个值其实就是像素点的值
Create by ibf on 2018/11/29
"""

import numpy as np
from PIL import Image

img = Image.open('a.png')
img_arr = np.array(img)
print(img_arr.shape)
img_arr = img_arr[206:306, 206:306, :]
img2 = Image.fromarray(img_arr, 'RGB')
img2.show()
