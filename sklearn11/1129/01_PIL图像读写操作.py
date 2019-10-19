# -- encoding:utf-8 --
"""
@File : 01_PIL图像读写操作
@Author: Octal_H
@Date : 2019/10/19
@Desc : 
"""
from PIL import Image

# 读取图像
img = Image.open('a.bmp')

# 图像可视化
img.show()

# 图像保存
img.save('a.png')
print('Down')