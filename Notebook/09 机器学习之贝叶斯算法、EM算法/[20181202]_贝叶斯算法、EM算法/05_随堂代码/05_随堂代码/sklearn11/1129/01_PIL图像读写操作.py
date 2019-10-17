# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/29
"""

from PIL import Image

# 读取图像
img = Image.open('a.bmp')

# 图像可视化
img.show()

# 图像保存
img.save('a.png')

print("Done!!!")
