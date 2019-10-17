# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/29
"""

from PIL import Image, ImageFilter, ImageOps

if __name__ == '__main__':
    blur = 2
    # 1. 加载图片
    img = Image.open('111.jpg')

    # 2. 图片转换为灰度图片
    img1 = img.convert("L")

    # 5. 模糊化
    for i in range(blur):
        img1 = img1.filter(ImageFilter.EMBOSS)
    for i in range(blur + 10):
        img1 = img1.filter(ImageFilter.SMOOTH)
    for i in range(blur):
        img1 = img1.filter(ImageFilter.DETAIL)
    for i in range(blur):
        img1 = img1.filter(ImageFilter.SHARPEN)

    img1.show()
