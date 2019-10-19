# -- encoding:utf-8 --
"""
@File : 05_基于PIL将图片转换为铅笔素描
@Author: Octal_H
@Date : 2019/10/19
@Desc : 
"""
from PIL import Image, ImageFilter, ImageOps

# 可选掌握
if __name__ == '__main__':
    # 1. 加载图像
    img = Image.open('b.jpg')

    # 2. 图片转换为灰度图片
    img1 = img.convert("L")

    # 3. 图片copy
    img2 = img1.copy()

    # 4. 反转图像（255-->0, 0-->255）
    img2 = ImageOps.invert(img2)
    # img2 = img2.point(lambda i: 255 - i)

    # 5. 模糊化
    for i in range(25):
        img2 = img2.filter(ImageFilter.BLUR)

    # 6. 比较原始像素值和模糊化之后的像素值，进行像素点的设置
    width, height = img1.size
    for x in range(width):
        for y in range(height):
            # 获取图像给定位置的像素点的值
            a = img1.getpixel((x, y))
            b = img2.getpixel((x, y))
            c = min(int(a * 255 / (256 - b * 1.0)), 255)
            # 给给定位置设置像素点的值
            img1.putpixel((x, y), c)
    img1.show()