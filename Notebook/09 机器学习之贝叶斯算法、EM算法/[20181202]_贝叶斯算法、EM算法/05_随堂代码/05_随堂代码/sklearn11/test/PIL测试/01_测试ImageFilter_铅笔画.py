# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/29
"""

from PIL import Image, ImageFilter, ImageOps

if __name__ == '__main__':
    blur = 25
    alpha = 1.0
    # 1. 加载图片
    img = Image.open('111.jpg')

    # 2. 图片转换为灰度图片
    img1 = img.convert("L")

    # 3. 图片copy
    img2 = img1.copy()

    # 4. 反转图像（255-->0, 0-->255）
    # img2 = ImageOps.invert(img2)
    img2 = img2.point(lambda i: 255 - i)

    # 5. 模糊化
    for i in range(blur):
        img2 = img2.filter(ImageFilter.BLUR)

    width, height = img1.size
    for x in range(width):
        for y in range(height):
            a = img1.getpixel((x, y))
            b = img2.getpixel((x, y))
            c = min(int(a * 255 / (256 - b * alpha)), 255)
            img1.putpixel((x, y), c)
    img1.show()
