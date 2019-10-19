# -- encoding:utf-8 --
"""
@File : 03_PIL图像的基本操作二
@Author: Octal_H
@Date : 2019/10/19
@Desc : 直接使用PIL的API对数据进行处理
"""

from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# 读取图像
img = Image.open('c.png')
img.show()

# 查看图像的基本信息
# format: 获取图像的格式是什么，eg：png、bmp....
# size: 获取图像的大小，也就是高度和宽度，返回为值: (宽度，高度)
# mode：返回图像是什么类型的图像，eg：RGB就是彩色图像、L就是灰度图像...https://pillow.readthedocs.io/en/5.2.x/handbook/concepts.html#modes
print((img.format, img.size, img.mode))

# 1. 图像转换为灰度图像
img1 = img.convert("L")
# img1.show()

# 2. 将灰度图像转换为黑白图像，也就是二值化图像，也就是像素点的值大于某个值的时候，给定为白色，小于某个值的时候给定为黑色
img2 = img1.point(lambda i: 255 if i > 252 else 0)
# img2.show()

# 3. 反转图像，一般只对灰度图像做（255->0, 0->255, 128->127）
img3 = img1.point(lambda i: 255 - i)
# img3.show()

# 4. 大小缩放
img4 = img.resize((1024, 700))
# img4.show()
img5 = img.resize((32, 32))
# img5.show()

# 5. 图像的旋转
# 30度表示逆时针旋转30度，如果给定负值，表示顺时针旋转
# expand: 表示旋转之后的图像大小是否发生变化，设置为True，表示变化，并且对于多余的位置使用fillcolor给定的颜色填充，默认为False，表示截断
img6 = img.rotate(30, expand=True, fillcolor=(255, 255, 255))
# img6.show()

# 6. 转置(左右内容或者上下的内容调换)
img7 = img.transpose(Image.FLIP_TOP_BOTTOM)
# img7 = img.transpose(Image.FLIP_LEFT_RIGHT)
# img7.show()

# 7. 剪切
# box：(left, upper, right, lower)也就是一个矩行的左上角和右下角的像素点的坐标
box = (200, 80, 380, 225)
img8 = img.crop(box)
# img8.show()

# 8. 图像的分裂和组合
r, g, b = img8.split()
img9 = Image.merge('RGB', (b, g, r))
# img9.show()

# 9. 粘贴
img10 = img.copy()
img10.paste(img9, box)
# img10.show()

# 10. 数据增强的操作
# 方式一：使用point这个API对像素点的值进行操作
img11 = img.point(lambda i: i * 1.5)
# img11.show()

# 方式二：直接分裂像素值，然后分别对不同通道的像素点进行处理
r, g, b = img.split()
r = r.point(lambda i: i * 1.2)
g = g.point(lambda i: i * 1.0 if i == 255 else i * 0.5)
img12 = Image.merge(img.mode, (r, g, b))
# img12.show()

# 方式三：直接使用PIL中的API做数据增强
# https://pillow.readthedocs.io/en/5.2.x/reference/ImageEnhance.html
# 平衡度、亮度、对比度、清晰度....
enhance = ImageEnhance.Contrast(img)
img13 = enhance.enhance(factor=1.5)
img13.show()



