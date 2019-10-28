# -- encoding:utf-8 --
"""
@File : hello_python.py
@Author: Octal_H
@Date : 2019/10/28
@Desc :
"""
import sys

if __name__ == '__main__':
    print("总的参数数目:{}".format(len(sys.argv)))
    for i, v in enumerate(sys.argv):
        print("第{}个参数:{}".format(i + 1, v))
