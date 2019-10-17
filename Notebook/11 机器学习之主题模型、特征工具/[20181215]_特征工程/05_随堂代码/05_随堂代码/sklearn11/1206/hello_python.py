# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/6
"""

import sys

if __name__ == '__main__':
    print("总的参数数目:{}".format(len(sys.argv)))
    for i, v in enumerate(sys.argv):
        print("第{}个参数:{}".format(i + 1, v))
