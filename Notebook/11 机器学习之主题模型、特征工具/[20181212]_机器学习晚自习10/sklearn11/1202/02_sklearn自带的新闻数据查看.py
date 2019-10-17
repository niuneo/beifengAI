# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/2
"""

from sklearn import datasets

# 加载数据
newsgroups = datasets.fetch_20newsgroups(data_home='../datas', subset='train', categories=None,
                                         remove=("headers", "footers", "quotes"))
print(type(newsgroups))
print("查看newsgroups这个对象的属性:")
for key in newsgroups:
    print(key)
print(newsgroups.target_names)
print(newsgroups.data[0])
print(newsgroups.target[0])
