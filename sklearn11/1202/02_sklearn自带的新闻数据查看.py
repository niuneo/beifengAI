# -- encoding:utf-8 --
"""
@File : 02_sklearn自带的新闻数据查看
@Author: Octal_H
@Date : 2019/10/29
@Desc : 
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
