# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/15
"""

import numpy as np

# 1. 相当于从原始的数据中获取有哪些单词可以做特征属性
data = []
stop_words = ['这个', '就是']
with open('doc_cut.txt', 'r', encoding='utf-8') as reader:
    for line in reader:
        for word in line.split(" "):
            if len(word) > 1:
                if word not in stop_words:
                    data.append(word)

feature_names = list(set(data))
feature_names.sort()
print("原始数据:\n{}".format(data))
print("特征属性:\n{}".format(feature_names))

# 2. 统计一下各个文档中各个特征属性单词是否出现作为特征值
features = []
with open('doc_cut.txt', 'r', encoding='utf-8') as reader:
    for line in reader:
        # a. 计算当前文本在各个特征属性单词上出现的次数
        result = {}
        for word in line.split(" "):
            if word in feature_names:
                if word not in result:
                    result[word] = 1
                else:
                    result[word] += 1
        # b. 构建当前文本对应的最终词向量
        feature = []
        for feature_name in feature_names:
            if feature_name not in result:
                feature.append(0)
            else:
                feature.append(1)
        features.append(feature)
print("最终的特征向量为:")
print(features)
