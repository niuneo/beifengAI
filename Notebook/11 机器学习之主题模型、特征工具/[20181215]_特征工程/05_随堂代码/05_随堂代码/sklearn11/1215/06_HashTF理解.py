# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/15
"""

import numpy as np

# 1. 统计一下各个文档中各个单词的Hash值出现次数作为特征值
features = []
max_feature_num = 200
stop_words = ['这个', '就是']
with open('doc_cut.txt', 'r', encoding='utf-8') as reader:
    for line in reader:
        # a. 计算当前文本在各个特征属性单词上出现的次数
        result = {}
        total_word = 0
        for word in line.split(" "):
            if len(word) > 1:
                if word not in stop_words:
                    total_word += 1
                    word_key = hash(word) % max_feature_num
                    if word_key not in result:
                        result[word_key] = 1
                    else:
                        result[word_key] += 1

        # b. 构建当前文本对应的最终词向量
        feature = []
        for feature_name in range(max_feature_num):
            if feature_name not in result:
                feature.append(0)
            else:
                feature.append(result[feature_name])
                # feature.append(result[feature_name] / total_word)
        features.append(feature)
print("最终的特征向量为:")
print(features)
