# -- encoding:utf-8 --
"""
@File : 01_标签提取
@Author: Octal_H
@Date : 2019/10/30
@Desc : 直接把所有的邮件信息当做一个整体，然后做分词，然后做TFIDF，然后做降维，最后做模型的方式
"""
import os
import numpy as np

def read_index_file(file_path):
    # 垃圾邮件spam用1标注，正常邮件ham用0标注
    type_dict = {"spam": "1", "ham": "0"}
    index_file = open(file_path)
    index_dict = {}
    try:
        for line in index_file:
            arr = line.strip().split(" ")
            if len(arr) == 2:
                key, value = arr
            # 添加字段到字典中
            value = value.strip().replace("../data", "").replace("\n", "")
            index_dict[value] = type_dict[key.lower()]
    finally:
        index_file.close()
    return index_dict

index_file_path = r"..\data\full\index"
index_dict = read_index_file(index_file_path)
print(index_dict)

# 2. 分别提取训练数据的标签和测试数据的标签
file_path = r"..\data\data"
dir_names = os.listdir(file_path)
print("总文件夹数目:{}".format(len(dir_names)))
train_labels = []
test_labels = []
for dir_name in dir_names:
    flag = int(dir_name)
    print("开始处理文件夹{}中的内容!!!".format(dir_name))
    dir_file_path = os.path.join(file_path, dir_name)
    file_names = os.listdir(dir_file_path)
    for file_name in file_names:
        key = "/{}/{}".format(dir_name, file_name)
        if key in index_dict:
            value = index_dict[key]
            if flag < 200:
                train_labels.append(value)
            else:
                test_labels.append(value)


np.save('./data/traim_email_labels.npy', train_labels)
np.save('./data/test_email_labels.npy', test_labels)

