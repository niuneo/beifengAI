# -- encoding:utf-8 --
"""
@File : 02_数据处理
@Author: Octal_H
@Date : 2019/10/30
@Desc : 
"""
import jieba
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib

# 1. 读取数据形成DataFrame
file_path = r"..\data\data"
dir_names = os.listdir(file_path)
print("总文件夹数目:{}".format(len(dir_names)))
train_datas = []
test_datas = []
for dir_name in dir_names:
    flag = int(dir_name)
    print("开始处理文件夹{}中的内容!!!".format(dir_name))
    dir_file_path = os.path.join(file_path, dir_name)
    file_names = os.listdir(dir_file_path)
    for file_name in file_names:
        data_file_path = os.path.join(dir_file_path, file_name)
        with open(data_file_path, encoding='gb2312', errors='ignore') as reader:
            content = reader.read()
            content = ' '.join(filter(lambda word: len(word.strip()) > 0, jieba.cut(content)))
            if flag < 200:
                train_datas.append(content)
            else:
                test_datas.append(content)

# 2. 对文本数据转换为具体的特征向量
tfidf = TfidfVectorizer(norm='l2', use_idf=True)
# 降维到20维
svd = TruncatedSVD(n_components=20)
train_datas = tfidf.fit_transform(train_datas)
train_datas = svd.fit_transform(train_datas)
test_datas = svd.transform(tfidf.transform(test_datas))

# 3. 保存数据
np.save('./data/traim_email_datas.npy', train_datas)
np.save('./data/test_email_datas.npy', test_datas)
joblib.dump(tfidf, './model/tfidf.pkl')
joblib.dump(svd, './model/svd.pkl')
print("Done!!!")