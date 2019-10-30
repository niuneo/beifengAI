# -- encoding:utf-8 --
"""
@File : 04_模型应用
@Author: Octal_H
@Date : 2019/10/30
@Desc : 
"""

import jieba
from sklearn.externals import joblib


class EmailFilter(object):
    def __init__(self, model_file_paths):
        """
        根据给定的这样一个模型从存储磁盘路径加载
        :param model_file_paths: 给定的一个list集合类型的文件夹路径
        """
        total_stage = 0
        stages = []
        for idx, file_path in enumerate(model_file_paths):
            total_stage += 1
            state_algo = joblib.load(file_path)
            stages.append((idx, hasattr(state_algo, 'predict'), state_algo))

        self.stages = stages
        self.total_stage = total_stage

    def predict(self, X):
        y_pred = X
        for idx, flag, stage_algo in self.stages:
            if flag:
                y_pred = stage_algo.predict(y_pred)
            else:
                y_pred = stage_algo.transform(y_pred)

        return y_pred

    def predict_with_file_path(self, file_path):
        """
        基于传入的邮件的路径信息做一个模型预测
        :param file_path:
        :return:
        """
        return self.predict(self.fetch_email_features(file_path))

    def fetch_email_features(self, file_path):
        """
        根据邮件的地址信息获取邮件的特征属性
        :param file_path:
        :return:
        """
        X = []
        with open(file_path, encoding='gb2312', errors='ignore') as reader:
            content = reader.read()
            content = ' '.join(filter(lambda word: len(word.strip()) > 0, jieba.cut(content)))
            X.append(content)
        return X


if __name__ == '__main__':
    email_filter = EmailFilter(['./model/tfidf.pkl', './model/svd.pkl', './model/algo.pkl'])
    # 1. 获取传入的参数
    file_path = r"..\data\data\040\032"

    # 2. 获取预测结果
    pred = email_filter.predict_with_file_path(file_path)

    # 3. 预测结果返回
    print("预测结果为:{}".format(pred))
