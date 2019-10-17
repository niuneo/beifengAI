# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/15
"""

import os
import jieba

files = os.listdir('./datas')
length = len(files)
with open('./doc_cut.txt', 'w', encoding='utf-8') as writer:
    for idx, file in enumerate(files):
        file_path = os.path.join('./datas', file)
        with open(file_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                new_line = ' '.join(jieba.lcut(line)).strip()
                if len(new_line) > 0:
                    writer.write(' {}'.format(new_line))
        if idx < length - 1:
            writer.writelines(' \n ')
