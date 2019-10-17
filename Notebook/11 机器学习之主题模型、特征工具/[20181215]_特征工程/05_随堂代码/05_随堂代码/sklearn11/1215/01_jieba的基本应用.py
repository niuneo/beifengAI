# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/15
"""

import jieba
import jieba.posseg as pseg

word_str = "因为在我们对文本数据进行理解的时候，机器不可能完整的对文本进行处理，只能把每个单词作为特征属性来进行处理，所以在所有的文本处理中，第一步就是分词操作,中共中央总书记"
# word_str = "梁国杨氏子九岁，甚聪惠。孔君平诣其父，父不在，乃呼儿出。为设果，果有杨梅。孔指以示儿曰：“此是君家果。”儿应声答曰：“未闻孔雀是夫子家禽。”"

# 一、基本应用
"""
def cut(self, sentence, cut_all=False, HMM=True)
    功能：对sentence这个字符串进行分词，返回一个迭代器类型
    sentence: 指定要分割的字符串对象
    cut_all: 是否采用全分割模型，默认不采用
    HMM: 是否进行隐马尔科夫模型发现新词
lcut: 直接将分词的结果转换为list集合
"""
seg_a = jieba.cut(word_str, HMM=True)
print(type(seg_a))
print(seg_a)
print(list(seg_a))
print(list(seg_a))

seg_b = jieba.lcut(word_str)
print(seg_b)

seg_c = jieba.cut_for_search(word_str)
print(list(seg_c))

# 二、提取单词的词性
words = pseg.cut(word_str)
print(list(words))

# 三、自定义词典信息
word_str = "梁国杨氏子九岁，甚聪惠"
seg_a = jieba.cut(word_str)
print(list(seg_a))

# 方式一：直接在代码中临时加入分词的单词
# """
# def add_word(self, word, freq=None, tag=None):
#     功能：在当前代码运行环境中，临时加入词语信息
#       word: 单词
#       freq: 词频，词频越高，该单词出现的可能性越高, 参数可选
#       tag：词性，eg：名词(n)、动词(v)...., 参数可选
# """
# jieba.add_word("聪惠")
# jieba.add_word("杨氏", freq=1000)
# jieba.add_word("子九岁")
# jieba.del_word('子九岁')
# seg_a = jieba.cut(word_str)
# print(list(seg_a))

# 方式二：自定义一个词典文件
jieba.load_userdict('./mydict.txt')
seg_a = jieba.cut(word_str)
print(list(seg_a))
