# -- encoding:utf-8 --
"""
Create by ibf on 2018/12/15
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 1. 加载文本数据
data = []
stop_words = ['这个', '就是']
with open('doc_cut.txt', 'r', encoding='utf-8') as reader:
    for line in reader:
        data.append(line)
print("原始数据:")
print(data)
print("\n")

# 1. 词袋法
"""
def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64)
  input: 指定数据的输入方式，content表示集合方式，file表示文件句柄的方式，filename表示文件路径的方式
  encoding: 给定字符编码格式
  lowercase: 是否将数据转换为小写
  stop_words: 给定停止词列表，默认为空
  token_pattern: 给定文本中的单词所需要满足的正则字符串
  max_features：给定最多的允许特征属性数目，默认不限制
  binary: 给定模型是BOW还是SOW，默认为False，表示BOW
"""
count = CountVectorizer(stop_words=stop_words, binary=False)
data1 = count.fit_transform(data)
print("特征属性单词:")
print(count.get_feature_names())
print("停止词:")
print(count.get_stop_words())
print("恢复数据:")
print(count.inverse_transform(data1))
print(data1.toarray())

# 2. HashTF
"""
def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word', n_features=(2 ** 20),
                 binary=False, norm='l2', alternate_sign=True,
                 non_negative=False, dtype=np.float64)
    n_features: 允许的最多的特征属性数目，一般需要设置
    non_negative: 是否不允许有负值出现，默认False表示允许出现
"""
hash = HashingVectorizer(n_features=100, stop_words=stop_words, non_negative=True)
print("停止词:")
print(hash.get_stop_words())
print("转换结果:")
print(hash.transform(data).toarray())

# 3. TFIDF
tfidf = TfidfVectorizer()
data2 = tfidf.fit_transform(data)
print(data2)
