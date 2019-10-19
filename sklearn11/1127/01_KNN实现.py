# -- encoding:utf-8 --
"""
@File : KNN实现
@Author: Octal_H
@Date : 2019/10/19
@Desc : 
"""
import numpy as np


class KNN(object):
    def __init__(self, k=10):
        self.k = k
        self.train = None

    def fit(self, train):
        self.train = train

    def calc_dist(self, data, x):
        return np.sqrt(np.sum((data[:-1] - x) ** 2))

    def fetch_k_neighbors(self, datas, x, k):
        k_neighbors = []
        count = 0
        max_dist = -1
        max_index = -1
        for data in datas:
            dist = self.calc_dist(data, x)
            if count < k:
                k_neighbors.append((data, dist))
                if dist > max_dist:
                    max_dist = dist
                    max_index = count
                count += 1
            elif dist < max_dist:
                k_neighbors[max_index] = (data, dist)
                max_index = np.argmax(list(map(lambda t: t[1], k_neighbors)))
                max_dist = k_neighbors[max_index][1]
        return k_neighbors

    def predict(self, X):
        result = []
        for x in X:
            label = self._predict(x)
            result.append(label)
        return np.asarray(result).reshape((-1, 1))

    def _predict(self, x):
        neighbors = self.fetch_k_neighbors(self.train, x, self.k)

        result = {}
        for record in neighbors:
            # i. 获取当前样本的目标属性y值, 假定最后一列就是y
            target_y = record[0][-1]
            # ii. 如果该值不在result列表中，那么进行添加，如果在，进行更新
            if target_y not in result:
                count = 0
            else:
                count = result.get(target_y)
            count += 1
            result[target_y] = count
        # b. 从字典数据中获取出现次数最多的那个类别
        max_count = -1
        result_type = None
        for k, v in result.items():
            if v > max_count:
                max_count = v
                result_type = k

        return result_type


if __name__ == '__main__':
    datas = np.array([
        [5, 3, 1],
        [6, 3, 1],
        [8, 3, 1],
        [2, 3, 0],
        [1, 3, 0],
        [4, 3, 0]
    ])
    knn = KNN(k=4)
    # k_neighbors = knn.fetch_k_neighbors(datas, x=np.array([0, 3]), k=3)
    # print(k_neighbors)
    knn.fit(datas)
    print(knn.predict(X=np.array([[0, 3], [9, 3.5]])))
