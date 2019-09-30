"""
 Create by zipee on 2019/9/28.
"""
__author__ = "zipee"

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# 查看fetch_mldata下载路径
# from sklearn.datasets.base import get_data_home
# print (get_data_home())
# 数据存放目录~/scikit_learn_data/mldata

mnist = fetch_mldata("MNIST original")

X, y = mnist["data"], mnist["target"]
# 前60000行为训练数据
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
# 60000行后为测试数据
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)

knn_clf = KNeighborsClassifier()
# 样本数据出于同一尺度下，无需归一化处理
knn_clf.fit(X_train, y_train)
# score = knn_clf.score(X_test, y_test)
# print(f'score = {score}')

# PCA降维处理数据
pca = PCA(0.90)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
pca_score = knn_clf.score(X_test_reduction, y_test)
print(f"pca_score = {pca_score}")
