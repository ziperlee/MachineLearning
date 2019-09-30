"""
 Create by zipee on 2019/9/28.
"""
__author__ = "zipee"

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# PCA 算法用以做数据的降维处理

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# 降维前
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
before_PCA_score = knn_clf.score(X_test, y_test)
print(f"before_PCA_score = {before_PCA_score}")

# 降维后
# 指定最终维度
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
knn_clf.fit(X_train_reduction, y_train)
after_PCA_score = knn_clf.score(X_test_reduction, y_test)
print(f"after_PCA_score = {after_PCA_score}")
# 绘制二维关系图，10个数字
pca.fit(X)
X_reduction = pca.transform(X)
for i in range(10):
    plt.scatter(X_reduction[y == i, 0], X_reduction[y == i, 1], alpha=0.8)
plt.show()

# 指定成分解释方差比例
# 完整成分解释方差
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
print(f"完整成分解释方差 explained_variance_ = {pca.explained_variance_ratio_}")
# 指定比例
pca = PCA(0.95)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
print(f"自动降维 n = {X_train_reduction.shape[1]}")
knn_clf.fit(X_train_reduction, y_train)
auto_PCA_score = knn_clf.score(X_test_reduction, y_test)
print(f"auto_PCA_score = {auto_PCA_score}")

# 绘制成分解释方差同维度的关系图
plt.plot(
    [i for i in range(X_train.shape[1])],
    [np.sum(pca.explained_variance_ratio_[: i + 1]) for i in range(X_train.shape[1])],
)
plt.show()
