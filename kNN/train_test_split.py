"""
 Create by zipee on 2019/9/16.
"""
__author__ = 'zipee'

# 分割数据集为训练数据集与测试数据集

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

X = iris.data
y = iris.target

# 索引乱序
# shuffled_indexes = np.random.permutation(len(X))
# random_state 随机数种子，可用于测试场景复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #, random_state=666
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# 创建knn分类器
knn_clf = KNeighborsClassifier(3)
knn_clf.fit(X_train, y_train)

# 预测
y_predict = knn_clf.predict(X_test)

# 准确率
score = sum(y_predict == y_test) / len(y_test)
print(score)