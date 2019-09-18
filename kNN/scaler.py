"""
 Create by zipee on 2019/9/16.
"""
__author__ = 'zipee'

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

# 数据集均值方差归一化
standardScaler = StandardScaler()
standardScaler.fit(X_train, y_train)
# 打印均值，方差
# 测试与训练数据集使用相同的均值方差
print(standardScaler.mean_, standardScaler.scale_)
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test)

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
print(knn_clf.score(X_test, y_test))
