"""
 Create by zipee on 2019/9/16.
"""
__author__ = 'zipee'

import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 手写数字训练数据
digits = datasets.load_digits()
digits.keys()

X = digits.data
y = digits.target
print(X.shape, y.shape)

# 绘制数字
some_digit = X[666]
some_digit_image = some_digit.reshape(8, 8)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
plt.show()

# 分割测试，训练数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
y_predict = knn_clf.predict(X_test)

score1 = accuracy_score(y_test, y_predict)
print(score1)

score2 = knn_clf.score(X_test, y_test)
print(score2)
