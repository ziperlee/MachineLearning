"""
 Create by zipee on 2019/9/23.
"""
__author__ = "zipee"

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor


boston = datasets.load_boston()
X = boston.data
y = boston.target

X = X[y < 50]
y = y[y < 50]

# 分割训练、测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 数据归一化
standardScaler = StandardScaler()
standardScaler.fit(X_train, y_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)

# 使用随机梯度下降法解决线性回归问题
# n_iter代表需要全量过几遍数据
sgd_reg = SGDRegressor(n_iter=100)
sgd_reg.fit(X_train_standard, y_train)
score = sgd_reg.score(X_test_standard, y_test)
print(score)
