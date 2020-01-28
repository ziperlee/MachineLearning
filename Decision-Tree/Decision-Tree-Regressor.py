"""
 Created by liwei on 2020/1/27.
"""
# 决策树解决回归问题

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)

# 训练数据远优于测试数据，显然过拟合
dt_reg.score(X_test, y_test)
dt_reg.score(X_train, y_train)