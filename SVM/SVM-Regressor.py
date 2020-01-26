"""
 Created by liwei on 2020/1/26.
"""
# svm解决回归问题

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# 多项式回归
from sklearn.svm import LinearSVR
# 同SVC，支持指定核函数
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def StandardLinearSVR(epsilon=0.1):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('linearSVR', LinearSVR(epsilon=epsilon))
    ])

svr = StandardLinearSVR()
svr.fit(X_train, y_train)

svr.score(X_test, y_test)