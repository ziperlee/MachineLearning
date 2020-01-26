"""
 Created by liwei on 2020/1/26.
"""

# 实现混淆矩阵，精准率和召回率

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target==9] = 1
y[digits.target!=9] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)

y_log_predict = log_reg.predict(X_test)

# 混淆矩阵
confusion_matrix(y_test, y_log_predict)

# 精准率
precision_score(y_test, y_log_predict)

# 召回率
recall_score(y_test, y_log_predict)