"""
 Created by liwei on 2020/1/28.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression()

from sklearn.svm import SVC
svm_clf = SVC()

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=666)

# 集成学习
# hard voting
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC()),
    ('dt_clf', DecisionTreeClassifier(random_state=666))],
                             voting='hard')

voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)

# 使用 Soft Voting Classifier
# 需要分类器支持概率计算
voting_clf2 = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC(probability=True)),
    ('dt_clf', DecisionTreeClassifier(random_state=666))],
                             voting='soft')
voting_clf2.fit(X_train, y_train)
voting_clf2.score(X_test, y_test)