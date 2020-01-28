"""
 Created by liwei on 2020/1/28.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# scikit-learn中bagging和pasting统一叫BaggingClassifier，两者通过bootstrap区分
bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                           n_estimators=500, max_samples=100,
                           bootstrap=True)
bagging_clf.fit(X_train, y_train)
bagging_clf.score(X_test, y_test)