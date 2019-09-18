"""
 Create by zipee on 2019/9/16.
"""
__author__ = 'zipee'

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

# 超参数数据集
param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]
    }
]

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

knn_clf = KNeighborsClassifier()

# n_jobs 多进程运行
# verbose 运行信息输出级别
grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 最佳分类器
print(grid_search.best_estimator_)
# 最佳结果
print(grid_search.best_score_)