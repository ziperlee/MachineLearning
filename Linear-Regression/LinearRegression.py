"""
 Create by zipee on 2019/9/18.
"""
__author__ = 'zipee'

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

boston = datasets.load_boston()
x = boston.data
y = boston.target

# 剔除边缘异常数据
X = x[y < 50.0]
y = y[y < 50.0]

# 拆分训练、测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

print(X_train.shape, y_train.shape)

# 线性回归法
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg_score = lin_reg.score(X_test, y_test)
y_predict = lin_reg.predict(X_test)
print(f'lin_reg_score = {lin_reg_score}')
# 多元系数
print(f'coefficients = {lin_reg.coef_}')
# 截距
print(f'intecept = {lin_reg.intercept_}')

# MSE 均方误差
mse = mean_squared_error(y_test, y_predict)
print(f'mse = {mse}')
# RMSE 均方根误差
rmse = sqrt(mse)
print(f'rmse = {mse}')
# MAE 平均绝对误差
mae = mean_absolute_error(y_test, y_predict)
print(f'mae = {mae}')


# Knn数据归一化处理
standardScaler = StandardScaler()
standardScaler.fit(X_train, y_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)

# Knn回归法
knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train_standard, y_train)
knn_reg_score = knn_reg.score(X_test_standard, y_test)
print(f'knn_reg_score = {knn_reg_score}')

# Knn回归法调参
param_grid = [
    {
        "weights": ["uniform"],
        "n_neighbors": [i for i in range(1, 11)]
    },
    {
        "weights": ["distance"],
        "n_neighbors": [i for i in range(1, 11)],
        "p": [i for i in range(1,6)]
    }
]
knn_reg2 = KNeighborsRegressor()
# n_jobs 默认使用linux-fork，windows需要加上if __name__ == '__main__': 以识别非linux操作系统
# grid_search = GridSearchCV(knn_reg2, param_grid, n_jobs=-1, verbose=1)
# if __name__ == '__main__':
#     grid_search.fit(X_train_standard, y_train)
grid_search = GridSearchCV(knn_reg2, param_grid, verbose=1)
grid_search.fit(X_train_standard, y_train)

# Knn最优參
print(f'best_params_ = {grid_search.best_params_}')
# Knn最优分（CV算法不同于线性回归评分算法R2）
print(f'best_score_ = {grid_search.best_score_}')
# Knn最优回归算法-分数（同于线性回归评分算法R2）
knn_true_score = grid_search.best_estimator_.score(X_test_standard, y_test)
print(f'knn_true_score = {knn_true_score}')