"""
 Create by zipee on 2019/10/9.
"""
__author__ = "zipee"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso

# 模型正则化


# 多项式回归
def PolynomialRegression(degree):
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree)),
            ("std_scaler", StandardScaler()),
            ("lin_reg", LinearRegression()),
        ]
    )


# 岭回归
def RidgeRegression(degree, alpha):
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree)),
            ("std_scaler", StandardScaler()),
            ("ridge_reg", Ridge(alpha=alpha)),
        ]
    )


# LASSO
def LassoRegression(degree, alpha):
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree)),
            ("std_scaler", StandardScaler()),
            ("lasso_reg", Lasso(alpha=alpha)),
        ]
    )


def plot_model(model):
    X_plot = np.linspace(-3, 3, 100).reshape(100, 1)
    y_plot = model.predict(X_plot)

    plt.scatter(x, y)
    plt.plot(X_plot[:, 0], y_plot, color="r")
    plt.axis([-3, 3, 0, 6])
    plt.show()


np.random.seed(42)
x = np.random.uniform(-3.0, 3.0, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x + 3 + np.random.normal(0, 1, size=100)

np.random.seed(666)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# 普通多项式回归-泛化能力差（均方误差大）
poly_reg = PolynomialRegression(degree=20)
poly_reg.fit(X_train, y_train)
y_poly_predict = poly_reg.predict(X_test)
poly_mean_squared_error = mean_squared_error(y_test, y_poly_predict)
print(f"poly_mean_squared_error = {poly_mean_squared_error}")
plot_model(poly_reg)

# 岭回归-泛化能力好（均方误差小）
ridge1_reg = RidgeRegression(20, 0.0001)
ridge1_reg.fit(X_train, y_train)

y1_predict = ridge1_reg.predict(X_test)
ridge_mean_squared_error = mean_squared_error(y_test, y1_predict)
print(f"ridge_mean_squared_error = {ridge_mean_squared_error}")
plot_model(ridge1_reg)

# 随着alpha变大，趋近于直线
# ridge3_reg = RidgeRegression(20, 100)
# ridge3_reg.fit(X_train, y_train)
# y3_predict = ridge3_reg.predict(X_test)
# mean_squared_error(y_test, y3_predict)
# plot_model(ridge3_reg)
#
# ridge4_reg = RidgeRegression(20, 10000000)
# ridge4_reg.fit(X_train, y_train)
# y4_predict = ridge4_reg.predict(X_test)
# mean_squared_error(y_test, y4_predict)
# plot_model(ridge4_reg)

# LASSO 具有特征选择能力
lasso1_reg = LassoRegression(20, 0.01)
lasso1_reg.fit(X_train, y_train)
y1_predict = lasso1_reg.predict(X_test)
lasso_mean_squared_error = mean_squared_error(y_test, y1_predict)
print(f"lasso_mean_squared_error = {lasso_mean_squared_error}")
plot_model(lasso1_reg)

# lasso2_reg = LassoRegression(20, 0.1)
# lasso2_reg.fit(X_train, y_train)
# y2_predict = lasso2_reg.predict(X_test)
# mean_squared_error(y_test, y2_predict)
#
# lasso3_reg = LassoRegression(20, 1)
# lasso3_reg.fit(X_train, y_train)
# y3_predict = lasso3_reg.predict(X_test)
# mean_squared_error(y_test, y3_predict)
