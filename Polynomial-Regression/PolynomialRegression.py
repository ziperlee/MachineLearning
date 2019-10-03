"""
 Create by zipee on 2019/10/2.
"""
__author__ = "zipee"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


x = np.random.uniform(-3, 3, 100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

# 多项式数据升维
poly = PolynomialFeatures(degree=2)
poly.fit(X)
X2 = poly.transform(X)

# 多出一列为X0
print(f"X2.shape = {X2.shape}")

# 多项式回归训练
lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)
# 系数
print(f"lin_reg2.coef_ = {lin_reg2.coef_}")
# 截距
print(f"lin_reg2.intercept_ = {lin_reg2.intercept_}")

# 绘图 x必须排序，因为plot点的连线根据数组的顺序来
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color="r")
plt.show()

# pipeline 一条龙服务，缩减代码量
poly_reg = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=2)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    ]
)
poly_reg.fit(X, y)
y_predict = poly_reg.predict(X)
# 系数
print(f"lin_reg2.coef_ = {lin_reg2.coef_}")
# 截距
print(f"lin_reg2.intercept_ = {lin_reg2.intercept_}")
