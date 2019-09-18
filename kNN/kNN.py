"""
 Create by zipee on 2018/8/19.
"""
__author__ = 'zipee'

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 超参数
# k
# weights 距离权重，默认无
# p 距离参数，默认2 欧拉距离
# metric 距离定义 默认明科夫斯基距离

raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# 训练数据集
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

# 待预测数据
x = np.array([8.093607318, 3.365731514])

kNN_classifier = KNeighborsClassifier(n_neighbors=6)

# 模型拟合
kNN_classifier.fit(X_train, y_train)

# 预测结果
# 传入np.array 不建议, 将被废除
# kNN_classifier.predict(x)

# 传入矩阵 推荐
X_predict = x.reshape(1, -1)
y_predict = kNN_classifier.predict(X_predict)
print(y_predict[0])