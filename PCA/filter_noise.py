"""
 Create by zipee on 2019/9/28.
"""
__author__ = "zipee"

# 实现手写识别图片降噪

from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def plot_digits(data):
    fig, axes = plt.subplots(
        10,
        10,
        figsize=(10, 10),
        subplot_kw={"xticks": [], "yticks": []},
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(
            data[i].reshape(8, 8), cmap="binary", interpolation="nearest", clim=(0, 16)
        )

    plt.show()


digits = datasets.load_digits()
X = digits.data
y = digits.target

# 手动增加噪音
noisy_digits = X + np.random.normal(0, 4, size=X.shape)

# 取出y==0的10行
example_digits = noisy_digits[y == 0, :][:10]
# 取出y==1..9的90行并叠加
for num in range(1, 10):
    example_digits = np.vstack([example_digits, noisy_digits[y == num, :][:10]])

# 绘图
plot_digits(example_digits)

# pca降维处理
pca = PCA(0.5).fit(noisy_digits)
components = pca.transform(example_digits)
# 升维还原（降噪）
filtered_digits = pca.inverse_transform(components)
plot_digits(filtered_digits)
