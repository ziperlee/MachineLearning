"""
 Create by zipee on 2019/9/28.
"""
__author__ = "zipee"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA


# from sklearn.datasets.base import get_data_home
# print (get_data_home())
# 数据存放目录~/scikit_learn_data/

# 画图函数
def plot_faces(faces):
    fig, axes = plt.subplots(
        6,
        6,
        figsize=(10, 10),
        subplot_kw={"xticks": [], "yticks": []},
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(62, 47), cmap="bone")
    plt.show()


faces = fetch_lfw_people()
# 指定特征人脸最小样本数
# faces2 = fetch_lfw_people(min_faces_per_person=60)

# 随机取出36张脸
random_indexes = np.random.permutation(len(faces.data))
X = faces.data[random_indexes]
example_faces = X[:36, :]

# 绘图
plot_faces(example_faces)

# svd_solver='randomized' 使用随机方式求解PCA
pca = PCA(svd_solver="randomized")
pca.fit(X)

# 绘制特征脸
plot_faces(pca.components_[:36, :])
