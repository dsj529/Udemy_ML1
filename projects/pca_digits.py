'''
Created on Apr 23, 2019

@author: dsj529
'''
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

import numpy as np


# 8x8 pixel per image -> 64 features !!! Humans are not able to cope with this
#    This is why we use PCA -> reduce the dimenions: we can visualize the data in 2D!!!
# We want to investigate if the distribution after PCA reveals the
#  distribution of the different classes, and if they are clearly separable
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

print(digits.images.shape)

n_row, n_col, max_n = 2, 5, 10

fig = plt.figure()
i=0

#===================================================================================================
# while i < max_n and i < digits.images.shape[0]:
#     p = fig.add_subplot(n_row, n_col, i + 1, xticks=[],
#     yticks=[])
#     p.imshow(digits.images[i], cmap=plt.cm.bone,interpolation='nearest')
#     #label the image with the target value
#     p.text(0, -1, str(digits.target[i]))
#     i = i + 1
#  
# plt.show()
#===================================================================================================

estimator = PCA(n_components=10)
X_pca = estimator.fit_transform(X_digits)

colors = ['black', 'blue', 'purple', 'yellow', 'white','red', 'lime', 'cyan', 'orange', 'gray']

plt.clf()
ax = fig.gca(projection='3d')

for i in range(len(colors)):
    px = X_pca[:, 0][y_digits == i]
    py = X_pca[:, 1][y_digits == i]
    pz = X_pca[:, 2][y_digits == i]
    ax.scatter(px, py, pz, c=colors[i], alpha=0.7)
    ax.legend(digits.target_names)
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
plt.show()

