import matplotlib.pyplot as plt
import numpy as np


def show_path(pth):
    a = np.load(pth)
    print
    a.shape
    plt.axis('off')
    plt.imshow(a, cmap='gray')


plt.subplot(131)
show_path('/Users/oliver/PycharmProjects/mlvae/combine.npy')
plt.subplot(132)
show_path('/Users/oliver/PycharmProjects/mlvae/content.npy')
plt.subplot(133)
show_path('/Users/oliver/PycharmProjects/mlvae/style.npy')
plt.show()