# -*- coding: utf-8 -*

# Data representations for neural networks

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# scalar 标量: 0阶张量
x = np.array(12)
print(x)  # 12
# x.ndim 表示张量的轴数（阶）
print(x.ndim)  # 0
# x.shape 表示张量的形状
print(x.shape)  # ()

# Vectors 向量:(rank-1 tensors)
# 5维向量不要和5维张量混淆，5维向量只有一个轴
x = np.array([12, 3, 6, 14, 7])
print(x, x.ndim, x.shape)  # [12  3  6 14  7] 1 (5,)


# Matrices 矩阵： (rank-2 tensors)
# 矩阵有两个轴（通常叫做行和列）
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x, x.ndim, x.shape)  # （3,5）

# Rank-3 and higher-rank tensors
x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]]])
print(x, x.ndim, x.shape)


# 加载数据
# 60000张训练图像，10000张测试图像
# 60000个矩阵组成的数组，每个矩阵是28*28的整数矩阵，值在0~255之间
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# ndim 属性，它是张量的轴的个数
print(train_images.ndim)
# shape 属性，它是张量的轴的尺寸组成的元组 轴的维度大小（元素个数）
print(train_images.shape)  # (60000, 28, 28)
# dtype 属性，它是张量中所包含数据的类型
print(train_images.dtype)

# 显示第五张图像
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
