# -*- coding: utf-8 -*

# The gears of neural networks: tensor operations 神经网络中的张量运算

import time
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 下面相当于三个张量运算
# 输入张量和张量W的点积运算，然后再和张量b相加，最后再通过relu进行修正线性单元
#keras.layers.Dense(512, activation='relu')
#output = relu(dot(W, input) + b)

# Element-wise operations python 实现（逐元素运算）


def naive_relu(x):
    assert len(x.shape) == 2

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)  # 逐元素修正
    return x


def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]  # 逐元素相加
    return x


x = np.array([[-5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])

y = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])

# print(x.shape)
print(naive_relu(x))
print(naive_add(x, y))

# NumPy内置函数实现是经过优化的，所以比纯Python实现快得多
x = np.random.random((20, 100))
y = np.random.random((20, 100))

t0 = time.time()
for _ in range(1000):
    z = x + y
    # np.maximum()函数是逐元素比较两个张量，返回一个新的张量，其中包含了两个张量中对应元素的最大值
    z = np.maximum(z, 0.)
print("Took: {0:.2f} s".format(time.time() - t0))

t0 = time.time()
for _ in range(1000):
    z = naive_add(x, y)
    z = naive_relu(z)
print("Took: {0:.2f} s".format(time.time() - t0))
