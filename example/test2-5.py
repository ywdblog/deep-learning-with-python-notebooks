# -*- coding: utf-8 -*

# Broadcasting


from tkinter import Y
import numpy as np


# 如果将两个形状不同的张量相加，会发生什么？
# 例如，我们将一个形状为(32, 10)的张量与形状为(10,)的张量相加
# 由于无法对两个形状不同的张量进行相加运算，因此我们需要对其中一个张量进行扩展，使其形状与另一个张量相同
# 这种将形状较小的张量扩展为形状较大的张量的过程，称为广播（broadcasting）
# 例如，我们可以将形状为(10,)的张量扩展为形状为(1, 10)的张量，然后再与形状为(32, 10)的张量相加
# 两个张量的形状相同后，就可以进行相加运算了
x = np.random.random((32, 10))
y = np.random.random((10,))


y = np.expand_dims(y, axis=0)
Y = np.concatenate([y] * 32, axis=0)
z = x+Y
print(z, z.shape)


# 现实世界中不用手动保持维度相同运算，不同维度的张量max运算（自动广播）
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
z = np.maximum(x, y)
print(z)
