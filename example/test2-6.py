# -*- coding: utf-8 -*

# Tensor product 点积运算


import numpy as np

# numpy中的点积运算
x = np.random.random((32,))
y = np.random.random((32,))
z = np.dot(x, y)

# python 原生实现的点积运算

# 两个向量的点积是一个标量，而且只有元素个数相同的向量才能进行点积运算

def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


# 矩阵和向量的点积
# 矩阵x和一个向量y做点积运算，其返回值是一个向量，其中每个元素是y和x每一行的点积，也就是x的第一维度的大小和y的第0维必须相同
def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z


# 矩阵和矩阵的点积
# 当且仅当x.shape[1] == y.shape[0]时，你才可以计算它们的点积（dot(x, y)）。
# 点积结果是一个形状为(x.shape[0],y.shape[1])的矩阵，其元素是x的行与y的列之间的向量点积
def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z
