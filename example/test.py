
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys 

 

def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z

# 矩阵和向量的点积
# 矩阵x和一个向量y做点积运算，其返回值是一个向量，其中每个元素是y和x每一行的点积，也就是x的第一维度的大小和y的第0维必须相同

# x = np.random.random((2, 5))
# y = np.random.random((5,))

x = np.array([[1,2,3,4,5],[4,5,6,7,8]])
y = np.array([1,2,3,4,5])
z = naive_matrix_vector_dot(x, y)

print(z,z.shape)
sys.exit(0)

def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

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


x = np.array([[1,2,3],[3,4,5]])
y = np.array([[1,2,3,4],[1,1,1,1],[1,1,2,2]])
z = naive_matrix_dot(x, y)
print( z)