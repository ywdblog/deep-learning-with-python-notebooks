import numpy as np

# 向量 * (不是点积运算)
def elementwise_multiply(a, b):
    assert a.shape == b.shape

    result = np.zeros_like(a)  # 创建一个与a形状相同的零数组

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            result[i, j] = a[i, j] * b[i, j]  # 逐元素相乘

    return result

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[2, 2, 2], [3, 3, 3]])

result = elementwise_multiply(a, b)
print(result)