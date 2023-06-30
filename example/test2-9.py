# -*- coding: utf-8 -*

import tensorflow as tf

# 1:GradientTape API的使用
# GradientTape API可以用来记录张量的运算过程，然后利用梯度带求导数

# tf.Variable 类型的变量可以被自动求导，而普通的张量类型的变量则不行
# tf.Variable 是一类用于保存可变状态的张量，它的值可以被改变，但是它的值是持久化的，也就是说，当你改变它的值之后，它的值会被保留下来，而不会被重置为默认值

# 1：标量处理

# 将标量x初始化为0
x = tf.Variable(0.)
# 创建一个GradientTape作用域
with tf.GradientTape() as tape:
    # 在作用域内做一些张量运算
    y = 2 * x + 3
# 利用梯度带求y关于x的导数（输出y相对于变量x的梯度）
grad_of_y_wrt_x = tape.gradient(y, x)
print("result1:", grad_of_y_wrt_x)

# 2：GradientTape 也可用于张量运算

# 形状为(2,2)的零张量
x = tf.Variable(tf.random.uniform((2, 2)))
with tf.GradientTape() as tape:
    y = 2 * x + 3
# grad_of_y_wrt_x 也是一个张量，它的形状与x相同，表示 y = 2 * x + 3 在x = [[0,0],[0,0]] 附近的曲率
grad_of_y_wrt_x = tape.gradient(y, x)

# 3：变量列表

# tf.random.uniform 生成一个形状为(2, 2)的张量，它的元素是在[0,1)区间内均匀分布的随机数
W = tf.Variable(tf.random.uniform((2, 2)))
# 形状为(2,)的零张量
b = tf.Variable(tf.zeros((2,)))
x = tf.random.uniform((2, 2))
with tf.GradientTape() as tape:
    # tf.matmul 点积运算
    y = tf.matmul(x, W) + b
# grad_of_y_wrt_W_and_b 是一个长度为2的列表，它的第一个元素是y关于W的导数，第二个元素是y关于b的导数
grad_of_y_wrt_W_and_b = tape.gradient(y, [W, b])
# result3: [<tf.Tensor: shape=(2, 2), dtype=float32, numpy=array([[1.3310782, 1.3310782],[1.2171937, 1.2171937]], dtype=float32)>, <tf.Tensor: shape=(2,), dtype=float32, numpy=array([2., 2.], dtype=float32)>]
print("result3:", grad_of_y_wrt_W_and_b)