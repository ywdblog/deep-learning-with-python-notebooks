# -*- coding: utf-8 -*

# Constant tensors and variables

import tensorflow as tf
import numpy as np

# 形状为(2,1)d的全1张量
x = tf.ones(shape=(2, 1))
y = np.ones(shape=(2, 1))
print(x, y)

# 形状为(2,1)的零张量
x = tf.zeros(shape=(2, 1))
y = np.zeros(shape=(2, 1))
print(x, y)

# 随机张量
x = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
y = np.random.normal(loc=0., scale=1., size=(3, 1))
print(x, y)

x = np.ones(shape=(2, 1))
x[0, 0] = 1
# 张量不能直接赋值

# 要训练模型，我们需要更新其状态，而模型状态是一组张量。如果张量不可赋值，那么我们该怎么做呢？
# 这时就需要用到变量（variable）。tf.Variable是一个类，其作用是管理TensorFlow中的可变状态

v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
print(v)
v.assign(tf.ones((3, 1)))
# 为子集元素赋值
v[0, 0].assign(3.)

# assign_add()和assign_sub()方法可以对变量的值进行增加和减少，等同于v = v + 1和v = v - 1
v.assign_add(tf.ones((3, 1)))
v.assign_sub(tf.ones((3, 1)))

# 张量数学运算
a = tf.ones((2, 2))
b = tf.square(a)
c = tf.sqrt(a)
d = b + c
# tf.matmul 点积运算
e = tf.matmul(a, b)
e *= d