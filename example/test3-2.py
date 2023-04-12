# -*- coding: utf-8 -*
# 重温 GramdoemtTape API

import tensorflow as tf

# 默认监视可训练变量，因为计算损失相对于可训练变量列表的梯度，是梯度带最常见的用途
input_var = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    result = tf.square(input_var)
# 计算结果相对输入梯度
gradient = tape.gradient(result, input_var)
print("result1:", gradient)

input_const = tf.constant(3.)
with tf.GradientTape() as tape:
    # 如果要监视常数张量，需要设置watch（why？避免浪费资源）
    tape.watch(input_const)
    result = tf.square(input_const)
gradient = tape.gradient(result, input_const)

# 嵌套梯度带
time = tf.Variable(0.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time ** 2
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)