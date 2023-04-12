# -*- coding: utf-8 -*
# An end-to-end example: A linear classifier in pure TensorFlow

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys

# 生成一些线性可分的数据：二维平面上的点，它们分为两个类别。生成方法是从一个具有特定协方差矩阵和特定均值的随机分布中抽取坐标来生成每一类点
# 设定：两个点云的协方差矩阵相同，但均值不同。也就是说两个点云具有相同的形状，但位置不同。

num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class)

# mean: 均值，cov: 协方差，size: 样本数
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class)

# print(positive_samples) # [[ 3.66077962  0.29349689] [ 2.28236475  2.35068635] ]

# np.vstack()沿着垂直方向堆叠数组构成一个新的数组
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
#print(len(inputs),inputs[1:4])
 
# labels for each sample are 0 or 1
# 前1000个是0，后1000个是1 [[0],[0]...[1],[1]]
# 如果inputs[0]属于类别0，则targets[i,0]为0 反之为1
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))

# plt.scatter 画散点图 
# 第一个参数是x,y轴的值（均值），第二个参数是形状（协方差），第三个参数是颜色
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()

# Creating the linear classifier variables
# 每个样本的输入是2维的，因此输入维度为2
# 输出维度为1，因为我们只需要一个值来表示样本属于正类的概率，接近1表示属于正类，接近0表示属于负类
output_dim = 1
input_dim = 2

W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

'''
# 前向函数 The forward pass function
# W = [[w1], [w2]]
# 预测值=输入*权重+偏置= prediction = [[w1], [w2]]•[x, y] + b = w1 * x + w2 * y + b。
'''
def model(inputs):
    return tf.matmul(inputs, W) + b

# 均方误差损失函数 The mean squared error loss function
def square_loss(targets, predictions):
    # tf.square()计算张量的平方
    per_sample_losses = tf.square(targets - predictions)
    # tf.reduce_mean()计算张量的平均值
    return tf.reduce_mean(per_sample_losses)

# the training loop
learning_rate = 0.1

# 一次就训练2000条
def training_step(inputs, targets):
    # 进行一次前向传播
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        # loss 函数
        loss = square_loss(predictions, targets)
    # 检索损失相对于模型变量（权重）的梯度
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    # 更新权重
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss

# 批量训练步骤
# 在所有数据上进行训练（计算梯度并更新权重），非小批量训练
for step in range(40):
    loss = training_step(inputs, targets)
    print(f"Loss at step {step}: {loss:.4f}")

predictions = model(inputs)
# 最后一个参数决定颜色，如果大于0.5则为红色，否则为蓝色
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()

# 对于给定点[x, y]，其预测值是prediction = [[w1],[w2]]•[x, y] + b = w1 * x + w2 * y + b
# 因此类别0的定义是w1 *x + w2 * y + b < 0.5，类别1的定义是w1 * x + w2 * y + b > 0.5
# 因此，我们可以将分类边界绘制为w1 * x + w2 * y + b = 0.5 （一条直线）
# 可能习惯看到像y = a * x + b这种形式的直线方程，如果将我们的直线方程写成这种形式，那么它将变为：y = - w1 / w2 * x + (0.5 - b) / w2

x = np.linspace(-1, 4, 100)
# 直线方程
y = - W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)