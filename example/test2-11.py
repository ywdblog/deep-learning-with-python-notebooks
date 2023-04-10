# -*- coding: utf-8 -*
#  Reimplementing our first example from scratch in TensorFlow

from tensorflow.keras.datasets import mnist
import math
import tensorflow as tf
import sys
imprt numpy as np

class NaiveDense:
    def __init__(self, input_size, output_size, activation):

        # activation是一个逐元素的函数（通常是relu，但最后一层是softmax）
        self.activation = activation

        # tf.random.uniform 生成一个形状为(2, 2)的张量，它的元素是在[0,1)区间内均匀分布的随机数
        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)

        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)

        # tf.Variable 类型的变量可以被自动求导，而普通的张量类型的变量则不行
        self.b = tf.Variable(b_initial_value)

    # 前向传播
    def __call__(self, inputs):
        # tf.matmul()函数是矩阵乘法
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    # 获取权重的快捷方法
    @property
    def weights(self):
        return [self.W, self.b]

# 将层连接起来
class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        print("inputs:", inputs)
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        # 从每一层中获取权重
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights

class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        # 计算总共有多少个batch
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        # 如果当前的index已经超过了总的样本数，那么就重新开始
        images = self.images[self.index: self.index + self.batch_size]
        labels = self.labels[self.index: self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        # predictions返回的实际上是张量
        predictions = model(images_batch)
        # print("predictions", predictions)

        # 计算损失
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions)

        # 计算平均损失
        average_loss = tf.reduce_mean(per_sample_losses)

    # 计算梯度
    # 计算损失相对权重的梯度，输出的gradients是一个列表，列表中的每个元素对应model.weights中的权重
    gradients = tape.gradient(average_loss, model.weights)

    # 更新权重
    update_weights(gradients, model.weights)
    return average_loss

learning_rate = 1e-3

# 更新权重
# 由于权重是tf.Variable类型的变量，所以可以直接使用assign_sub方法来更新权重
# 学习率乘以梯度，然后减去这个值，就是更新后的权重
def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign_sub(g * learning_rate)

# 一般不会手动更新权重，而是使用优化器来更新权重
'''
from tensorflow.keras import optimizers
optimizer = optimizers.SGD(learning_rate=1e-3)
def update_weights2(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))
'''

def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])

fit(model, train_images, train_labels, epochs=10, batch_size=128)
print(len(model.weights))
# assert len(model.weights) == 4

predictions = model(test_images)
# predictions.numpy 返回的是一个numpy数组
predictions = predictions.numpy()

# np.argmax 返回的是每一行中最大值的索引
predicted_labels = np.argmax(predictions, axis=1)

# 计算准确率
# matches是一个布尔数组，表示预测的标签和真实的标签是否相同
# matches.mean() 就是预测正确的样本数除以总样本数
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")

# 总结

'''
1：张量构成了现代机器学习系统的基石。它具有不同的dtype（数据类型）、rank（阶）、shape（形状）等。

2：你可以通过张量运算（比如加法、张量积或逐元素乘法）对数值张量进行操作。这些运算可看作几何变换。一般来说，深度学习的所有内容都有几何解释。

3：深度学习模型由简单的张量运算链接而成，它以权重为参数，权重就是张量。模型权重保存的是模型所学到的“知识”。

4：学习是指找到一组模型参数值，使模型在一组给定的训练数据样本及其对应目标值上的损失函数最小化。

5：学习的过程：随机选取包含数据样本及其目标值的批量，计算批量损失相对于模型参数的梯度。随后将模型参数沿着梯度的反方向移动一小步（移动距离由学习率决定）。这个过程叫作小批量随机梯度下降。

6：整个学习过程之所以能够实现，是因为神经网络中所有张量运算都是可微的。因此，可以利用求导链式法则来得到梯度函数。这个函数将当前参数和当前数据批量映射为一个梯度值。这一步叫作反向传播。

7：在将数据输入模型之前，你需要先对这二者进行定义。损失是在训练过程中需要最小化的量。它衡量的是当前任务是否已成功解决。
优化器是利用损失梯度对参数进行更新的具体方式，比如RMSprop优化器、带动量的随机梯度下降（SGD）等
'''
