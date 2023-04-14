# -*- coding: utf-8 -*
# Anatomy of a neural network: Understanding core Keras APIs
## Layers: The building blocks of deep learning
### The base Layer class in Keras
#### A Dense layer implemented as a Layer subclass

'''
不同类型的层适用于不同的张量格式和不同类型的数据处理：
- Keras的Dense累：全链接层，向量数据存储中shape为(sample,features)的二阶张量中
- 循环层（LSTM活Conv1D）：(sample,timesteps,features)的三阶张量中
- 二维卷积层(Conv2D)：图像数据
'''

from tensorflow import keras
import tensorflow as tf
import sys 

# Keras的Layer类是所有层的基类，它封装了层的核心功能，包括状态（权重）和计算（一次前向传播），分别在build()和call()方法中实现

class SimpleDense(keras.layers.Layer):

    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    # 权重一般是在build方法中创建的
    def build(self, input_shape):
        # -1表示最后一个维度
        input_dim = input_shape[-1]
        print("input_shape",input_shape)

        # add_weight()方法创建权重 shape=(input_dim, self.units)表示权重的形状 initializer="random_normal"表示权重的初始化方式
        # self.W <tf.Variable 'simple_dense/Variable:0' shape=(784, 32) dtype=float32
        # shape 表示权重的形状 784表示输入的维度 32表示输出的维度
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer="random_normal")
        # 权重也可以独立创建，比如下面的代码 self.W = tf.Variable(tf.random.normal([input_dim, self.units]), name="W")
        
        print("self.W",self.W)
         
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros")

    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y


my_dense = SimpleDense(units=32, activation=tf.nn.relu)
# input_tensor 2表示批量大小 784表示输入的维度
input_tensor = tf.ones(shape=(2, 784))

output_tensor = my_dense(input_tensor)
# shape=(2, 32) 表示输出的形状 2表示批量大小 32表示输出的维度
# 批量大小什么用处呢？批量大小表示一次输入多少个样本，比如批量大小为2，表示一次输入2个样本，这样可以加快运算速度
print(output_tensor.shape) # (2,32)