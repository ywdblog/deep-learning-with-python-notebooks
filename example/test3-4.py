# -*- coding: utf-8 -*
# Anatomy of a neural network: Understanding core Keras APIs
## Layers: The building blocks of deep learning
### The base Layer class in Keras
#### A Dense layer implemented as a Layer subclass

from tensorflow import keras
import tensorflow as tf
import sys 

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
        # shape=(784, 32) 表示权重的形状 784表示输入的维度 32表示输出的维度
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer="random_normal")
        
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
print(output_tensor.shape) # (2,32)