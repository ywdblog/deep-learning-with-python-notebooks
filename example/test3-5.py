# -*- coding: utf-8 -*
# Anatomy of a neural network: Understanding core Keras APIs
## Layers: The building blocks of deep learning
### The base Layer class in Keras
#### Automatic shape inference: Building layers on the fly


'''
问题：既然Keras的Layer类是所有层的基类，直接通过层的__call__()方法来调用，为什么还要实现call()和build()方法呢？主要原因就是为了能够及时更新状态

 
在test2-11.py：
model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])
层兼容性：下一层的输入维度必须与上一层的输出维度相同，如果不同，需要在层中指定输入维度

其实keras中能自己解决这个问题，只需要在第一层中指定输入维度即可，如下：
layer = layers.Dense(32, activation="relu")
model = models.Sequential([
    layers.Dense(32, activation="relu"),
    layers.Dense(32)
])
'''

'''
为了在NaiveDense中实现自动推断输入维度，需要在build()方法中获取输入的形状，然后根据输入的形状来创建权重，这样就可以在__call__()方法中直接调用call()方法了，就是SimpleDense的实现方式
'''

from tensorflow import keras
import tensorflow as tf
import sys 

import numpy as np
 
class SimpleDense(keras.layers.Layer):

    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        #self.activation = activation
        self.activation = keras.activations.get(activation)
 
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer="random_normal")
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros")

    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y
    
    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)
            self.built = True
        return self.call(inputs)

model = keras.Sequential([
    SimpleDense(32, activation="relu"),
    SimpleDense(64, activation="relu"),
    SimpleDense(32, activation="relu"),
    SimpleDense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Generate some random training data
train_data = np.random.rand(1000, 10)
train_labels = np.random.randint(0, 10, size=(1000,))
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)

# Train the model
model.fit(train_data, train_labels, epochs=10, batch_size=32)

# Generate some random test data
test_data = np.random.rand(100, 10)
test_labels = np.random.randint(0, 10, size=(100,))
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data, test_labels)

# Print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
