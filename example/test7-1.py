import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import sys

# Different ways to build Keras models
## The Sequential model


# The Sequential model
model = keras.Sequential()
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
# Calling a model for the first time to build it 
model.build(input_shape=(None, 3))
model.weights

# 函数式API

inputs = keras.Input(shape=(3,), name="my_input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs) # 通过指定输入和输出来实例化模型

## 1：A simple example
print("",inputs.shape,inputs.dtype) 
print( outputs.shape, outputs.dtype)

model.summary() # 打印出模型概述信息