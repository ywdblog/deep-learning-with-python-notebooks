# -*- coding: utf-8 -*
# Anatomy of a neural network: Understanding core Keras APIs
## From layers to models

from tensorflow import keras
import tensorflow as tf
import sys 
import numpy as np

''' 
在Keras中构建模型通常有两种方法：直接作为Model类的子类，或者使用函数式API
机器学习就是在预先定义的可能性空间内，利用反馈信号的指引，寻找特定输入数据的有用表示。通过选择网络拓扑结构，你可以将可能性空间（假设空间）限定为一系列特定的张量运算，将输入数据映射为输出数据。然后，你要为这些张量运算的权重张量寻找一组合适的值。
要从数据中学习，你必须对其进行假设。这些假设定义了可学习的内容。因此，假设空间的结构（模型架构）是非常重要的。它编码了你对问题所做的假设，即模型的先验知识
'''

# 1：The "compile" step: Configuring the learning process
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(optimizer="rmsprop",
              loss="mean_squared_error",
              metrics=["accuracy"])

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])
print( keras.losses.CategoricalCrossentropy() ) #其它损失函数

# learning_rate 参数
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])
print(keras.metrics.AUC) #其它指标

# 2:Picking a loss function

'''
神经网络会采取各种方法使损失最小化，如果损失函数与成功完成当前任务不完全相关，那么神经网络最终的结果可能会不符合你的预期。想象一下，利用SGD训练一个愚蠢而又无所不能的人工智能体，损失函数选择得非常糟糕

指导原则
- 二分类问题：使用二元交叉熵损失函数
- 多分类问题：使用分类交叉熵损失函数。
'''

# 3：Understanding the fit() method：Training the model on Numpy data or tf.data datasets

inputs = np.random.random((1000, 32))
targets = np.random.random((1000, 10))
history = model.fit(
    inputs, # Numpy array of input data
    targets,
    epochs=5, # Number of epochs to train the model
    batch_size=128 # Number of samples per gradient update
)

print(history.history) #查看训练过程中的损失和指标

# 4：Monitoring loss and metrics on validation data

#validation_data 和 train_data  
indices_permutation = np.random.permutation(len(inputs))
shuffled_inputs = inputs[indices_permutation]
shuffled_targets = targets[indices_permutation]

num_validation_samples = int(0.3 * len(inputs))
val_inputs = shuffled_inputs[:num_validation_samples]
val_targets = shuffled_targets[:num_validation_samples]
training_inputs = shuffled_inputs[num_validation_samples:]
training_targets = shuffled_targets[num_validation_samples:]
model.fit(
    training_inputs,
    training_targets,
    epochs=5,
    batch_size=16,
    validation_data=(val_inputs, val_targets)
)

# 5：Inference: Using a model after training

new_inputs = np.random.random((1000, 32))
predictions = model.predict(new_inputs)
 