import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import sys

# Different ways to build Keras models
## The Functional API



'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 my_input (InputLayer)       [(None, 3)]               0

 dense (Dense)               (None, 64)                256

 dense_1 (Dense)             (None, 10)                650

=================================================================
Total params: 906
Trainable params: 906
Non-trainable params: 0
'''

## 2：多输入，多输出模型
vocabulary_size = 10000
num_tags = 100
num_departments = 4

title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

features = layers.Concatenate()([title, text_body, tags]) # 通过拼接将输入特征组合成张量features
features = layers.Dense(64, activation="relu")(features) #利用中间层，将输入特征重组为更加丰富的表示

priority = layers.Dense(1, activation="sigmoid", name="priority")(features) #定义模型输出 sigmoid定义了（0，1）之间的标量
department = layers.Dense(
    num_departments, activation="softmax", name="department")(features) #定义模型输出，softmax 定义了一个概率分布

model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])

model.summary() # 打印出模型概述信息

''' 
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 title (InputLayer)             [(None, 10000)]      0           []

 text_body (InputLayer)         [(None, 10000)]      0           []

 tags (InputLayer)              [(None, 100)]        0           []

 concatenate (Concatenate)      (None, 20100)        0           ['title[0][0]',
                                                                  'text_body[0][0]',
                                                                  'tags[0][0]']

 dense_2 (Dense)                (None, 64)           1286464     ['concatenate[0][0]']

 priority (Dense)               (None, 1)            65          ['dense_2[0][0]']

 department (Dense)             (None, 4)            260         ['dense_2[0][0]']

==================================================================================================
'''


import numpy as np

num_samples = 1280

# randint 第一个参数是下限，第二个参数是上限，第三个参数是size
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))

print(title_data.shape,title_data)

'''
(1280, 10000) [[1 0 0 ... 0 1 1]
 [1 1 1 ... 0 0 0]
 [0 1 0 ... 0 1 0]
 ...
 [0 1 1 ... 0 0 1]
 [0 1 0 ... 1 0 1]
 [1 1 0 ... 0 0 1]]
'''
 
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# random.random(size) 生成size大小的随机数
priority_data = np.random.random(size=(num_samples, 1))
print(priority_data.shape,priority_data)
'''
(1280, 1) [[0.87114627]
 [0.3977715 ]
 [0.00420959]
 ...
 [0.26722614]
 [0.67718007]
 [0.40154989]]
'''
 
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

model.compile(optimizer="rmsprop",
              loss=["mean_squared_error", "categorical_crossentropy"],
              metrics=[["mean_absolute_error"], ["accuracy"]])
model.fit([title_data, text_body_data, tags_data],
          [priority_data, department_data],
          epochs=1)

# model.evaluate() 返回损失值和指标值
model.evaluate([title_data, text_body_data, tags_data],
               [priority_data, department_data])
priority_preds, department_preds = model.predict([title_data, text_body_data, tags_data])

# The power of the Functional API: Access to layer connectivity
# 函数式模型是一种图数据结构。这便于我们查看层与层之间是如何连接的，并重复使用之前的图节点（层输出）作为新模型的一部分
# 模型可视化与特征提取

# None表示批量大小
keras.utils.plot_model(model, "ticket_classifier.png")
keras.utils.plot_model(model, "ticket_classifier_with_shape_info.png", show_shapes=True)

print (model.layers)
'''
[<keras.engine.input_layer.InputLayer object at 0x000001827FE1D210>, 
<keras.engine.input_layer.InputLayer object at 0x000001820B92E0E0>, 
<keras.engine.input_layer.InputLayer object at 0x000001820B92D960>, 
<keras.layers.merging.concatenate.Concatenate object at 0x000001820B92FA00>, 
<keras.layers.core.dense.Dense object at 0x00000182088EA650>,
<keras.layers.core.dense.Dense object at 0x000001820BA03E20>,
<keras.layers.core.dense.Dense object at 0x000001820BA03D60>]
'''
print (model.layers[3].input)
'''
[<KerasTensor: shape=(None, 10000) dtype=float32 (created by layer 'title')>, 
<KerasTensor: shape=(None, 10000) dtype=float32 (created by layer 'text_body')>,
<KerasTensor: shape=(None, 100) dtype=float32 (created by layer 'tags')>]
'''
print (model.layers[3].output)
'''
KerasTensor(
type_spec=TensorSpec(shape=(None, 20100), dtype=tf.float32, name=None), 
name='concatenate/concat:0', description="created by layer 'concatenate'")
'''


# Creating a new model by reusing intermediate layer outputs
# 特征提取

features = model.layers[4].output
difficulty = layers.Dense(3, activation="softmax", name="difficulty")(features)

new_model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department, difficulty])
keras.utils.plot_model(new_model, "updated_ticket_classifier.png", show_shapes=True)