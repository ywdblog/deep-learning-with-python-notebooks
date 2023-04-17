from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 1：Loading the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000) #低频次被舍弃，且只出现一次都单词对于分类没有意义

print (train_data[0],train_labels[0],max([max(sequence) for sequence in train_data]) )

# 1-1：ecoding reviews back to text
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]) #key是数字 value是单词
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]])

print(decoded_review)

# 2：Preparing the data

# 整数列表的长度各不相同，但神经网络处理的是大小相同的数据批量。需要将列表转换为张量，通常有两种方法：
# - 填充列表：使其长度相等，再将列表转换成形状为(samples, max_length)的整数张量，然后在模型第一层使用能处理这种整数张量的层（Embedding层）
# - 对列表进行multi-hot编码，将其转换为由0和1组成的向量，对列表进行multi-hot编码，将其转换为由0和1组成的向量

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    # en
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1. 
    return results

# 2-1 ：Encoding the integer sequences via multi-hot encoding （向量化）

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(x_train[0])

# 标签向量化
y_train = np.asarray(train_labels).astype("float32") #
y_test = np.asarray(test_labels).astype("float32")

# 3：Building your model

# 3-1：Model definition

# 输入数据是向量，标签是标量(0,1)，这种最简单的一类问题，一般用带有relu激活函数的密集链接层(Dense)的简单堆叠
# 两个关键决策（社交网络多少层，每层有多少个单元）
# 两个中间层，每层16个单元
# 第三层输出一个标量预测值
model = keras.Sequential([
    layers.Dense(16, activation="relu"), #16表示该层的unit个数，表示空间的维度大小
    layers.Dense(16, activation="relu"), #16个单元对应的权重矩阵W的形状为(input_dimension, 16)，与W做点积相当于把输入数据投影到16维表示空间中（然后再加上偏置向量b并应用relu运算）
    layers.Dense(1, activation="sigmoid") #sigmoid函数则将任意值“压缩”到[0, 1]区间内，输出可以看作概率值
])
# 空间的维度直观理解为“模型学习内部表示时所拥有的自由度”。单元越多（表示空间的维度越高），模型就能学到更加复杂的表示，但模型的计算代价也变得更大，并可能导致学到不必要的模式。
# 一个二分类问题，模型输出是一个概率值（模型最后一层只有一个单元并使用sigmoid激活函数），所以最好使用binary_crossentropy（二元交叉熵）损失函数

# 3-2: Compiling the model
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# 4：Validating your approach

# 4-1：Setting aside a validation set （校验集）

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 4-2：Training your model

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val)) #一般用验证集来监控训练过程中的模型精度

history_dict = history.history
history_dict.keys() #history对象包括训练集/校验集的精度指标和loss指标

# 4-3：Plotting the training and validation loss
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 4-4 :Plotting the training and validation accuracy

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 模型在训练数据上的表现越来越好，但在前所未见的数据上不一定表现得越来越好。准确地说，这种现象叫作过拟合（overfit）
# 在第4轮之后，你对训练数据过度优化，最终学到的表示仅针对于训练数据，无法泛化到训练集之外的数据

# 4-5 :Retraining a model from scratch
 
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid") #sigmoid激活函数，输出值在0-1之间
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512) # epochs 第四轮就结束
results = model.evaluate(x_test, y_test)

# 5：Using a trained model to generate predictions on new data

print(model.predict(x_test))