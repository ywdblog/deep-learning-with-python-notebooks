from tensorflow.keras.datasets import reuters
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ngle-label,multiclass classification

# 1:Loading the Reuters dataset
#The Reuters dataset 46 mutually exclusive topics

 
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)

print(train_data[10]) # 0-45表示46个主题

# 2:Preparing the data

# multi-hot编码
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    # en
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1. 
    return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
 

# Encoding the input data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Encoding the labels
# one-hot 
y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

# keras 内置方法实现one-hot编码
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# 3：Building your model

# 输出类别为46，所以最后一层的输出维度为46
# 输出类别比较多，所以空间维度比较大，16维度不够，所以使用64维度

# 3-1 ：Model definition

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    #46 输出一个46维的向量，每个维度对应一个输出类别
    # softmax激活函数，输出在0-1之间，所有输出的和为1，输出每个维度的概率
    layers.Dense(46, activation="softmax")  
])

# 3-2：Compiling the model

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy", # 多分类交叉熵损失函数
              metrics=["accuracy"])

# 4：Validating your approach

# 4-1:Setting aside a validation set
# 留出1000个样本作为验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# 4-2:training the model

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 4-3:Plotting the training and validation loss

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib.pyplot as plt
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 4-4:Plotting the training and validation accuracy

# 模型在9轮之后开始过拟合
plt.clf()
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 5：Retraining a model from scratch

model = keras.Sequential([
  layers.Dense(64, activation="relu"),
  layers.Dense(64, activation="relu"),
  layers.Dense(46, activation="softmax")
])
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train,
          y_train,
          epochs=9, #模型在9轮之后开始过拟合
          batch_size=512)
results = model.evaluate(x_test, y_test)

# 返回损失值和精度
print(results) # [0.9800776243209839, 0.7844999432563782] 78%的准确率

#对于均衡的二分类问题，完全随机的分类器能达到50%的精度，该例子有46个类别，各类别的样本数量可能还不一样。
# 那么一个随机基准模型的精度是多少呢？ 

import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
hits_array.mean() # 0.1908356941960061 19%的准确率，所以该模型的准确率还是不错的

# 6：Generating predictions on new data

predictions = model.predict(x_test)
print(predictions[0].shape) # (46,) 46个概率值
print(np.sum(predictions[0])) # 1.0  概率分布的和为1
print(np.argmax(predictions[0])) # 3 该样本属于第3个类别（最大的概率值）

# 7：A different way to handle the labels and the loss

# 7-1：Encoding the labels as integers
y_train = np.array(train_labels)
y_test = np.array(test_labels)

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy", # 稀疏分类交叉熵（对于整数）
              metrics=["accuracy"])