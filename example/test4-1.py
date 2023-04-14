from tensorflow.keras.datasets import imdb
import numpy as np

# 1：Loading the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

print (train_data[0],train_labels[0],max([max(sequence) for sequence in train_data]) )

# 1-1：ecoding reviews back to text
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]) #key是数字 value是单词
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]])

print(decoded_review)

# 2：Preparing the data

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
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# 3：Building your model

from tensorflow import keras
from tensorflow.keras import layers

# 3-1：Model definition

# 输入数据是向量，标签是标量(0,1)，这种最简单的一类问题一般用带有relu激活函数的密集链接层(Dense)的简单堆叠
# 两个关键决策（社交网络多少层，每层有多少个单元）
# 两个中间层，每层16个单元
# 第三层输出一个标量预测值
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

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
                    validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys() #history对象包括训练集/校验集的精度指标和loss

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

# 4-5 :Retraining a model from scratch
# 避免过拟合，中第四轮后结束

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

# 5：Using a trained model to generate predictions on new data

print(model.predict(x_test))
