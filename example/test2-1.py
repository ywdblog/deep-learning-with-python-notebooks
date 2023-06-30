# -*- coding: utf-8 -*
# matplotlib
# tensorflow

# A first look at a neural network

from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 打印训练集的维度
print(train_images.shape)
print(len(train_labels))

# 打印训练集的标签维度
print(train_labels.shape)
print(len(train_labels))

# 包含 60000 个 28x28 的数字图像的训练集
print(train_images[:2])

# 网络层的架构
# 网络层的堆叠：tf.keras.Sequential
# 网络层的配置：layers.Dense
# 网络层的激活函数：activation="relu"
# 网络层的输出：layers.Dense(10, activation="softmax")
# 包含两个dense层的网络（全连接网络层），relu激活函数，10路softmax分类层
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 编译模型
# 损失函数：loss="sparse_categorical_crossentropy"
# 优化器：optimizer="rmsprop"
# 指标：metrics=["accuracy"]
# rmsprop 优化器是一种随机梯度下降方法，它使用了梯度平方的指数加权移动平均值来调整学习率
# sparse_categorical_crossentropy 损失函数是用于多分类问题的交叉熵损失函数
# metrics=["accuracy"] 表示精度
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 准备图像数据
# 将图像数据从整数转换为浮点数
# 将图像数据从三维数组转换为二维数组
# 将图像数据归一化
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

# 准备标签
# 将标签从整数转换为浮点数
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# 训练模型
# 为什么要将训练数据分为训练集和验证集？
# epochs=5  为什么要训练5个周期？
# batch_size=128 为什么要设置batch_size=128？ 
# batch_size 表示每次训练的样本数
# 输出结果：loss: 0.0280 - accuracy: 0.9914 - val_loss: 0.0729 - val_accuracy: 0.9800
# loss 表示训练集上的损失 损失表示模型的准确性
# accuracy 表示训练集上的精度 
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# 评估模型
# 输出结果：[0.07294893205165863, 0.9800000190734863]
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
# predictions[0] 表示第一个数字的概率值
# argmax() 返回最大值（概率）的索引
# predictions[0][7] 表示第一个数字的第8个概率值，即数字7的概率值
print(predictions[0], predictions[0].argmax(), predictions[0][7],test_labels[0])

# 保存模型
model.save("model.h5")

# 在测试集上评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")

# 加载模型
from tensorflow.keras.models import load_model
loadmodel = load_model("model.h5")
#loadmodel.evaluate 表示在测试集上评估模型
test_loss, test_acc = loadmodel.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")