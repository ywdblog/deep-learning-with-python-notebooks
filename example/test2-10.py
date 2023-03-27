# -*- coding: utf-8 -*

#  Looking back at our first example

from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
 

# 模型包含两个链接在一起的Dense层，每层都对输入数据做一些简单的张量运算，这些运算都涉及权重张量。
# 权重张量是该层的属性，里面保存了模型所学到的知识
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# sparse_categorical_crossentropy 是损失函数，用于学习权重张量的反馈信号，在训练过程中，模型会尽量最小化这个损失函数
# rmsprop 是优化器，用于根据模型看到的数据和自身的损失函数更新模型的权重
# 降低损失值是通过小批量随机梯度下降来实现的。梯度下降的具体方法由第一个参数就是optimizer的值来指定
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


 
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

 
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# 模型开始在训练数据上进行迭代（每个小批量包含128个样本），共迭代5轮（在所有训练数据上迭代一次叫作一轮epoch）
# 对于每批数据，模型会计算损失相对于权重的梯度（利用反向传播算法，这一算法源自微积分的链式法则），并将权重沿着减小该批量对应损失值的方向移动
model.fit(train_images, train_labels, epochs=5, batch_size=128)

 
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
 
print(predictions[0], predictions[0].argmax(), predictions[0][7],test_labels[0])
 
model.save("model2-9.h5")

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")
 
from tensorflow.keras.models import load_model
loadmodel = load_model("model2-9.h5")
test_loss, test_acc = loadmodel.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")


 