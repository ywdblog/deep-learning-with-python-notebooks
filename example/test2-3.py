# -*- coding: utf-8 -*

# Manipulating tensors in NumPy keras_example

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 1：张量切片
# 可以沿着每个张量轴对其进行切片，沿着第一个轴切片，沿着第二个轴切片，沿着第三个轴切片
my_slice1 = train_images[10:100]
my_slice2 = train_images[10:100, :, :]
my_slice3 = train_images[10:100, 0:28, 0:28]
my_slice4 = train_images[:, 14:, 14:]
# 负数索引
my_slice5 = train_images[:, 7:-7, 7:-7]
print(my_slice1.shape, my_slice2.shape,
      my_slice3.shape, my_slice4.shape, my_slice5)

# 2：批量数据   The notion of data batches
# 样本的第一个轴（0轴）通常叫做批量轴（batch axis），样本轴，样本维度
# 在Keras中，批量数据通常以形状为(samples, features)的二维张量组织
batch = train_images[128:256]
n = 2
batch = train_images[128 * n:128 * (n + 1)]

# 向量数据 Vector data 2D tensors of shape (samples, features)，每个样本可以是一个特征向量
# 时间序列数据 Timeseries data or sequence data 3D tensors of shape (samples, timesteps, features)，每个样本是特征向量组成的序列
# 图像数据 Image data 4D tensors of shape (samples, height, width, channels) or (samples, channels, height, width)，每个样本是一个高度为height，宽度为width，通道数为channels的图像
# 视频数据 Video data 5D tensors of shape (samples, frames, height, width, channels) or (samples, frames, channels, height, width)，每个样本是一个高度为height，宽度为width，通道数为channels的视频，由frames个连续帧组成
