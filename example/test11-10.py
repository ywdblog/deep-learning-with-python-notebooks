# Multi-head attention
## The Transformer encoder
### 一个非向量化实现多头注意力的伪代码

import os, pathlib, shutil, random
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

import numpy as np
import sys 

def self_attention(input_sequence):
    output = np.zeros(shape=input_sequence.shape)
    # 对输入序列中的每个元素进行遍历
    for i, x in enumerate(input_sequence):
        scores = np.zeros(shape=(len(input_sequence),))
        for j, y in enumerate(input_sequence):
            # 计算该词元与其他每个词元之间的点积(注意力分数)
            scores[j] = np.dot(x,y.T) #y.T is the transpose of y
        # 利用规范化因子对注意力分数进行缩放，以确保它们的总和为1（softmax）
        scores /= np.sqrt(input_sequence.shape[1])
        new_x_representation = np.zeros(shape=x.shape)

        #key step in self-attention
        for j, y in enumerate(input_sequence):
            # 利用注意力分数对每个词元的表示进行加权求和
            new_x_representation += y * scores[j]
            output[i] = new_x_representation # 保存新的表示
    return output

arr = self_attention(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

# Keras有一个内置层来实现多头注意力，称为MultiHeadAttention
# import tensorflow as tf
# from tensorflow.keras.layers import Dense, MultiHeadAttention
# import numpy as np
# num_heads = 4
# embed_dim = 256 
# mha_layer = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
# outputs = mha_layer(inputs,inputs,inputs)


# 自注意力的机制
# 想象搜索引擎
# outputs = sum(values*pairwise_scores(queries, keys))))  

# 在实践中，键和值通常是一个序列，比如机器翻译，查询是目标序列，键和值都是源序列
# 在实践中，如果只做序列分类，那么查询、键、值这三者是相同的：将一个序列与自身进行对比，用整个序列的上下文来丰富每个词元的表示

