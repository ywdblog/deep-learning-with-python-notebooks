# Multi-head attention
## The Transformer encoder
### Embedding

import os, pathlib, shutil, random
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
 
# LayerNormalization 伪代码实现
def layer_LayerNormalization(batch_of_sequences):
    # mean 表示每个词元的平均值
    # variance 表示每个词元的方差
    # 输出的每个词元都是经过规范化的
    mean = np.mean(batch_of_sequences, axis=-1, keepdims=True)
    variance = np.var(batch_of_sequences, axis=-1, keepdims=True)
    return (batch_of_sequences - mean) / variance  

# 在该任务中，layer normalization 比 batch normalization 更适合，因为每个样本都是一个独立的序列，而不是一个批次中的一部分。
def batch_normalization(batch_of_images) :
    # 输入形状：batch_size , height, width, channels
    mean = np.mean(batch_of_images, axis=(0, 1, 2), keepdims=True)
    variance = np.var(batch_of_images, axis=(0, 1, 2), keepdims=True)
    return (batch_of_images - mean) / variance

batch_size = 32
base_dir = pathlib.Path("aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"



train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train", batch_size=batch_size
)
val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/val", batch_size=batch_size
)
test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)
text_only_train_ds = train_ds.map(lambda x, y: x)


from tensorflow.keras import layers

max_length = 600
max_tokens = 20000
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)
text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=1)
int_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=1)
int_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=1)
 

# Transformer encoder implemented as a subclassed `Layer`

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        # 输入词元向量的维度
        self.embed_dim = embed_dim
        # 内部密集层的维度
        self.dense_dim = dense_dim
        # 注意力头的数量
        self.num_heads = num_heads
        
        # MultiHeadAttention layer 
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        
        # Dense layers
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        # LayerNormalization layer 作用是对输入进行归一化处理
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        
        # embedding inputs and prepare for attention
        # embedding 生成的掩码是二维的，注意力的掩码是三维的，所以需要增加一个维度
        # tf.newaxis 为张量增加一个维度
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        # self.attention 为 MultiHeadAttention layer 第一个参数是 query，第二个参数是 key，第三个参数是 value
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        
        # layernorm_1 作用是对输入进行归一化处理，然后与输入相加，得到 proj_input，再通过 dense_proj 层得到 proj_output，再与 proj_input 相加，得到 layernorm_2 的输出
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    # get_config() 方法用于获取当前对象的配置信息，返回一个字典，字典中包含了对象的参数信息
    # 在自定义层中，如果需要保存自定义层的配置信息，需要重写 get_config() 方法
    def get_config(self):
        # 不包含权重值，因为该层的所有权重都从头开始创建
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

 
    
# Using the Transformer encoder for text classification


vocab_size = 20000
embed_dim = 256
num_heads = 2
dense_dim = 32

inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
# transformerencoder 返回的是完整的序列，所以需要对序列进行池化操作，得到一个向量，然后再通过一个全连接层得到最终的输出 

x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("transformer_encoder.keras",
                                    save_best_only=True)
]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=20, callbacks=callbacks)

# 从文件中加载模型时，应该在加载模型过程中提供自定义层的类，否则会报错
model = keras.models.load_model(
    "transformer_encoder.keras",
    custom_objects={"TransformerEncoder": TransformerEncoder})
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")


# 在本例中，Transformer是一种序列处理架构，最初是为机器翻译而开发的。
# 然而刚刚见到的Transformer编码器根本就不是一个序列模型。你注意到了吗？
# 它由密集层和注意力层组成，前者独立处理序列中的词元，后者则将词元视为一个集合。你可以改变序列中的词元顺序，并得到完全相同的成对注意力分数和完全相同的上下文感知表示
# 如果不查看序列，又怎么能很好进行机器翻译呢？
# 解决方案：Transformer是一种混合方法，它在技术上是不考虑顺序的，但将顺序信息手动注入数据表示中。这就是缺失的那部分，它叫作位置编码（positional encoding

