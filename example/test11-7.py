# Two approaches for representing groups of words: Sets and sequences
##  Processing words as a sequence: The sequence model approach
### Learning word embeddings with the Embedding layer


# 词嵌入空间通常包含上千个可解释（性别，年龄）的向量，它们可能都很有用
## （1）从头开始训练词向量，一开始是随机的词向量，然后对这些词向量进行学习，学习方式与学习神经网络权重相同。 
## 世界上有许多种语言，它们之间并不是同构的，因为语言反映的是特定文化和特定背景，某些语义关系的重要性因任务而异，所以一般对每个任务学习新的嵌入空间。

import os, pathlib, shutil, random
from tensorflow import keras
from tensorflow.keras import layers
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

 
max_length = 600
max_tokens = 20000
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)

text_only_train_ds = train_ds.map(lambda x, y: x) 
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

inputs = keras.Input(shape=(None,), dtype="int64")
# 输入形状(batch_size, sequence_length)，输出是(batch_size, sequence_length, embedding_dimension)的3阶浮点数张量
embedded = layers.Embedding(
    input_dim=max_tokens, output_dim=256, mask_zero=True)(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded) #LSTM只需要处理256维的向量序列，而不是2000维的序列
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
# 将Embedding层实例化时，它的权重（内部的词向量字典）是随机初始化的，就像其他层一样。
# 在训练过程中，利用反向传播来逐渐调节这些词向量，改变空间结构，使其可以被下游模型利用。
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("embeddings_bidir_gru_with_masking.keras",
                                    save_best_only=True)
]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
model = keras.models.load_model("embeddings_bidir_gru_with_masking.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")