# Two approaches for representing groups of words: Sets and sequences
##  Processing words as a sequence: The sequence model approach
### LSTM model && embedding

import os, pathlib, shutil, random
from tensorflow import keras
from tensorflow.keras import layers
batch_size = 32
base_dir = pathlib.Path("aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"
# for category in ("neg", "pos"):
#     os.makedirs(val_dir / category)
#     files = os.listdir(train_dir / category)
#     random.Random(1337).shuffle(files)
#     num_val_samples = int(0.2 * len(files))
#     val_files = files[-num_val_samples:]
#     for fname in val_files:
#         shutil.move(train_dir / category / fname,
#                     val_dir / category / fname)

# text_dataset_from_directory 从磁盘加载文本数据 生成tf.data.Dataset对象 
# 该对象是一个元组（texts,labels） texts是一个形状为（batch_size,）的字符串张量，labels是一个形状为（batch_size,）的整数张量（0或1） 
train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train", batch_size=batch_size
)
val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/val", batch_size=batch_size
)
test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)

for inputs,targets in train_ds:
    print("inputs[0]",inputs[0])
    print("targets[0]",targets[0])
    print("inputs.shape",inputs.shape)
    break 
 
### Preparing integer sequence datasets
# layers.TextVectorization 层将文本转换为整数序列
# max_length 限制每个样本的长度
# max_tokens 限制词表的大小
# output_sequence_length 指定输出序列的长度
max_length = 600
max_tokens = 20000
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)

# train_ds.map 方法将一个函数应用于数据集中的每个元素  map方法返回一个新的数据集，该数据集包含应用函数的结果
text_only_train_ds = train_ds.map(lambda x, y: x) # 准备一个数据集，只包含文本，不包含标签
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
embedded = layers.Embedding(
    input_dim=max_tokens, output_dim=256, mask_zero=True)(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
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