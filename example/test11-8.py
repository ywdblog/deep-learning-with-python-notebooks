# Two approaches for representing groups of words: Sets and sequences
##  Processing words as a sequence: The sequence model approach
### Learning word embeddings with the Embedding layer
#### 处理掩码序列的Embedding层 Understanding padding and masking

 
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

# extVectorization层中使用了output_sequence_length=max_length选项（max_length为600），如果不够600，就会在序列的末尾填充0，使其长度达到600，从而能够与其它序列对齐。
# 双向RNN模型，正序处理单元，在最后的迭代只会看到表示填充的0的掩码，如果原始句子很短，那么这个模型就会看到很多0，这样就会影响模型的性能，存储在RNN中的信息就会被淹没在0中。
# 需要用某种方式来告诉RNN，它应该跳过这些迭代。有一个API可以实现此功能：掩码（masking）。

inputs = keras.Input(shape=(None,), dtype="int64")
# mask_zero=True 表示将输入中的0作为掩码，这样RNN不会在输入中的0上执行任何计算
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