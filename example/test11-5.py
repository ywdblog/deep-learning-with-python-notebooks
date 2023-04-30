# Two approaches for representing groups of words: Sets and sequences
## Processing words as a set: The bag-of-words approach
### Bigrams with binary encoding
#### Bigrams with TF-IDF encoding

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


# 二元语法点两个变体
## count：可以为二元语法表示添加更多的信息，方法就是计算每个单词或每个N元语法的出现次数（文本的词频直方图）
## TF-IDF：将单词计数减去均值并除以方差，对其进行规范化（均值和方差是对整个训练数据集进行计算得到的），但独热编码稀疏性很大，将每个特征都减去均值，那么就会破坏稀疏性，无论使用哪种规范化方法，都应该只用除法。那用什么作分母呢，TF-IDF表示词频–逆文档频次

text_vectorization = layers.TextVectorization(
    ngrams = 2,
    max_tokens=20000,
    output_mode="tf_idf",  # Configuring `TextVectorization` to return TF-IDF-weighted outputs
    #  output_mode="count" # Configuring the `TextVectorization` layer to return token counts
 
)

text_only_train_ds = train_ds.map(lambda x, y: x)   
text_vectorization.adapt(text_only_train_ds)

tfidf_2gram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=1)
tfidf_2gram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=1)
tfidf_2gram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=1)

def get_model(max_token=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_token,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop",
                  loss= "binary_crossentropy",
                  metrics=["accuracy"])
    
    return model 

model = get_model()
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("tfidf_2gram.keras",
                                    save_best_only=True)
]
model.fit(tfidf_2gram_train_ds.cache(),
          validation_data=tfidf_2gram_val_ds.cache(),
          epochs=10,
          callbacks=callbacks)
model = keras.models.load_model("tfidf_2gram.keras")
print(f"Test acc: {model.evaluate(tfidf_2gram_test_ds)[1]:.3f}")

### 导出能够处理原始字符串的模型
# 在前面的例子中，将文本标准化、拆分和建立索引都作为tf.data管道的一部分。但如果想导出一个独立于这个管道的模型，我们应该确保模型包含文本预处理（否则需要在生产环境中重新实现，这可能很困难，或者可能导致训练数据与生产数据之间的微妙差异，在keras中很简单

# shape=(1,)表示输入是一个字符串
inputs = keras.Input(shape=(1,), dtype="string")
# 应用文本预处理
processed_inputs = text_vectorization(inputs)
# 应用前面训练好的模型
outputs = model(processed_inputs)
# 将端到端的模型实例化
inference_model = keras.Model(inputs, outputs)

import tensorflow as tf
raw_text_data = tf.convert_to_tensor([
    ["That was an excellent movie, I loved it."],
])
predictions = inference_model(raw_text_data)
print(f"{float(predictions[0] * 100):.2f} percent positive")

 


 