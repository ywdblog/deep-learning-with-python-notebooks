from tensorflow.keras.layers import TextVectorization
import re
import string
import tensorflow as tf

# Using the TextVectorization layer


# test11-1.py 中的实现方式并不高效，因为它需要在内存中存储整个词汇表，这会导致内存不足的问题，可采用如下方式解决

# 文本向量化
text_vectorization = TextVectorization(
    # 设置该层的输出模式为整数
    # 设置该层的返回值是编码为整数索引的单词序列
    output_mode="int",
)

# 自定义标准化函数
def custom_standardization_fn(string_tensor):
    lowercase_string = tf.strings.lower(string_tensor)
    return tf.strings.regex_replace(
        lowercase_string, f"[{re.escape(string.punctuation)}]", "")

# 自定义分词函数（作用对象是tf.strings张量）
def custom_split_fn(string_tensor):
    return tf.strings.split(string_tensor)

text_vectorization = TextVectorization( 
    output_mode="int",
    standardize=custom_standardization_fn,
    split=custom_split_fn,
)

dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
]

#dataset参数是一个tf.data.Dataset对象或者是一个可迭代的Python对象
text_vectorization.adapt(dataset)

vocabulary = text_vectorization.get_vocabulary()
test_sentence = "I write, rewrite, and still rewrite again"

encoded_sentence = text_vectorization(test_sentence)
print(encoded_sentence)

inverse_vocab = dict(enumerate(vocabulary))
decoded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence)
print(decoded_sentence)




## （1）在tf.data 管道中使用TextVectorization层
string_dataset = tf.data.Dataset.from_tensor_slices(dataset).batch(1)
int_sequence_dataset = string_dataset.map(text_vectorization,num_parallel_calls=tf.data.experimental.AUTOTUNE)
print(int_sequence_dataset)

## （2）将TextVectorization层作为模型的一部分

# text_input = tf.keras.Input(shape=(), dtype=tf.string)
# vectorized_text = text_vectorization(text_input)
# embedding = tf.keras.layers.Embedding(...)(vectorized_text)