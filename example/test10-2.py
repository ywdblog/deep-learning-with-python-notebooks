import numpy as np

from tensorflow import keras
from tensorflow.keras import layers


# 1: Simple RNNs  伪代码

def f(input_t,state_t):
    output_t = np.random.random((2,))
    return output_t
state_t = 0
timesteps = 100 # timesteps表示时间步
input_features = 32 # input_features表示输入的维度 
inputs = np.random.random((timesteps, input_features))
for input_t in inputs:
    output_t = f(input_t,state_t)
    state_t = output_t
    print(output_t)


# 2: Simple RNNs  numpy实现

 
timesteps = 100
input_features = 32
output_features = 64
inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features,)) # 每一个输出都是一个shape为(output_features,)的向量
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))
successive_outputs = []
for input_t in inputs:
    # W 的第二个维度是input_features，所以input_t是一个shape为(input_features,)的向量
    # U 的第二个维度是output_features，所以state_t是一个shape为(output_features,)的向量
    # np.dot(W, input_t) 的结果是一个shape为(output_features,)的向量 
    # np.dot(U, state_t) 的结果是一个shape为(output_features,)的向量  
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.stack(successive_outputs, axis=0) # shape为(timesteps,output_features)


# An RNN layer that returns only its last output state

num_features = 14
steps = 120
inputs = keras.Input(shape=(steps, num_features))
outputs = layers.SimpleRNN(16, return_sequences=False)(inputs)
print(outputs.shape)

