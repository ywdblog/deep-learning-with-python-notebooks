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

# 输出shape为(timesteps,output_features)的向量序列，出张量中的每个时间步t都包含入序列中时间步0到t的信息，即关于过去的全部信息

# 3：An RNN layer that can process sequences of any length

num_features = 14
inputs = keras.Input(shape=(None, num_features))
outputs = layers.SimpleRNN(16)(inputs)

'''
SimpleRNN层能够像其他Keras层一样处理序列批量，而不是像NumPy示例中的那样只能处理单个序列。
也就是说，它接收形状为(batch_size, timesteps, input_features)的输入，而不是(timesteps,input_features)。
指定初始Input()的shape参数时，可以将timesteps设为None，这样神经网络就能够处理任意长度的序列
'''

# 4：An RNN layer that returns its full output sequence（最后一个输出状态）


'''
所有的循环层都可以在两种模式下运行，
一种是返回每个时间步连续输出的完整序列，即形状为(batch_size,timesteps, output_features)的3阶张量；
另一种是只返回每个输入序列的最终输出，即形状为(batch_size, output_features)的2阶张量
'''

num_features = 14
steps = 120
inputs = keras.Input(shape=(steps, num_features))
outputs = layers.SimpleRNN(16, return_sequences=False)(inputs)
print(outputs.shape)

# 5：Stacking RNN layers 神经网络堆叠

#为了提高神经网络的表示能力，有时将多个循环层逐个堆叠也是很有用的。在这种情况下，你需要让所有中间层都返回完整的输出序列

inputs = keras.Input(shape=(steps, num_features))
x = layers.SimpleRNN(16, return_sequences=True)(inputs)
x = layers.SimpleRNN(16, return_sequences=True)(x)
outputs = layers.SimpleRNN(16)(x)