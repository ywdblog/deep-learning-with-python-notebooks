import os
from tensorflow import keras
import numpy as np

import sys 
fname = os.path.join("jena_climate_2009_2016.csv")

with open(fname) as f:
    data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
print(header)
print(len(lines))
 
temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    temperature[i] = values[1]
    raw_data[i, :] = values[:]
 

num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples:", num_train_samples) 
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)


mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std

# print("raw_data:", raw_data)

# import numpy as np
# from tensorflow import keras
# int_sequence = np.arange(10)
# dummy_dataset = keras.utils.timeseries_dataset_from_array(
#     data=int_sequence[:-3],
#     targets=int_sequence[3:],
#     sequence_length=3,
#     batch_size=2,
# )

# for inputs, targets in dummy_dataset:
#     for i in range(inputs.shape[0]):
#         print([int(x) for x in inputs[i]], int(targets[i]))

sampling_rate = 6 #每6个数据保留一个
sequence_length = 120 # 过去5天的数据
delay = sampling_rate * (sequence_length + 24 - 1) # 序列目标：24小时后的数据
batch_size = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay], 
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)

test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples)


for samples, targets in train_dataset:
    print("samples shape:", samples.shape) # (256, 120, 14) 每个样本包含120个时间步，每个时间步包含14个特征
    print("targets shape:", targets.shape) # (256,)
    break

from tensorflow import keras
from tensorflow.keras import layers

# sequence_length = 120 过去5天的数据
# 第二个维度的大小是14，因为每个时间步包含14个特征
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(16)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

 
callbacks = [
    keras.callbacks.ModelCheckpoint("jena_lstm.keras",
                                    save_best_only=True)
]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset,
                    callbacks=callbacks)

model = keras.models.load_model("jena_lstm.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")