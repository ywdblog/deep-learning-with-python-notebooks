import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import boston_housing

# regression 回归问题 预测的是连续值，注意和logistic回归区分（分类问题）

# 1：Loading the Boston Housing dataset

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape) # 404个样本，每个样本有13个数值特征（比如人均犯罪率、住宅的平均房间数）
print(test_data.shape) # 102个样本，每个样本有13个特征
print(train_targets) # 404个样本，每个样本有1个目标值

# 2：Preparing the data
# 2-1：Normalizing the data

# 将取值范围差异很大的数据输入到神经网络中，这是有问题的
# 普遍采用的最佳处理方法是对每个特征进行标准化，即对于输入数据的每个特征（输入数据矩阵的每一列），减去特征平均值，再除以标准差，这样得到的特征平均值为0，标准差为1

mean = train_data.mean(axis=0) # 求每个特征的平均值
train_data -= mean # 每个特征减去平均值
std = train_data.std(axis=0) # 求每个特征的标准差
train_data /= std # 每个特征除以标准差
test_data -= mean # 测试数据也要做同样的处理 和训练数据保持一致（均值和标准差）
test_data /= std  

# import
# 对测试数据进行标准化的平均值和标准差都是在训练数据上计算得到的
# 不能使用测试数据来计算标准化的平均值和标准差

# 3：Building your model
# 3-1 ：Model definition

# build_model 函数用于构建模型，因为后面要使用K-fold validation，所以需要一个函数来构建模型
# 较小的模型可以降低过拟合
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        # 模型的最后一层只有一个单元且没有激活，它是一个线性层。这是标量回归（标量回归是预测单一连续值的回归）的典型设置。添加激活函数将限制输出范围
        layers.Dense(1) # 输出一个值，所以只有一个单元
    ])
    # mse 均方误差 
    # mae 平均绝对误差
    # mae vs mse mae更加稳健，mse对异常值敏感  
    # rmsprop 优化器 
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

# 4：Validating your approach using K-fold validation
# 4-1：K-fold validation

# 验证分数对于验证集的划分方式可能会有很大的方差，这样就无法对模型进行可靠的评估
# K折交叉验证：将可用数据划分为K个分区（K通常取4或5），实例化K个相同的模型，然后将每个模型在K-1个分区上训练，并在剩下的一个分区上进行评估。模型的验证分数等于这K个验证分数的平均值

k = 4
num_val_samples = len(train_data) // k # 404 // 4 = 101
num_epochs = 100 
all_scores = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    # np.concatenate 拼接数组
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=16, verbose=0)
    # evaluate 返回损失值和指标值
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae) # 保存每个fold的mae

print(all_scores) # [2.128, 2.592, 2.753, 2.396] 
print(np.mean(all_scores)) # 2.459 np.mean 求平均值

# 4-2：Saving the validation logs at each fold

# 训练500个轮次，记录每个轮次的验证分数
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)

# 4-3：Building the history of successive mean K-fold validation scores

# Compute the average of the per-epoch MAE scores for all folds
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# 4-4：Plotting validation scores

import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

# 4-5：Plotting validation scores, excluding the first 10 data points
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

# 5：Training the final model

# 最终训练130轮
model = build_model()
model.fit(train_data, train_targets,
          epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

# 6：Generating predictions on new data

predictions = model.predict(test_data)
print(predictions[0])