# Two approaches for representing groups of words: Sets and sequences
## Processing words as a set: The bag-of-words approach
### Single words (unigrams) with binary encoding

import os, pathlib, shutil, random
from tensorflow import keras
from tensorflow.keras import layers

# Preparing the IMDB movie reviews data
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

# Displaying the shapes and dtypes of the first batch
for inputs,targets in train_ds:
    print("inputs[0]",inputs[0])
    print("targets[0]",targets[0])
    print("inputs.shape",inputs.shape)
    break 
 

# Preprocessing our datasets with a `TextVectorization` layer
text_vectorization = layers.TextVectorization(
    ngrams = 1,
    max_tokens=20000,
    output_mode="multi_hot",
)

 

text_only_train_ds = train_ds.map(lambda x, y: x)  
text_vectorization.adapt(text_only_train_ds)

binay_1gram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=1)


binay_1gram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=1)

binay_1gram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=1)

# Inspecting the output of our binary unigram dataset

for inputs, targets in binay_1gram_train_ds:
    print("inputs.shape:", inputs.shape)
    print("inputs.dtype:", inputs.dtype)
    print("targets.shape:", targets.shape)
    print("targets.dtype:", targets.dtype)
    print("inputs[0]:", inputs[0])
    print("targets[0]:", targets[0])
    break

# Our model-building utility

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

callbacks = [
    keras.callbacks.ModelCheckpoint("binary_1gram.keras",
    save_best_only=True)
]

# Training and testing the binary unigram model
 
model = get_model()
model.summary()
model.fit(binay_1gram_train_ds, epochs=1, validation_data=binay_1gram_val_ds,callbacks=callbacks)

model = keras.models.load_model("binary_1gram.keras")
print(f"Test acc: {model.evaluate(binay_1gram_test_ds)[1]:.3f}")


 