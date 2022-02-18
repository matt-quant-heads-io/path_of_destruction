import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pandas as pd
import numpy as np
from keras.utils import np_utils


obs_size = 5
data_size = 5
version = 3

pod_root_path = f'ddave_exp_traj_obs_{obs_size}_ep_len_77_goal_size_{data_size}'

gameCharacters=" #@H$V*"
tile_map = dict((i, gameCharacters[i]) for i, s in enumerate(["empty", "solid", "player", "exit", "diamond", "key", "spike"]))
print(tile_map)

dfs = []
X = []
y = []

for file in os.listdir(pod_root_path):
    print(f"compiling df {file}")
    if '.csv' in file:
        df = pd.read_csv(f"{pod_root_path}/{file}")
        dfs.append(df)

df = pd.concat(dfs)

df = df.sample(frac=1).reset_index(drop=True)
y_true = df[['target']]
y = np_utils.to_categorical(y_true)
df.drop('target', axis=1, inplace=True)
y = y.astype('int32')

for idx in range(len(df)):
    x = df.iloc[idx, :].values.astype('int32').reshape((obs_size, obs_size, 7))
    X.append(x)

X = np.array(X)

model_abs_path = f"ddave_model_obs_{obs_size}_goal_size_{data_size}_model_num_{version}.h5"

# Model for obs 5
if obs_size == 5:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(obs_size, obs_size, 7), padding="SAME"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="SAME"),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding="SAME"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
elif obs_size == 9:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(obs_size, obs_size, 7), padding="SAME"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="SAME"),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
elif obs_size == 15:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(obs_size, obs_size, 7)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation='softmax')
    ])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=[tf.keras.metrics.CategoricalAccuracy()])
mcp_save = ModelCheckpoint(model_abs_path, save_best_only=True, monitor='categorical_accuracy', mode='max')
history = model.fit(X, y, epochs=500, steps_per_epoch=64, verbose=2, callbacks=[mcp_save])