import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

point_num = 30

data = pd.read_csv('input/rectangle_sets/rectangle_set_1000_30_points.csv', index_col=[0])
data_train, data_test = train_test_split(data, test_size=0.2)

X = data_train[['x_in_' + str(i) for i in range(point_num)] + ['y_in_' + str(i) for i in range(point_num)]]
y = data_train[['x_out_' + str(i) for i in range(4)] + ['y_out_' + str(i) for i in range(4)]]

X_test = data_test[['x_in_' + str(i) for i in range(point_num)] + ['y_in_' + str(i) for i in range(point_num)]]
y_test = data_test[['x_out_' + str(i) for i in range(4)] + ['y_out_' + str(i) for i in range(4)]]
print('X_test shape: ' + str(X_test.shape))
print('y_test shape: ' + str(y_test.shape))

X_val = X[-400:]
y_val = y[-400:]
X_train = X[:-400]
y_train = y[:-400]
print('X_val shape: ' + str(X_val.shape))
print('y_val shape: ' + str(y_val.shape))
print('X_train shape: ' + str(X_train.shape))
print('y_train shape: ' + str(y_train.shape))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='sigmoid', input_shape=(60,)),
    tf.keras.layers.Dense(8)
])


def keras_l2_distance(tgt, pred):
    distance = np.zeros(tgt.shape[0])

    for i in range(4):
        distance += tf.math.sqrt((tgt[:, i] - pred[:, i]) ** 2 + (tgt[:, i + 4] - pred[:, i + 4]) ** 2, name=None)

    return tf.reduce_mean(distance)


model.compile(loss=keras_l2_distance, optimizer='adam')
model.summary()

print('# Fit model on training data')
history = model.fit(X_train, y_train,
                    batch_size=64,
                    epochs=400,
                    validation_data=(X_val, y_val))

print('\n# Evaluate on test data')
results = model.evaluate(X_test, y_test, batch_size=128)
print('test loss, test acc:', results)

print('\n# Generate predictions for 3 samples')
predictions = model.predict(X_test[:3])
print('predictions shape:', predictions.shape)

plt.figure(figsize=(5, 5))
ax = plt.subplot(1, 1, 1)
ax.plot(history.history['loss'], label="loss")
ax.plot(history.history['val_loss'], label="val_loss")
plt.xlabel('epochs')
plt.ylabel('loss distance')
ax.legend()
plt.show()

for i in range(3):
    plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)
    ax.plot(X_test.iloc[i, 0:30], X_test.iloc[i, 30:60], '.', label="data_test")
    ax.plot(y_test.iloc[i, 0:4], y_test.iloc[i, 4:8], '.', label="target_test")
    ax.plot(predictions[i, 0:4], predictions[i, 4:8], '.', label="prediction_test")
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.xlabel('x')
    plt.ylabel('y')
    ax.legend()
    plt.show()
