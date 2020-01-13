import tensorflow as tf
from keras import backend as K

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

point_num = 30
batch_size = 64

data = pd.read_csv('input/rectangle_sets/rectangle_set_1000_30_points.csv', index_col=[0])
data_train, data_test = train_test_split(data, test_size=0.2)

X_columns = ['x_in_' + str(i) for i in range(point_num)] + ['y_in_' + str(i) for i in range(point_num)]
y_columns = ['height', 'width', 'center_x', 'center_y', 'rotation']

data_train['rotation'] = data_train['rotation'] / 180 * np.pi
data_test['rotation'] = data_test['rotation'] / 180 * np.pi

X = data_train[X_columns]
y = data_train[y_columns]
X_test = data_test[X_columns]
y_test = data_test[y_columns]
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
    tf.keras.layers.Dense(5),
])


def rectangle_transform_tensor(param):
    # This function returns the coordinates of the corners of the rectangle
    height = param[:, 0]
    width = param[:, 1]
    center_x = param[:, 2]
    center_y = param[:, 3]
    rotation = param[:, 4]

    # Coordinates without rotation
    # Anticlockwise order starting from the lower left corner
    x1 = center_x - width / 2
    x2 = center_x + width / 2
    x3 = center_x + width / 2
    x4 = center_x - width / 2
    y1 = center_y - height / 2
    y2 = center_y - height / 2
    y3 = center_y + height / 2
    y4 = center_y + height / 2

    # Clockwise rotation
    theta = rotation

    x1 = (x1 - center_x) * tf.cos(theta) - (y1 - center_y) * tf.sin(theta) + center_x
    x2 = (x2 - center_x) * tf.cos(theta) - (y2 - center_y) * tf.sin(theta) + center_x
    x3 = (x3 - center_x) * tf.cos(theta) - (y3 - center_y) * tf.sin(theta) + center_x
    x4 = (x4 - center_x) * tf.cos(theta) - (y4 - center_y) * tf.sin(theta) + center_x
    y1 = (x1 - center_x) * tf.sin(theta) + (y1 - center_y) * tf.cos(theta) + center_y
    y2 = (x2 - center_x) * tf.sin(theta) + (y2 - center_y) * tf.cos(theta) + center_y
    y3 = (x3 - center_x) * tf.sin(theta) + (y3 - center_y) * tf.cos(theta) + center_y
    y4 = (x4 - center_x) * tf.sin(theta) + (y4 - center_y) * tf.cos(theta) + center_y

    x1 = tf.reshape(x1, [-1, 1])
    x2 = tf.reshape(x2, [-1, 1])
    x3 = tf.reshape(x3, [-1, 1])
    x4 = tf.reshape(x4, [-1, 1])
    y1 = tf.reshape(y1, [-1, 1])
    y2 = tf.reshape(y2, [-1, 1])
    y3 = tf.reshape(y3, [-1, 1])
    y4 = tf.reshape(y4, [-1, 1])

    return tf.concat([x1, x2, x3, x4, y1, y2, y3, y4], 1)


def keras_l2_distance_8(tgt, pred):
    distance = np.zeros((tgt.shape[0], 4))
    tgt = rectangle_transform_tensor(tgt)
    pred = rectangle_transform_tensor(pred)

    for i in range(4):
        for j in range(4):
            distance[:, i] += tf.math.sqrt(
                (tgt[:, j] - pred[:, j]) ** 2 +
                (tgt[:, j + 4] - pred[:, j + 4]) ** 2,
                name=None)
        zeros = tf.zeros((tgt.shape[0], tgt.shape[1] - 1))
        rot = tf.concat([zeros, pred], 1)
        pred +=
        pred = rectangle_transform_tensor(pred)

    return tf.reduce_mean(tf.reduce_max(distance, axis=1))


def keras_l2_distance_max(tgt, pred):
    tgt = rectangle_transform_tensor(tgt)
    pred = rectangle_transform_tensor(pred)

    distance = tf.concat([
        tf.reshape(tf.math.sqrt((tgt[:, 0] - pred[:, 0]) ** 2 + (tgt[:, 4] - pred[:, 4]) ** 2, name=None), [-1, 1]),
        tf.reshape(tf.math.sqrt((tgt[:, 1] - pred[:, 1]) ** 2 + (tgt[:, 5] - pred[:, 5]) ** 2, name=None), [-1, 1]),
        tf.reshape(tf.math.sqrt((tgt[:, 2] - pred[:, 2]) ** 2 + (tgt[:, 6] - pred[:, 6]) ** 2, name=None), [-1, 1]),
        tf.reshape(tf.math.sqrt((tgt[:, 3] - pred[:, 3]) ** 2 + (tgt[:, 7] - pred[:, 7]) ** 2, name=None), [-1, 1]),
    ], 1)

    return tf.reduce_mean(tf.reduce_max(distance, axis=1))


def keras_l2_distance_5(tgt, pred):
    distance = np.zeros(tgt.shape[0])

    for i in range(5):
        distance += tgt[:, i] - pred[:, i]

    return tf.reduce_mean(distance)


model.compile(loss=keras_l2_distance_8, optimizer='adam')
model.summary()

print('# Fit model on training data')
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=50,
                    validation_data=(X_val, y_val))

print('\nhistory dict:', history.history)

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


def rectangle_transform(height, width, center_x, center_y, rotation):
    # This function returns the coordinates of the corners of the rectangle

    # Coordinates initialization
    x = np.zeros(4)
    y = np.zeros(4)

    # Coordinates without rotation
    # Anticlockwise order starting from the lower left corner
    x[0] = center_x - width / 2
    x[1] = center_x + width / 2
    x[2] = center_x + width / 2
    x[3] = center_x - width / 2
    y[0] = center_y - height / 2
    y[1] = center_y - height / 2
    y[2] = center_y + height / 2
    y[3] = center_y + height / 2

    x_rot = np.zeros(4)
    y_rot = np.zeros(4)

    # Coordinates with rotation
    # Clockwise rotation
    theta = rotation
    for i in range(4):
        x_rot[i] = (x[i] - center_x) * np.cos(theta) - \
                   (y[i] - center_y) * np.sin(theta) + center_x
        y_rot[i] = (x[i] - center_x) * np.sin(theta) + \
                   (y[i] - center_y) * np.cos(theta) + center_y

    return x_rot, y_rot

for i in range(3):
    plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)
    x_rect_test, y_rect_test = rectangle_transform(y_test.iloc[i, 0],
                                                   y_test.iloc[i, 1],
                                                   y_test.iloc[i, 2],
                                                   y_test.iloc[i, 3],
                                                   y_test.iloc[i, 4])
    x_rect_pred, y_rect_pred = rectangle_transform(predictions[i, 0],
                                                   predictions[i, 1],
                                                   predictions[i, 2],
                                                   predictions[i, 3],
                                                   predictions[i, 4])
    ax.plot(X_test.iloc[i, 0:30], X_test.iloc[i, 30:60], '.', label="data_test")
    ax.plot(x_rect_test, y_rect_test, '.', label="target_test")
    ax.plot(x_rect_pred, y_rect_pred, '.', label="prediction_test")
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.xlabel('x')
    plt.ylabel('y')
    ax.legend()
    plt.show()
