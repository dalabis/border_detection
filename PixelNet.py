import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Reshape, Dropout, Flatten
from tensorflow.keras.models import Model
import numpy as np


def input_transform_net(point_cloud, dim=2):
    """ Input (XYZ) Transform Net, input is BxNxd gray image
            Return:
                Transformation matrix of size dxd """
    num_point = point_cloud.shape[1]

    input_image = tf.expand_dims(point_cloud, -1)
    net = Conv2D(64, (1, dim), strides=(1, 1), padding='valid')(input_image)
    net = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(net)
    net = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(net)
    net = MaxPooling2D(pool_size=(num_point, 1), padding='valid')(net)

    net = Flatten()(net)
    net = Dense(512)(net)
    net = Dense(256)(net)

    weights = tf.Variable(initial_value=tf.zeros([256, dim * dim]), dtype=tf.float32)
    biases = tf.Variable(initial_value=tf.zeros([dim * dim]), dtype=tf.float32)
    biases.assign_add(tf.reshape(tf.eye(dim), [dim * dim]))
    transform = tf.matmul(net, weights)
    transform = tf.nn.bias_add(transform, biases)

    transform = Reshape([dim, dim])(transform)

    return transform


def feature_transform_net(inputs, dim=64):
    """ Feature Transform Net, input is BxNx1xd
            Return:
                Transformation matrix of size dxd """
    num_point = inputs.shape[1]

    net = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(inputs)
    net = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(net)
    net = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(net)
    net = MaxPooling2D(pool_size=(num_point, 1), padding='valid')(net)

    net = Flatten()(net)
    net = Dense(512)(net)
    net = Dense(256)(net)

    weights = tf.Variable(initial_value=tf.zeros([256, dim * dim]), dtype=tf.float32)
    biases = tf.Variable(initial_value=tf.zeros([dim * dim]), dtype=tf.float32)
    biases.assign_add(tf.reshape(tf.eye(dim), [dim * dim]))
    transform = tf.matmul(net, weights)
    transform = tf.nn.bias_add(transform, biases)

    transform = Reshape([dim, dim])(transform)

    return transform


def model_pixel_net(num_point, dim, class_num):
    """ Classification PointNet, input is BxNxd, output Bxc """

    # input points
    inputs = Input(shape=(dim * num_point))
    point_cloud = Reshape([num_point, dim])(inputs)
    # input transform
    input_transform = input_transform_net(point_cloud, dim=2)
    point_cloud_transformed = tf.matmul(point_cloud, input_transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    # mlp (64, 64)
    net = Conv2D(64, (1, dim), strides=(1, 1), padding='valid')(input_image)
    net = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(net)

    # feature transform
    feature_transform = feature_transform_net(net, dim=64)
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), feature_transform)
    net_transformed = tf.expand_dims(net_transformed, 2)

    # mlp (64, 128, 1024)
    net = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(net_transformed)
    net = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(net)
    net = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(net)

    # Symmetric function: max pooling
    net = MaxPooling2D(pool_size=(num_point, 1), padding='valid')(net)

    # mlp (512, 64, class_size) / output scores
    net = Flatten()(net)
    net = Dense(512)(net)
    net = Dropout(0.7)(net)
    net = Dense(256)(net)
    net = Dropout(0.7)(net)
    outputs = Dense(class_num)(net)

    model = Model(inputs=inputs, outputs=outputs)

    return model
