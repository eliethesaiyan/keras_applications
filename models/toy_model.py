import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import Model


def _conv_block(inputs, filters, kernel, strides):
    '''
    :param filters: output filters
    :param kernel: kernel used in convolution
    :param strides: strides used for convolution
    :return : x : activation of the convolution block
    '''
    x = layers.Conv2D(filters, kernel_size=kernel, strides=strides)(inputs)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)

    return x


def make_toy_model(filters, kernel, strides ):

    inputs = layers.Input(shape=(None, None, 3))
    x = _conv_block(inputs, filters, kernel, strides)
    output = layers.Dense(10, activation='relu')(x)
    model = Model(inputs, output)

    return model


