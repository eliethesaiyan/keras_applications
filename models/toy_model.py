import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import models


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
    output = layers.Dense(10, activation='relu')(x)

    return output


def make_functional_model():

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(rate=0.3)(x)
    output = layers.Dense(10)(x)
    model = models.Model(inputs=inputs, outputs=output)
    model.build(input_shape=(32, 32, 3))

    return model


def make_sequential_model():

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    return model