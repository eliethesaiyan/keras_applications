import tensorflow as tf
from models.toy_model import make_toy_model



(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

filters = 64
kernel = (3,3)
strides = 2
model = make_toy_model(filters=filters, kernel=kernel, strides=strides)
model.compile(optimizer='sgd', loss='mse')
model.fit((X_train, y_train))
