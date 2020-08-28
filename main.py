import tensorflow as tf
import matplotlib.pyplot as plt
from models.toy_model import make_sequential_model
from models.toy_model import make_functional_model




(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
#y_train = tf.keras.utils.to_categorical(y_train, dtype='float32')
#y_test = tf.keras.utils.to_categorical(y_test, dtype='float32')

print('shapes')
print(X_train.shape)
print(y_train[:5])
filters = 32
kernel = (3, 3)
# model = make_sequential_model(filters=filters, kernel=kernel, strides=1)
# model = make_sequential_model()
model = make_functional_model()
model.summary()
model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics='accuracy')
history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                    validation_split=0.25)
eval_score = model.evaluate(X_test, y_test)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print('evaluation score %.2f', eval_score)
predict_score = model.predict(X_test[:10])
print('{}, {}, {}, {}, {}, {},{},{}, {}, {}'.format(*[class_names[i] for i in
                                           tf.keras.backend.argmax(predict_score)]))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
