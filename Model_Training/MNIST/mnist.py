# -- coding: utf-8 --
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Dense, Flatten, Reshape

try:  # import for pycharm project directory
    from MNIST import ROOTS
except:  # import for Ubuntu non-project directory
    import ROOTS


model_h5 = os.path.join(ROOTS.MNIST_output_model_path, 'mnist.h5')
conv_mode = '1D'  # '1D' or '2D'

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# # preview the images
# class_names = ['0', '1', '2', '3', '4',
#                '5', '6', '7', '8', '9']
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


if conv_mode == '1D':
    # For command line TFLite converter
    model_input = Input(shape=(28, 28), name='model_input')
    model = Conv1D(4, 3, activation='relu', padding='same')(model_input)
    model = MaxPooling1D(pool_size=2)(model)
    model = Conv1D(4, 3, activation='relu', padding='same')(model)
    model = MaxPooling1D(pool_size=2)(model)
    model = Conv1D(2, 3, activation='relu', padding='same')(model)
    model = MaxPooling1D(pool_size=2)(model)
    model = Flatten()(model)
    model_end = Dense(10, name='model_output')(model)

elif conv_mode == '2D':
    # train_images = train_images.reshape(60000, 28, 28, 1)
    # test_images = test_images.reshape(10000, 28, 28, 1)

    # For command line TFLite converter
    model_input = Input(shape=(28, 28), name='model_input')
    model = Reshape(target_shape=(28, 28, 1))(model_input)
    model = Conv2D(4, (3, 3), activation='relu', padding='same')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(4, (3, 3), activation='relu', padding='same')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(2, (3, 3), activation='relu', padding='same')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Flatten()(model)
    model_end = Dense(10, name='model_output')(model)
else:
    model_input = Input(shape=(28, 28), name='model_input')
    model_end = Dense(10, name='model_output')(model_input)

model = Model(model_input, model_end)

model.summary()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels,
                    validation_data=(test_images, test_labels),
                    epochs=50,
                    verbose=1)

model.save(model_h5)  # High_API h5 format

# Plot the accuracy and loss history
plt.figure()
plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
