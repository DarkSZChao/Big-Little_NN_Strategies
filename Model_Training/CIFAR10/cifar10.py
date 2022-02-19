# -- coding: utf-8 --
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

try:  # import for pycharm project directory
    from CIFAR10 import ROOTS
except:  # import for Ubuntu non-project directory
    import ROOTS


model_h5 = os.path.join(ROOTS.CIFAR10_output_model_path, 'cifar10.h5')

# Load CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# train_images = train_images[:, 0:24, :, :]
# test_images = test_images[:, 0:24, :, :]

# train_images_grey = np.zeros([50000, 32, 32, 1])
# train_images_grey[:, :, :, 0] = 0.3*train_images[:, :, :, 0] + 0.4*train_images[:, :, :, 1] + 0.3*train_images[:, :, :, 2]
# train_images_grey = train_images_grey.reshape(50000, 32, 32)
# test_images_grey = np.zeros([10000, 32, 32, 1])
# test_images_grey[:, :, :, 0] = 0.3*test_images[:, :, :, 0] + 0.4*test_images[:, :, :, 1] + 0.3*test_images[:, :, :, 2]
# test_images_grey = test_images_grey.reshape(10000, 32, 32)

# # preview the images
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()


# For command line TFLite converter
model_input = Input(shape=(32, 32, 3), name='model_input')
model = Conv2D(32, (3, 3), activation='relu', padding='same')(model_input)
model = MaxPooling2D(pool_size=(2, 2))(model)
model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)
model = MaxPooling2D(pool_size=(2, 2))(model)
model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)
model = MaxPooling2D(pool_size=(2, 2))(model)
model = Flatten()(model)
model = Dense(64)(model)
model_end = Dense(10, name='model_output')(model)
model = Model(model_input, model_end)

model.summary()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels,
                    validation_data=(test_images, test_labels),
                    epochs=10,
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
