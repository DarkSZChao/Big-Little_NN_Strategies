# -- coding: utf-8 --
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

try:  # import for pycharm project directory
    from CIFAR10 import ROOTS
except:  # import for Ubuntu non-project directory
    import ROOTS

model_h5 = os.path.join(ROOTS.CIFAR10_output_model_path, 'cifar10_airplane.h5')

# Load CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# train_images = train_images / 255.0  # Normalize pixel values to be between 0 and 1
# test_images = test_images / 255.0  # Normalize pixel values to be between 0 and 1

train_images_10_90 = train_images
train_labels_10_90 = train_labels
test_images_10_90 = test_images
test_labels_10_90 = test_labels
# train_labels_10_90[train_labels_10_90 > 0] = 1
# test_labels_10_90[test_labels_10_90 > 0] = 1

train_images_airplane = train_images[(train_labels == 0).reshape(-1)]
train_labels_airplane = train_labels[(train_labels == 0).reshape(-1)]
train_images_automobile = train_images[(train_labels == 1).reshape(-1)]
train_labels_automobile = train_labels[(train_labels == 1).reshape(-1)]
train_images_bird = train_images[(train_labels == 2).reshape(-1)]
train_labels_bird = train_labels[(train_labels == 2).reshape(-1)]
train_images_cat = train_images[(train_labels == 3).reshape(-1)]
train_labels_cat = train_labels[(train_labels == 3).reshape(-1)]
train_images_deer = train_images[(train_labels == 4).reshape(-1)]
train_labels_deer = train_labels[(train_labels == 4).reshape(-1)]
train_images_dog = train_images[(train_labels == 5).reshape(-1)]
train_labels_dog = train_labels[(train_labels == 5).reshape(-1)]
train_images_frog = train_images[(train_labels == 6).reshape(-1)]
train_labels_frog = train_labels[(train_labels == 6).reshape(-1)]
train_images_horse = train_images[(train_labels == 7).reshape(-1)]
train_labels_horse = train_labels[(train_labels == 7).reshape(-1)]
train_images_ship = train_images[(train_labels == 8).reshape(-1)]
train_labels_ship = train_labels[(train_labels == 8).reshape(-1)]
train_images_truck = train_images[(train_labels == 9).reshape(-1)]
train_labels_truck = train_labels[(train_labels == 9).reshape(-1)]
train_images_non_airplane = np.concatenate([train_images_automobile, train_images_bird, train_images_cat, train_images_deer,
                                            train_images_dog, train_images_frog, train_images_horse, train_images_ship, train_images_truck])
train_labels_non_airplane = np.concatenate([train_labels_automobile, train_labels_bird, train_labels_cat, train_labels_deer,
                                            train_labels_dog, train_labels_frog, train_labels_horse, train_labels_ship, train_labels_truck])

sample_num = 5000
sample_list = random.sample([i for i in range(45000)], sample_num)
train_images_non_airplane = train_images_non_airplane[sample_list]
train_labels_non_airplane = train_labels_non_airplane[sample_list]
train_images_50_50 = np.concatenate([train_images_airplane, train_images_non_airplane])
train_labels_50_50 = np.concatenate([train_labels_airplane, train_labels_non_airplane])

sample_num = 10000
sample_list = random.sample([i for i in range(10000)], sample_num)
train_images_50_50 = train_images_50_50[sample_list]
train_labels_50_50 = train_labels_50_50[sample_list]
train_labels_50_50[train_labels_50_50 > 0] = 1

test_images_airplane = test_images[(test_labels == 0).reshape(-1)]
test_labels_airplane = test_labels[(test_labels == 0).reshape(-1)]
test_images_automobile = test_images[(test_labels == 1).reshape(-1)]
test_labels_automobile = test_labels[(test_labels == 1).reshape(-1)]
test_images_bird = test_images[(test_labels == 2).reshape(-1)]
test_labels_bird = test_labels[(test_labels == 2).reshape(-1)]
test_images_cat = test_images[(test_labels == 3).reshape(-1)]
test_labels_cat = test_labels[(test_labels == 3).reshape(-1)]
test_images_deer = test_images[(test_labels == 4).reshape(-1)]
test_labels_deer = test_labels[(test_labels == 4).reshape(-1)]
test_images_dog = test_images[(test_labels == 5).reshape(-1)]
test_labels_dog = test_labels[(test_labels == 5).reshape(-1)]
test_images_frog = test_images[(test_labels == 6).reshape(-1)]
test_labels_frog = test_labels[(test_labels == 6).reshape(-1)]
test_images_horse = test_images[(test_labels == 7).reshape(-1)]
test_labels_horse = test_labels[(test_labels == 7).reshape(-1)]
test_images_ship = test_images[(test_labels == 8).reshape(-1)]
test_labels_ship = test_labels[(test_labels == 8).reshape(-1)]
test_images_truck = test_images[(test_labels == 9).reshape(-1)]
test_labels_truck = test_labels[(test_labels == 9).reshape(-1)]
test_images_non_airplane = np.concatenate([test_images_automobile, test_images_bird, test_images_cat, test_images_deer,
                                           test_images_dog, test_images_frog, test_images_horse, test_images_ship, test_images_truck])
test_labels_non_airplane = np.concatenate([test_labels_automobile, test_labels_bird, test_labels_cat, test_labels_deer,
                                           test_labels_dog, test_labels_frog, test_labels_horse, test_labels_ship, test_labels_truck])

sample_num = 1000
sample_list = random.sample([i for i in range(9000)], sample_num)
test_images_non_airplane = test_images_non_airplane[sample_list]
test_labels_non_airplane = test_labels_non_airplane[sample_list]
test_images_50_50 = np.concatenate([test_images_airplane, test_images_non_airplane])
test_labels_50_50 = np.concatenate([test_labels_airplane, test_labels_non_airplane])

sample_num = 2000
sample_list = random.sample([i for i in range(2000)], sample_num)
test_images_50_50 = test_images_50_50[sample_list]
test_labels_50_50 = test_labels_50_50[sample_list]
test_labels_50_50[test_labels_50_50 > 0] = 1

# preview the images
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']
# plt.figure(figsize=(10, 10))
# for i in range(30):
#     plt.subplot(5, 6, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_images_10_90[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[test_labels_10_90[i][0]])
# plt.show()


model = models.Sequential([
    layers.Conv2D(4, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(8, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(8, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(2, activation='relu'),
])

model.summary()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_images_50_50, train_labels_50_50,
                    validation_data=(test_images_50_50, test_labels_50_50),
                    epochs=20,
                    shuffle=True,
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
