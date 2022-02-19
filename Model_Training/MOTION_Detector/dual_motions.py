# -- coding: utf-8 --
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten

try:  # import for pycharm project directory
    from MOTION_Detector import ROOTS
    from MOTION_Detector.load_dataset import load_data_for_MOTION_Detector
except:  # import for Ubuntu non-project directory
    import ROOTS
    from load_dataset import load_data_for_MOTION_Detector


def model_structure():
    # for command line TFLite converter
    _model_input = Input(shape=(384, 2), name='model_input')
    _model = Conv1D(4, 3, activation='relu', padding='same')(_model_input)
    _model = MaxPooling1D(pool_size=2)(_model)
    _model = Conv1D(4, 3, activation='relu', padding='same')(_model)
    _model = MaxPooling1D(pool_size=2)(_model)
    _model = Conv1D(2, 3, activation='relu', padding='same')(_model)
    _model = MaxPooling1D(pool_size=2)(_model)
    _model = Flatten()(_model)
    _model = Dense(2, name='model_output')(_model)
    _model = Model(_model_input, _model)
    return _model


if __name__ == '__main__':
    model_h5 = os.path.join(ROOTS.MOTION_Detector_output_model_path, 'dual_motions.h5')

    train_data, train_labels, val_data, val_labels = load_data_for_MOTION_Detector(sensor='s3',  # when sensor='all', the function return dictionary type which includes s1, s2, s3
                                                                                   current_activity='all',  # from 0 to 5
                                                                                   data_range=[-128, 127],
                                                                                   data_type=np.int16,
                                                                                   label_categorical='off')
    train_data = train_data.reshape(-1, 384)
    val_data = val_data.reshape(-1, 384)

    train_data_dual = np.zeros([len(train_labels) - 1, 384, 2]).astype(np.int16)
    train_labels_dual = np.zeros(len(train_labels) - 1).astype(np.int16)
    for i in range(len(train_labels) - 1):
        # extract dual training data, form it into 384*2
        sub_data = np.column_stack([train_data[i], train_data[i + 1]])
        train_data_dual[i] = sub_data
        # extract dual training label
        if train_labels[i] == train_labels[i + 1]:
            train_labels_dual[i] = 1
        else:
            train_labels_dual[i] = 0

    val_data_dual = np.zeros([len(val_labels) - 1, 384, 2]).astype(np.int16)
    val_labels_dual = np.zeros(len(val_labels) - 1).astype(np.int16)
    for i in range(len(val_labels) - 1):
        # extract dual validation data, form it into 384*2
        sub_data = np.column_stack([val_data[i], val_data[i + 1]])
        val_data_dual[i] = sub_data
        # extract dual validation label
        if val_labels[i] == val_labels[i + 1]:
            val_labels_dual[i] = 1
        else:
            val_labels_dual[i] = 0

    model = model_structure()
    model.summary()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])


    def scheduler(epoch, lr):
        if epoch < 2:
            lr = 0.005
        # elif epoch == 20:
        #     lr = 0.001
        else:
            # lr = lr * 0.95
            lr = lr * math.exp(-lr * 50)
        return lr


    lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)

    checkpoint_path = os.path.join(ROOTS.MOTION_Detector_output_model_path, 'checkpoint/cp-{epoch:02d}.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True)

    history = model.fit(train_data_dual, train_labels_dual,
                        validation_data=(val_data_dual, val_labels_dual),
                        callbacks=[lr_callback, cp_callback],
                        epochs=50,
                        verbose=1)

    model.save(model_h5)  # High_API h5 format

    print('Epoch with maximum val_accuracy:', np.argmax(history.history['val_acc']) + 1, history.history['val_acc'][np.argmax(history.history['val_acc'])] * 100, '%')

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
