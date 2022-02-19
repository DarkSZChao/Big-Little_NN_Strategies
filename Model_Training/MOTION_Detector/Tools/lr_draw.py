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
    _model_input = Input(shape=(128, 3), name='model_input')
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
    activity = 5
    model_h5 = os.path.join(ROOTS.MOTION_Detector_output_model_path, 'single_motion_' + str(activity) + '.h5')

    train_data, train_labels, val_data, val_labels = load_data_for_MOTION_Detector(sensor='s3',  # when sensor='all', the function return dictionary type which includes s1, s2, s3
                                                                                   current_activity=activity,  # from 0 to 5
                                                                                   data_range=[0, 255],
                                                                                   data_type=np.uint8,
                                                                                   label_categorical='off')
    train_data = train_data.astype(np.int16) - 128
    val_data = val_data.astype(np.int16) - 128

    model = model_structure()
    model.summary()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])

    lr_szc = []
    lr_szc2 = []

    def scheduler(epoch, lr):
        if epoch < 2:
            lr = 0.005
            lr2 = 0.005
        elif epoch == 20:
            lr = 0.003
            lr2 = 0.005
        else:
            # lr = lr * 0.95
            lr = lr * math.exp(-lr * 50)
            lr2 = 0.005
        lr_szc.append(lr)
        lr_szc2.append(lr2)
        return lr


    lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)

    checkpoint_path = os.path.join(ROOTS.MOTION_Detector_output_model_path, 'checkpoint/cp-{epoch:02d}.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True)

    history = model.fit(train_data, train_labels,
                        validation_data=(val_data, val_labels),
                        callbacks=[lr_callback, cp_callback],
                        epochs=50,
                        verbose=1)

    model.save(model_h5)  # High_API h5 format

    print('Epoch with maximum val_accuracy:', np.argmax(history.history['val_acc']) + 1, history.history['val_acc'][np.argmax(history.history['val_acc'])] * 100, '%')

    plt.figure()
    plt.plot(lr_szc, label='Ours')
    plt.plot(lr_szc2, label='Default')
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.ylim([0, 0.006])
    plt.legend(loc='lower right')
    plt.show()
