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
    from MOTION_Detector.load_dataset import load_single_motion_for_MOTION_Detector
except:  # import for Ubuntu non-project directory
    import ROOTS
    from load_dataset import load_data_for_MOTION_Detector
    from load_dataset import load_single_motion_for_MOTION_Detector


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
    activity = 0
    model_h5 = os.path.join(ROOTS.MOTION_Detector_output_model_path, 'S' + str(activity) + '_s3.h5')

    # train_data, train_labels, val_data, val_labels = load_data_for_MOTION_Detector(sensor='s3',
    #                                                                                quantization_to_0_255='on',
    #                                                                                label_categorical='off',
    #                                                                                current_activity='all')

    train_data_0, val_data_0 = load_single_motion_for_MOTION_Detector(sensor='s3', current_activity=0, data_range=[0, 255], data_type=np.int16)
    train_data_1, val_data_1 = load_single_motion_for_MOTION_Detector(sensor='s3', current_activity=1, data_range=[0, 255], data_type=np.int16)
    train_data_2, val_data_2 = load_single_motion_for_MOTION_Detector(sensor='s3', current_activity=2, data_range=[0, 255], data_type=np.int16)
    train_data_3, val_data_3 = load_single_motion_for_MOTION_Detector(sensor='s3', current_activity=3, data_range=[0, 255], data_type=np.int16)
    train_data_4, val_data_4 = load_single_motion_for_MOTION_Detector(sensor='s3', current_activity=4, data_range=[0, 255], data_type=np.int16)
    train_data_5, val_data_5 = load_single_motion_for_MOTION_Detector(sensor='s3', current_activity=5, data_range=[0, 255], data_type=np.int16)

    sample_num_main = 1225
    sample_list = random.sample([i for i in range(1226)], sample_num_main)
    train_data_0 = train_data_0[sample_list]

    sample_num_sub = int(sample_num_main / 5)
    sample_list = random.sample([i for i in range(1073)], sample_num_sub)
    train_data_1 = train_data_1[sample_list]
    sample_list = random.sample([i for i in range(986)], sample_num_sub)
    train_data_2 = train_data_2[sample_list]
    sample_list = random.sample([i for i in range(1286)], sample_num_sub)
    train_data_3 = train_data_3[sample_list]
    sample_list = random.sample([i for i in range(1374)], sample_num_sub)
    train_data_4 = train_data_4[sample_list]
    sample_list = random.sample([i for i in range(1407)], sample_num_sub)
    train_data_5 = train_data_5[sample_list]

    train_label_0 = np.ones(sample_num_main)
    train_label_1 = np.zeros(sample_num_sub)
    train_label_2 = np.zeros(sample_num_sub)
    train_label_3 = np.zeros(sample_num_sub)
    train_label_4 = np.zeros(sample_num_sub)
    train_label_5 = np.zeros(sample_num_sub)

    train_data_merged = np.concatenate([train_data_0, train_data_1, train_data_2, train_data_3, train_data_4, train_data_5])
    train_labels_merged = np.concatenate([train_label_0, train_label_1, train_label_2, train_label_3, train_label_4, train_label_5])

    sample_num_main = 495
    sample_list = random.sample([i for i in range(496)], sample_num_main)
    val_data_0 = val_data_0[sample_list]

    sample_num_sub = int(sample_num_main / 5)
    sample_list = random.sample([i for i in range(471)], sample_num_sub)
    val_data_1 = val_data_1[sample_list]
    sample_list = random.sample([i for i in range(420)], sample_num_sub)
    val_data_2 = val_data_2[sample_list]
    sample_list = random.sample([i for i in range(491)], sample_num_sub)
    val_data_3 = val_data_3[sample_list]
    sample_list = random.sample([i for i in range(532)], sample_num_sub)
    val_data_4 = val_data_4[sample_list]
    sample_list = random.sample([i for i in range(537)], sample_num_sub)
    val_data_5 = val_data_5[sample_list]

    val_label_0 = np.ones(sample_num_main)
    val_label_1 = np.zeros(sample_num_sub)
    val_label_2 = np.zeros(sample_num_sub)
    val_label_3 = np.zeros(sample_num_sub)
    val_label_4 = np.zeros(sample_num_sub)
    val_label_5 = np.zeros(sample_num_sub)

    val_data_merged = np.concatenate([val_data_0, val_data_1, val_data_2, val_data_3, val_data_4, val_data_5])
    val_labels_merged = np.concatenate([val_label_0, val_label_1, val_label_2, val_label_3, val_label_4, val_label_5])

    model = model_structure()
    model.summary()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(train_data_merged, train_labels_merged,
                        validation_data=(val_data_merged, val_labels_merged),
                        shuffle=True,
                        epochs=100,
                        verbose=1)

    model.save(model_h5)  # High_API h5 format

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
