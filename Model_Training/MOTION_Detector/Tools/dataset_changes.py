# -- coding: utf-8 --
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten

from MOTION_Detector import ROOTS
from MOTION_Detector.load_dataset import load_data_for_MOTION_Detector
from MOTION_Detector.load_dataset import load_single_motion_for_MOTION_Detector

if __name__ == '__main__':
    _, train_labels, _, val_labels = load_data_for_MOTION_Detector(sensor='s3',  # when sensor='all', the function return dictionary type which includes s1, s2, s3
                                                                   current_activity='all',  # from 0 to 5
                                                                   data_range=[0, 255],
                                                                   data_type=np.uint8,
                                                                   label_categorical='off')
    counter = 0
    current_class = 0
    for i in range(len(train_labels)):
        current_label = train_labels[i]
        if current_label != current_class:
            counter += 1
            current_class = current_label

    print(counter)
