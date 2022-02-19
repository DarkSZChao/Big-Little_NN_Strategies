# -- coding: utf-8 --

import os
import time

import numpy as np
import tensorflow as tf

from Tools import ROOTS_Tools
from Tools.load_dataset import load_data_for_Tools


def h5_validation(_model, _input_data):
    if ROOTS_Tools.SENSOR == 'all':
        _output_data = _model.predict([_input_data['s1'], _input_data['s2'], _input_data['s3']])
    else:
        _output_data = _model.predict(_input_data)

    return _output_data.reshape(-1)


if __name__ == '__main__':
    # Load the h5 model
    model_h5 = os.path.join(ROOTS_Tools.output_model_path, 'S4_s3.h5')
    model = tf.keras.models.load_model(model_h5)

    data, labels = load_data_for_Tools()
    example_set = range(len(labels))

    accuracy = 0
    time_start = time.time()
    if ROOTS_Tools.SENSOR == 'all':  # for MOTION_Detector model with 3 sensor inputs
        for example in example_set:
            input_data = {'s1': data['s1'][[example]], 's2': data['s2'][[example]], 's3': data['s3'][[example]]}
            output_data = h5_validation(model, input_data)
            output_label = np.argmax(output_data)

            print(output_data)
            print('Output Label:', output_label)
            print('Expect label:', labels[example], '\n')

            if output_label == labels[example]:
                accuracy += 1
    else:
        for example in example_set:
            input_data = data[[example]]
            output_data = h5_validation(model, input_data)
            output_label = np.argmax(output_data)

            # print(output_data)
            # print('Output Label:', output_label)
            # print('Expect label:', labels[example], '\n')

            if output_label == labels[example]:
                accuracy += 1
    time_end = time.time()

    print('Accuracy:', str(accuracy / len(example_set) * 100), '%')
    print('Time cost:', time_end - time_start, 's')
