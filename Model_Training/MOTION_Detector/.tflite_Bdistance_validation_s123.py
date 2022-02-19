# -- coding: utf-8 --
import os
import time

import numpy as np
import tensorflow as tf
import scipy.spatial.distance as dist
from matplotlib import pyplot as plt

from MOTION_Detector import ROOTS
from MOTION_Detector.load_dataset import load_data_for_MOTION_Detector


def load_tflite_interpreter(_model_path):
    # Load the TFLite model and allocate tensors.
    _interpreter = tf.lite.Interpreter(model_path=_model_path)
    _interpreter.allocate_tensors()
    # Get input and output tensors.
    _input_details = _interpreter.get_input_details()
    _output_details = _interpreter.get_output_details()

    return _interpreter, _input_details, _output_details


def data_distance_diff(_previous_data, _next_data, _p, _covMat, Mahalanobis_distance=True):
    _previous_data = _previous_data.reshape(-1)
    _next_data = _next_data.reshape(-1)

    if Mahalanobis_distance:
        _distance_diff = dist.mahalanobis(_previous_data, _next_data, _covMat)
    else:
        a = _previous_data.astype(int)
        b = _next_data.astype(int)
        _distance_diff = pow(sum(pow(abs(a - b), _p)), 1 / _p)

    return int(_distance_diff)


def B_tflite_validation(_interpreter, _input_details, _output_details, _input_data_s1, _input_data_s2, _input_data_s3):
    # Set wanted type to input data
    _interpreter.set_tensor(_input_details[0]['index'], _input_data_s1)
    _interpreter.set_tensor(_input_details[1]['index'], _input_data_s2)
    _interpreter.set_tensor(_input_details[2]['index'], _input_data_s3)

    _interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    _output_data = _interpreter.get_tensor(_output_details[0]['index'])
    _output_data = _output_data.reshape(-1)

    _output_label_B = np.argmax(_output_data)

    # print(_output_data)
    # print('Output Label:', _output_label_B)
    # print('Expect label:', train_labels[example], '\n')

    _BIG_model_activate = False

    return _BIG_model_activate, _output_label_B


if __name__ == '__main__':
    multi_motions = os.path.join(ROOTS.MOTION_Detector_output_model_path, 'Best_model_-128127/multi_motions_s123_91.tflite')

    multi_motions_interpreter, multi_motions_input, multi_motions_output = load_tflite_interpreter(multi_motions)

    train_data, train_labels, val_data, val_labels = load_data_for_MOTION_Detector(sensor='all',  # when sensor='all', the function return dictionary type which includes s1, s2, s3
                                                                                   current_activity='all',  # from 0 to 5
                                                                                   data_range=[0, 255],
                                                                                   data_type=np.uint8,
                                                                                   label_categorical='off')
    example_set = range(len(val_labels))
    # example_set = [0, 1, 2, 100, 101, 102, 300, 301, 302, 1800, 1801, 1802, 500, 501, 502, 600, 601, 602]

    accuracy = 0
    output_label_B = 0
    BIG_model_activate = True
    count_S = 0
    count_B = 0
    distance_diff = []

    train_data_s3 = val_data['s3'].reshape(-1, 384).transpose(1, 0)
    covMat = np.matrix(np.cov(train_data_s3))

    time_start = time.time()
    for example in example_set:
        input_data_s1 = val_data['s1'][[example]]
        input_data_s2 = val_data['s2'][[example]]
        input_data_s3 = val_data['s3'][[example]]
        if not BIG_model_activate:

            if example == 0:
                previous_data = np.zeros([1, 128, 3]).astype(np.uint8)
            else:
                previous_data = val_data['s3'][[example - 1]]
            next_data = val_data['s3'][[example]]
            distance_diff = (data_distance_diff(previous_data, next_data, 1, covMat, Mahalanobis_distance=False))

            if distance_diff > 2000:
                output_label_B = 7
                BIG_model_activate = True
            else:
                output_label_B = output_label_B
                BIG_model_activate = False

            count_S += 1
            if output_label_B == val_labels[example]:
                accuracy += 1

        if BIG_model_activate:
            BIG_model_activate, output_label_B = B_tflite_validation(multi_motions_interpreter, multi_motions_input, multi_motions_output, input_data_s1, input_data_s2, input_data_s3)
            count_B += 1
            if output_label_B == val_labels[example]:
                accuracy += 1

    time_end = time.time()
    print('Accuracy:', accuracy / len(example_set) * 100, '%')
    print('Time cost:', time_end - time_start, 's')
    print('SMALL Inference times:', count_S)
    print('BIG Inference times:', count_B)
