# -- coding: utf-8 --
import os
import time

import numpy as np
import tensorflow as tf

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


def S_tflite_validation(_interpreter, _input_details, _output_details, _input_data, _output_label_B):
    # Set wanted type to input data
    _interpreter.set_tensor(_input_details[0]['index'], _input_data)

    _interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    _output_data = _interpreter.get_tensor(_output_details[0]['index'])
    _output_data = _output_data.reshape(-1)

    _output_label_S = np.argmax(_output_data)

    # print(_output_data)
    if _output_label_S == 1:
        _BIG_model_activate = False
        _estimate_label = _output_label_B
        # print('Output Label remains:', _output_label_B)
    else:
        _BIG_model_activate = True
        _estimate_label = 7
        # print('Output Label changes, Invoke BIG')
    # print('Expect label:', train_labels[example], '\n')

    return _BIG_model_activate, _estimate_label


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
    multi_motions = os.path.join(ROOTS.MOTION_Detector_output_model_path, 'Best_model_-128127/B_s123_3IN_91.tflite')
    single_motion_0 = os.path.join(ROOTS.MOTION_Detector_output_model_path, 'Best_model_NNoM/S0_s3.tflite')
    single_motion_1 = os.path.join(ROOTS.MOTION_Detector_output_model_path, 'Best_model_NNoM/S1_s3.tflite')
    single_motion_2 = os.path.join(ROOTS.MOTION_Detector_output_model_path, 'Best_model_NNoM/S2_s3.tflite')
    single_motion_3 = os.path.join(ROOTS.MOTION_Detector_output_model_path, 'Best_model_NNoM/S3_s3.tflite')
    single_motion_4 = os.path.join(ROOTS.MOTION_Detector_output_model_path, 'Best_model_NNoM/S4_s3.tflite')
    single_motion_5 = os.path.join(ROOTS.MOTION_Detector_output_model_path, 'Best_model_NNoM/S5_s3.tflite')

    multi_motions_interpreter, multi_motions_input, multi_motions_output = load_tflite_interpreter(multi_motions)
    single_motion_0_interpreter, single_motion_0_input, single_motion_0_output = load_tflite_interpreter(single_motion_0)
    single_motion_1_interpreter, single_motion_1_input, single_motion_1_output = load_tflite_interpreter(single_motion_1)
    single_motion_2_interpreter, single_motion_2_input, single_motion_2_output = load_tflite_interpreter(single_motion_2)
    single_motion_3_interpreter, single_motion_3_input, single_motion_3_output = load_tflite_interpreter(single_motion_3)
    single_motion_4_interpreter, single_motion_4_input, single_motion_4_output = load_tflite_interpreter(single_motion_4)
    single_motion_5_interpreter, single_motion_5_input, single_motion_5_output = load_tflite_interpreter(single_motion_5)

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

    time_start = time.time()
    for example in example_set:
        input_data_s1 = val_data['s1'][[example]]
        input_data_s2 = val_data['s2'][[example]]
        input_data_s3 = val_data['s3'][[example]]
        if not BIG_model_activate:
            if output_label_B == 0:
                BIG_model_activate, estimate_label = S_tflite_validation(single_motion_0_interpreter, single_motion_0_input, single_motion_0_output, input_data_s3, output_label_B)
            elif output_label_B == 1:
                BIG_model_activate, estimate_label = S_tflite_validation(single_motion_1_interpreter, single_motion_1_input, single_motion_1_output, input_data_s3, output_label_B)
            elif output_label_B == 2:
                BIG_model_activate, estimate_label = S_tflite_validation(single_motion_2_interpreter, single_motion_2_input, single_motion_2_output, input_data_s3, output_label_B)
            elif output_label_B == 3:
                BIG_model_activate, estimate_label = S_tflite_validation(single_motion_3_interpreter, single_motion_3_input, single_motion_3_output, input_data_s3, output_label_B)
            elif output_label_B == 4:
                BIG_model_activate, estimate_label = S_tflite_validation(single_motion_4_interpreter, single_motion_4_input, single_motion_4_output, input_data_s3, output_label_B)
            elif output_label_B == 5:
                BIG_model_activate, estimate_label = S_tflite_validation(single_motion_5_interpreter, single_motion_5_input, single_motion_5_output, input_data_s3, output_label_B)
            count_S += 1
            if estimate_label == val_labels[example]:
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
