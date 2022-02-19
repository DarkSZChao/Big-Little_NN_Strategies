# -- coding: utf-8 --

import os
import time

import numpy as np
import tensorflow as tf

from Tools import ROOTS_Tools
from Tools.load_dataset import load_data_for_Tools


def load_tflite_interpreter(_model_path):
    # Load the TFLite model and allocate tensors.
    _interpreter = tf.lite.Interpreter(model_path=_model_path)
    _interpreter.allocate_tensors()
    # Get input and output tensors.
    _input_details = _interpreter.get_input_details()
    _output_details = _interpreter.get_output_details()

    return _interpreter, _input_details, _output_details


def tflite_validation(_interpreter, _input_details, _output_details, _input_data):
    # Set wanted type to input data
    if ROOTS_Tools.SENSOR == 'all':
        _interpreter.set_tensor(_input_details[0]['index'], _input_data['s1'])
        _interpreter.set_tensor(_input_details[1]['index'], _input_data['s2'])
        _interpreter.set_tensor(_input_details[2]['index'], _input_data['s3'])
    else:
        _interpreter.set_tensor(_input_details[0]['index'], _input_data)

    _interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    _output_data = _interpreter.get_tensor(_output_details[0]['index'])

    return _output_data.reshape(-1)


if __name__ == '__main__':
    # Load the tflite model
    model_tflite = os.path.join(ROOTS_Tools.output_model_path, 'Best_model_NNoM/S4_s3.tflite')
    tflite_interpreter, tflite_input_details, tflite_output_details = load_tflite_interpreter(model_tflite)

    data, labels = load_data_for_Tools()
    example_set = range(len(labels))
    # example_set = np.arange(51, 77)
    # example_set = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    accuracy = 0
    time_start = time.time()
    if ROOTS_Tools.SENSOR == 'all':  # for MOTION_Detector model with 3 sensor inputs
        for example in example_set:
            input_data = {'s1': data['s1'][[example]], 's2': data['s2'][[example]], 's3': data['s3'][[example]]}
            output_data = tflite_validation(tflite_interpreter, tflite_input_details, tflite_output_details, input_data)
            output_label = np.argmax(output_data)

            print(output_data)
            print('Output Label:', output_label)
            print('Expect label:', labels[example], '\n')

            if output_label == labels[example]:
                accuracy += 1
    else:
        for example in example_set:
            input_data = data[[example]]
            output_data = tflite_validation(tflite_interpreter, tflite_input_details, tflite_output_details, input_data)
            output_label = np.argmax(output_data)

            # print(output_data)
            # print('Output Label:', output_label)
            # print('Expect label:', labels[example], '\n')

            if output_label == labels[example]:
                accuracy += 1
    time_end = time.time()

    print('Accuracy:', accuracy / len(example_set) * 100, '%')
    print('Time cost:', time_end - time_start, 's')
