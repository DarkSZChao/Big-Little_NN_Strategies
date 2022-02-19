# -- coding: utf-8 --

import os

import tensorflow as tf

from Tools import ROOTS_Tools

model_tflite = os.path.join(ROOTS_Tools.output_model_path, 'multi_motions_s3.tflite')

# Use `tf.lite.Interpreter` to load the written .tflite back from the file system.
interpreter = tf.lite.Interpreter(model_path=model_tflite)
all_tensor_details = interpreter.get_tensor_details()
interpreter.allocate_tensors()

for tensor_item in all_tensor_details:
    print("Weight %s:" % tensor_item["name"])
    print(interpreter.tensor(tensor_item["index"])())
