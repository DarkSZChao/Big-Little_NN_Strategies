# -- coding: utf-8 --

import os

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.split(ROOT_DIR)[0]

CURRENT_MODEL = 'MOTION_Detector'
# CURRENT_MODEL = 'CIFAR10'
# CURRENT_MODEL = 'MNIST'

output_dataset_path = os.path.join(ROOT_DIR, CURRENT_MODEL, 'Output_Dataset')
output_model_path = os.path.join(ROOT_DIR, CURRENT_MODEL, 'Output_Models')

if CURRENT_MODEL == 'MOTION_Detector':
    SENSOR = 'all'
    # SENSOR = 's3'

    ACTIVITY = 'all'
    # ACTIVITY = 4

else:
    SENSOR = None
    ACTIVITY = None

# DATASET = 'train'
DATASET = 'val'

DATASET_Drange = [0, 255]
# DATASET_Drange = [-128, 127]

DATASET_Dtype = np.uint8
# DATASET_Dtype = np.int16
