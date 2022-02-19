# -- coding: utf-8 --
import numpy as np
from tensorflow.keras import datasets

from MOTION_Detector.load_dataset import load_data_for_MOTION_Detector
from Tools import ROOTS_Tools


# import data and label for Tool scripts
def load_data_for_Tools(current_model=ROOTS_Tools.CURRENT_MODEL, dataset=ROOTS_Tools.DATASET, data_range=ROOTS_Tools.DATASET_Drange, data_type=ROOTS_Tools.DATASET_Dtype):
    if current_model == 'MOTION_Detector':
        _train_data, _train_labels, _val_data, _val_labels = load_data_for_MOTION_Detector(sensor=ROOTS_Tools.SENSOR,  # CURRENT_MODEL can be 'MOTION_Detector', 'CIFAR10' or 'MNIST'
                                                                                           data_range=[0, 255],
                                                                                           data_type=np.float64,
                                                                                           label_categorical='off',
                                                                                           current_activity=ROOTS_Tools.ACTIVITY)
        try:  # for the case of using one sensor
            _train_data = (_train_data / 255 * (data_range[1] - data_range[0]) + data_range[0]).astype(data_type)
            _val_data = (_val_data / 255 * (data_range[1] - data_range[0]) + data_range[0]).astype(data_type)
        except:
            _train_data['s1'] = (_train_data['s1'] / 255 * (data_range[1] - data_range[0]) + data_range[0]).astype(data_type)
            _train_data['s2'] = (_train_data['s2'] / 255 * (data_range[1] - data_range[0]) + data_range[0]).astype(data_type)
            _train_data['s3'] = (_train_data['s3'] / 255 * (data_range[1] - data_range[0]) + data_range[0]).astype(data_type)
            _val_data['s1'] = (_val_data['s1'] / 255 * (data_range[1] - data_range[0]) + data_range[0]).astype(data_type)
            _val_data['s2'] = (_val_data['s2'] / 255 * (data_range[1] - data_range[0]) + data_range[0]).astype(data_type)
            _val_data['s3'] = (_val_data['s3'] / 255 * (data_range[1] - data_range[0]) + data_range[0]).astype(data_type)

    elif current_model == 'CIFAR10':
        (_train_data, _train_labels), (_val_data, _val_labels) = datasets.cifar10.load_data()
        _train_labels = _train_labels.reshape(-1)
        _val_labels = _val_labels.reshape(-1)
        _train_data = (_train_data / 255 * (data_range[1] - data_range[0]) + data_range[0]).astype(data_type)
        _val_data = (_val_data / 255 * (data_range[1] - data_range[0]) + data_range[0]).astype(data_type)

    elif current_model == 'MNIST':
        (_train_data, _train_labels), (_val_data, _val_labels) = datasets.mnist.load_data()
        _train_data = (_train_data / 255 * (data_range[1] - data_range[0]) + data_range[0]).astype(data_type)
        _val_data = (_val_data / 255 * (data_range[1] - data_range[0]) + data_range[0]).astype(data_type)

    else:
        _train_data, _train_labels, _val_data, _val_labels = [], [], [], []

    if dataset == 'train':
        return _train_data, _train_labels
    elif dataset == 'val':
        return _val_data, _val_labels


if __name__ == '__main__':
    data, labels = load_data_for_Tools()
    print(labels[100])
