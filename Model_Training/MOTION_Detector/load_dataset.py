# -- coding: utf-8 --

import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

try:  # import for pycharm project directory
    from MOTION_Detector import ROOTS
except:  # import for Ubuntu non-project directory
    import ROOTS

dataset_path = ROOTS.MOTION_Detector_input_dataset_path

train_data_path = os.path.join(dataset_path, 'train', 'signals')
train_labels_path = os.path.join(dataset_path, 'train')
val_data_path = os.path.join(dataset_path, 'test', 'signals')
val_labels_path = os.path.join(dataset_path, 'test')

scaler = MinMaxScaler()


# --Import train data of sensor 1--
def load_train_data_s1(_data_range, _data_type):
    train_data_s1_x = np.loadtxt(os.path.join(train_data_path, 'body_acc_x_train.txt'))
    train_data_s1_y = np.loadtxt(os.path.join(train_data_path, 'body_acc_y_train.txt'))
    train_data_s1_z = np.loadtxt(os.path.join(train_data_path, 'body_acc_z_train.txt'))
    _train_data_s1 = np.array([train_data_s1_x, train_data_s1_y, train_data_s1_z])
    _train_data_s1 = _train_data_s1.transpose([1, 2, 0])
    # Redistribute numbers from 0 to 1
    _train_data_s1 = scaler.fit_transform(_train_data_s1.reshape(-1, _train_data_s1.shape[-1])).reshape(_train_data_s1.shape)
    # quantization to range of...
    _train_data_s1 = (_train_data_s1 * (_data_range[1] - _data_range[0]) + _data_range[0]).astype(_data_type)
    print('传感器1: %s, 传感器1的X轴: %s' % (str(_train_data_s1.shape), str(train_data_s1_x.shape)))
    return _train_data_s1


# --Import train data of sensor 2--
def load_train_data_s2(_data_range, _data_type):
    train_data_s2_x = np.loadtxt(os.path.join(train_data_path, 'body_gyro_x_train.txt'))
    train_data_s2_y = np.loadtxt(os.path.join(train_data_path, 'body_gyro_y_train.txt'))
    train_data_s2_z = np.loadtxt(os.path.join(train_data_path, 'body_gyro_z_train.txt'))
    _train_data_s2 = np.array([train_data_s2_x, train_data_s2_y, train_data_s2_z])
    _train_data_s2 = _train_data_s2.transpose([1, 2, 0])
    # Redistribute numbers from 0 to 1
    _train_data_s2 = scaler.fit_transform(_train_data_s2.reshape(-1, _train_data_s2.shape[-1])).reshape(_train_data_s2.shape)
    # quantization to range of...
    _train_data_s2 = (_train_data_s2 * (_data_range[1] - _data_range[0]) + _data_range[0]).astype(_data_type)
    print('传感器2: %s, 传感器2的X轴: %s' % (str(_train_data_s2.shape), str(train_data_s2_x.shape)))
    return _train_data_s2


# --Import train data of sensor 3--
def load_train_data_s3(_data_range, _data_type):
    train_data_s3_x = np.loadtxt(os.path.join(train_data_path, 'total_acc_x_train.txt'))
    train_data_s3_y = np.loadtxt(os.path.join(train_data_path, 'total_acc_y_train.txt'))
    train_data_s3_z = np.loadtxt(os.path.join(train_data_path, 'total_acc_z_train.txt'))
    _train_data_s3 = np.array([train_data_s3_x, train_data_s3_y, train_data_s3_z])
    _train_data_s3 = _train_data_s3.transpose([1, 2, 0])
    # Redistribute numbers from 0 to 1
    _train_data_s3 = scaler.fit_transform(_train_data_s3.reshape(-1, _train_data_s3.shape[-1])).reshape(_train_data_s3.shape)
    # quantization to range of...
    _train_data_s3 = (_train_data_s3 * (_data_range[1] - _data_range[0]) + _data_range[0]).astype(_data_type)
    print('传感器3: %s, 传感器3的X轴: %s' % (str(_train_data_s3.shape), str(train_data_s3_x.shape)))
    return _train_data_s3


# --Import validation data of sensor 1--
def load_val_data_s1(_data_range, _data_type):
    val_data_s1_x = np.loadtxt(os.path.join(val_data_path, 'body_acc_x_test.txt'))
    val_data_s1_y = np.loadtxt(os.path.join(val_data_path, 'body_acc_y_test.txt'))
    val_data_s1_z = np.loadtxt(os.path.join(val_data_path, 'body_acc_z_test.txt'))
    _val_data_s1 = np.array([val_data_s1_x, val_data_s1_y, val_data_s1_z])
    _val_data_s1 = _val_data_s1.transpose([1, 2, 0])
    # Redistribute numbers from 0 to 1
    _val_data_s1 = scaler.fit_transform(_val_data_s1.reshape(-1, _val_data_s1.shape[-1])).reshape(_val_data_s1.shape)
    # quantization to range of...
    _val_data_s1 = (_val_data_s1 * (_data_range[1] - _data_range[0]) + _data_range[0]).astype(_data_type)
    print('传感器1: %s, 传感器1的X轴: %s' % (str(_val_data_s1.shape), str(val_data_s1_x.shape)))
    return _val_data_s1


# --Import validation data of sensor 2--
def load_val_data_s2(_data_range, _data_type):
    val_data_s2_x = np.loadtxt(os.path.join(val_data_path, 'body_gyro_x_test.txt'))
    val_data_s2_y = np.loadtxt(os.path.join(val_data_path, 'body_gyro_y_test.txt'))
    val_data_s2_z = np.loadtxt(os.path.join(val_data_path, 'body_gyro_z_test.txt'))
    _val_data_s2 = np.array([val_data_s2_x, val_data_s2_y, val_data_s2_z])
    _val_data_s2 = _val_data_s2.transpose([1, 2, 0])
    # Redistribute numbers from 0 to 1
    _val_data_s2 = scaler.fit_transform(_val_data_s2.reshape(-1, _val_data_s2.shape[-1])).reshape(_val_data_s2.shape)
    # quantization to range of...
    _val_data_s2 = (_val_data_s2 * (_data_range[1] - _data_range[0]) + _data_range[0]).astype(_data_type)
    print('传感器2: %s, 传感器2的X轴: %s' % (str(_val_data_s2.shape), str(val_data_s2_x.shape)))
    return _val_data_s2


# --Import validation data of sensor 3--
def load_val_data_s3(_data_range, _data_type):
    val_data_s3_x = np.loadtxt(os.path.join(val_data_path, 'total_acc_x_test.txt'))
    val_data_s3_y = np.loadtxt(os.path.join(val_data_path, 'total_acc_y_test.txt'))
    val_data_s3_z = np.loadtxt(os.path.join(val_data_path, 'total_acc_z_test.txt'))
    _val_data_s3 = np.array([val_data_s3_x, val_data_s3_y, val_data_s3_z])
    _val_data_s3 = _val_data_s3.transpose([1, 2, 0])
    # Redistribute numbers from 0 to 1
    _val_data_s3 = scaler.fit_transform(_val_data_s3.reshape(-1, _val_data_s3.shape[-1])).reshape(_val_data_s3.shape)
    # quantization to range of...
    _val_data_s3 = (_val_data_s3 * (_data_range[1] - _data_range[0]) + _data_range[0]).astype(_data_type)
    print('传感器3: %s, 传感器3的X轴: %s' % (str(_val_data_s3.shape), str(val_data_s3_x.shape)))
    return _val_data_s3


# --Import train label--
def load_train_labels(label_categorical, current_activity):
    _train_labels = np.loadtxt(os.path.join(train_labels_path, 'y_train.txt')) - 1  # 标签是从1开始的
    if current_activity == 'all':
        pass
    else:
        _train_labels[_train_labels == current_activity] = 6
        _train_labels[_train_labels < 6] = 0
        _train_labels[_train_labels == 6] = 1
    if label_categorical == 'yes' or label_categorical == 'on':
        _train_labels = to_categorical(_train_labels)  # for loss='categorical_crossentropy'
    print('传感器标签: %s' % str(_train_labels.shape))
    return _train_labels.astype(int)


# --Import validation label--
def load_val_labels(label_categorical, current_activity):
    _val_labels = np.loadtxt(os.path.join(val_labels_path, 'y_test.txt')) - 1  # 标签是从1开始的
    if current_activity == 'all':
        pass
    else:
        _val_labels[_val_labels == current_activity] = 6
        _val_labels[_val_labels < 6] = 0
        _val_labels[_val_labels == 6] = 1
    if label_categorical == 'yes' or label_categorical == 'on':
        _val_labels = to_categorical(_val_labels)  # for loss='categorical_crossentropy'
    print('传感器标签: %s' % str(_val_labels.shape))
    return _val_labels.astype(int)


def load_data_for_MOTION_Detector(sensor, current_activity, data_range, data_type=np.int16, label_categorical='off'):
    if sensor == 's1':
        print('训练数据: ')
        _train_data_s1 = load_train_data_s1(data_range, data_type)
        _train_labels = load_train_labels(label_categorical, current_activity)
        print('验证数据: ')
        _val_data_s1 = load_val_data_s1(data_range, data_type)
        _val_labels = load_val_labels(label_categorical, current_activity)
        return _train_data_s1, _train_labels, _val_data_s1, _val_labels

    elif sensor == 's2':
        print('训练数据: ')
        _train_data_s2 = load_train_data_s2(data_range, data_type)
        _train_labels = load_train_labels(label_categorical, current_activity)
        print('验证数据: ')
        _val_data_s2 = load_val_data_s2(data_range, data_type)
        _val_labels = load_val_labels(label_categorical, current_activity)
        return _train_data_s2, _train_labels, _val_data_s2, _val_labels

    elif sensor == 's3':
        print('训练数据: ')
        _train_data_s3 = load_train_data_s3(data_range, data_type)
        _train_labels = load_train_labels(label_categorical, current_activity)
        print('验证数据: ')
        _val_data_s3 = load_val_data_s3(data_range, data_type)
        _val_labels = load_val_labels(label_categorical, current_activity)
        return _train_data_s3, _train_labels, _val_data_s3, _val_labels

    elif sensor == 'all':
        print('训练数据: ')
        _train_data_s1 = load_train_data_s1(data_range, data_type)
        _train_data_s2 = load_train_data_s2(data_range, data_type)
        _train_data_s3 = load_train_data_s3(data_range, data_type)
        _train_labels = load_train_labels(label_categorical, current_activity)
        print('验证数据: ')
        _val_data_s1 = load_val_data_s1(data_range, data_type)
        _val_data_s2 = load_val_data_s2(data_range, data_type)
        _val_data_s3 = load_val_data_s3(data_range, data_type)
        _val_labels = load_val_labels(label_categorical, current_activity)
        _train_data = {'s1': _train_data_s1, 's2': _train_data_s2, 's3': _train_data_s3}
        _val_data = {'s1': _val_data_s1, 's2': _val_data_s2, 's3': _val_data_s3}
        return _train_data, _train_labels, _val_data, _val_labels


def load_single_motion_for_MOTION_Detector(sensor, current_activity, data_range, data_type=np.int16):
    if sensor == 's1':
        print('训练数据: ')
        _train_data_s1 = load_train_data_s1(data_range, data_type)
        _train_labels = load_train_labels('off', current_activity)
        # extract single motion from all 6 motions
        _train_data_s1 = _train_data_s1[_train_labels == 1]
        print('提取单个标签: %d. Shape: %s' % (current_activity, str(_train_data_s1.shape)))

        print('验证数据: ')
        _val_data_s1 = load_val_data_s1(data_range, data_type)
        _val_labels = load_val_labels('off', current_activity)
        # extract single motion from all 6 motions
        _val_data_s1 = _val_data_s1[_val_labels == 1]
        print('提取单个标签: %d. Shape: %s' % (current_activity, str(_val_data_s1.shape)))

        return _train_data_s1, _val_data_s1

    elif sensor == 's2':
        print('训练数据: ')
        _train_data_s2 = load_train_data_s2(data_range, data_type)
        _train_labels = load_train_labels('off', current_activity)
        # extract single motion from all 6 motions
        _train_data_s2 = _train_data_s2[_train_labels == 1]
        print('提取单个标签: %d. Shape: %s' % (current_activity, str(_train_data_s2.shape)))

        print('验证数据: ')
        _val_data_s2 = load_val_data_s2(data_range, data_type)
        _val_labels = load_val_labels('off', current_activity)
        # extract single motion from all 6 motions
        _val_data_s2 = _val_data_s2[_val_labels == 1]
        print('提取单个标签: %d. Shape: %s' % (current_activity, str(_val_data_s2.shape)))

        return _train_data_s2, _val_data_s2

    elif sensor == 's3':
        print('训练数据: ')
        _train_data_s3 = load_train_data_s3(data_range, data_type)
        _train_labels = load_train_labels('off', current_activity)
        # extract single motion from all 6 motions
        _train_data_s3 = _train_data_s3[_train_labels == 1]
        print('提取单个标签: %d. Shape: %s' % (current_activity, str(_train_data_s3.shape)))

        print('验证数据: ')
        _val_data_s3 = load_val_data_s3(data_range, data_type)
        _val_labels = load_val_labels('off', current_activity)
        # extract single motion from all 6 motions
        _val_data_s3 = _val_data_s3[_val_labels == 1]
        print('提取单个标签: %d. Shape: %s' % (current_activity, str(_val_data_s3.shape)))

        return _train_data_s3, _val_data_s3

    elif sensor == 'all':
        print('训练数据: ')
        _train_data_s1 = load_train_data_s1(data_range, data_type)
        _train_data_s2 = load_train_data_s2(data_range, data_type)
        _train_data_s3 = load_train_data_s3(data_range, data_type)
        _train_labels = load_train_labels('off', current_activity)
        # extract single motion from all 6 motions
        _train_data_s1 = _train_data_s1[_train_labels == 1]
        _train_data_s2 = _train_data_s2[_train_labels == 1]
        _train_data_s3 = _train_data_s3[_train_labels == 1]
        print('提取单个标签: %d. Shape: %s' % (current_activity, str(_train_data_s1.shape)))
        print('提取单个标签: %d. Shape: %s' % (current_activity, str(_train_data_s2.shape)))
        print('提取单个标签: %d. Shape: %s' % (current_activity, str(_train_data_s3.shape)))

        print('验证数据: ')
        _val_data_s1 = load_val_data_s1(data_range, data_type)
        _val_data_s2 = load_val_data_s2(data_range, data_type)
        _val_data_s3 = load_val_data_s3(data_range, data_type)
        _val_labels = load_val_labels('off', current_activity)
        # extract single motion from all 6 motions
        _val_data_s1 = _val_data_s1[_val_labels == 1]
        _val_data_s2 = _val_data_s2[_val_labels == 1]
        _val_data_s3 = _val_data_s3[_val_labels == 1]
        print('提取单个标签: %d. Shape: %s' % (current_activity, str(_val_data_s1.shape)))
        print('提取单个标签: %d. Shape: %s' % (current_activity, str(_val_data_s2.shape)))
        print('提取单个标签: %d. Shape: %s' % (current_activity, str(_val_data_s3.shape)))

        return _train_data_s1, _train_data_s2, _train_data_s3, _val_data_s1, _val_data_s2, _val_data_s3


if __name__ == '__main__':
    # train_data_s1, train_labels, val_data_s1, val_labels = load_data_for_MOTION_Detector(sensor='s1', current_activity=0, data_range=[0, 255], data_type=np.int16, label_categorical='off')
    train_data, train_labels, val_data, val_labels = load_data_for_MOTION_Detector(sensor='all',  # when sensor='all', the function return dictionary type which includes s1, s2, s3
                                                                                   current_activity='all',  # from 0 to 5
                                                                                   data_range=[0, 255],
                                                                                   data_type=np.int16,
                                                                                   label_categorical='off')

    print(train_data['s1'])
    # train_data_s3, val_data_s3 = load_single_motion_for_MOTION_Detector(sensor='s3', current_activity=5, data_range=[0, 255], data_type=np.int16)
