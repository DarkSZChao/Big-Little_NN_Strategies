# -- coding: utf-8 --

import os
import sys

import numpy as np

from Tools import ROOTS_Tools
from Tools.load_dataset import load_data_for_Tools

np.set_printoptions(threshold=sys.maxsize)  # print out the whole numpy array

INPUT_MODE = '1'  # '1' or '3'
BOARD = 'SparkFun'  # 'SparkFun' or 'ECM3532' or 'Apollo2' or 'STM32L4R5ZI-P'
generate_data, generate_labels = load_data_for_Tools()  # set DATASET_Drange = [0, 255] in ROOTS_Tools.py for SparkFun, [-128, 127] for the others
# data_No = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# data_No = [0, 1, 2, 100, 101, 102, 200, 201, 202, 2000, 2001, 2002, 2100, 2101, 2102, 600, 601, 602]
# data_No = list(range(0, 150))
data_No = list(range(400, 410)) + list(range(1031, 1041)) + list(range(1160, 1170)) + list(range(1550, 1560)) + list(range(1700, 1710)) + list(range(2460, 2470))
lines_body, labels = [], []
SampleCount = str(len(data_No))
categories = str(generate_labels.max() + 1)

if ROOTS_Tools.CURRENT_MODEL == 'MOTION_Detector':
    if ROOTS_Tools.SENSOR == 'all':  # for MOTION_Detector model with 3 sensor inputs
        data_len_single = str(len(generate_data['s1'][0].reshape(-1)))
        generate_data_merged = np.concatenate([generate_data['s1'], generate_data['s2'], generate_data['s3']], 2)
        data_len_merged = str(len(generate_data_merged[0].reshape(-1)))
        if INPUT_MODE == '3':  # organize dataset for BIG model config of 3 inputs
            # convert data from type of ndarray to str
            data_s1, data_s2, data_s3 = [], [], []
            for i in range(len(data_No)):
                sub_data = generate_data['s1'][data_No[i]].reshape(-1)  # stretch to one dimension
                data_s1.append(','.join(str(sub_data).strip('[]').split()))
            data_s1 = '},\n{'.join(data_s1)

            for i in range(len(data_No)):
                sub_data = generate_data['s2'][data_No[i]].reshape(-1)  # stretch to one dimension
                data_s2.append(','.join(str(sub_data).strip('[]').split()))
            data_s2 = '},\n{'.join(data_s2)

            for i in range(len(data_No)):
                sub_data = generate_data['s3'][data_No[i]].reshape(-1)  # stretch to one dimension
                data_s3.append(','.join(str(sub_data).strip('[]').split()))
            data_s3 = '},\n{'.join(data_s3)

            # convert labels from type of ndarray to str
            labels = []
            for i in range(len(data_No)):
                sub_label = generate_labels[data_No[i]]
                labels.append(str(sub_label))
            labels = ','.join(labels)

            # write content for dataset.h
            if BOARD == 'SparkFun':
                lines_body = ['// Datasets\n',
                              'const int Input_data_s1[SAMPLE_COUNT][', data_len_single, '] = {\n',
                              '{', data_s1, '}\n',
                              '};\n\n',

                              'const int Input_data_s2[SAMPLE_COUNT][', data_len_single, '] = {\n',
                              '{', data_s2, '}\n',
                              '};\n\n',

                              'const int Input_data_s3[SAMPLE_COUNT][', data_len_single, '] = {\n',
                              '{', data_s3, '}\n',
                              '};\n\n']

            elif BOARD == 'ECM3532':
                lines_body = ['// Datasets\n',
                              'const q7_t Input_data_s1[SAMPLE_COUNT][', data_len_single, '] = {\n',
                              '{', data_s1, '}\n',
                              '};\n\n',

                              'const q7_t Input_data_s2[SAMPLE_COUNT][', data_len_single, '] = {\n',
                              '{', data_s2, '}\n',
                              '};\n\n',

                              'const q7_t Input_data_s3[SAMPLE_COUNT][', data_len_single, '] = {\n',
                              '{', data_s3, '}\n',
                              '};\n\n']

            elif BOARD == 'Apollo2' or BOARD == 'STM32L4R5ZI-P':
                lines_body = ['// Datasets\n',
                              'const int8_t Input_data_s1[SAMPLE_COUNT][', data_len_single, '] = {\n',
                              '{', data_s1, '}\n',
                              '};\n\n',

                              'const int8_t Input_data_s2[SAMPLE_COUNT][', data_len_single, '] = {\n',
                              '{', data_s2, '}\n',
                              '};\n\n',

                              'const int8_t Input_data_s3[SAMPLE_COUNT][', data_len_single, '] = {\n',
                              '{', data_s3, '}\n',
                              '};\n\n']

        elif INPUT_MODE == '1':  # organize dataset for BIG model config of 1 input (merged)
            # convert data from type of ndarray to str
            data_merged, data_s3 = [], []
            for i in range(len(data_No)):
                sub_data = generate_data_merged[data_No[i]].reshape(-1)  # stretch to one dimension
                data_merged.append(','.join(str(sub_data).strip('[]').split()))
            data_merged = '},\n{'.join(data_merged)

            for i in range(len(data_No)):
                sub_data = generate_data['s3'][data_No[i]].reshape(-1)  # stretch to one dimension
                data_s3.append(','.join(str(sub_data).strip('[]').split()))
            data_s3 = '},\n{'.join(data_s3)

            # convert labels from type of ndarray to str
            labels = []
            for i in range(len(data_No)):
                sub_label = generate_labels[data_No[i]]
                labels.append(str(sub_label))
            labels = ','.join(labels)

            # write content for dataset.h
            if BOARD == 'SparkFun':
                lines_body = ['// Datasets\n',
                              'const int Input_data_s123[SAMPLE_COUNT][', data_len_merged, '] = {\n',
                              '{', data_merged, '}\n',
                              '};\n\n',

                              'const int Input_data_s3[SAMPLE_COUNT][', data_len_single, '] = {\n',
                              '{', data_s3, '}\n',
                              '};\n\n']

            elif BOARD == 'ECM3532':
                lines_body = ['// Datasets\n',
                              'const q7_t Input_data_s123[SAMPLE_COUNT][', data_len_merged, '] = {\n',
                              '{', data_merged, '}\n',
                              '};\n\n',

                              'const q7_t Input_data_s3[SAMPLE_COUNT][', data_len_single, '] = {\n',
                              '{', data_s3, '}\n',
                              '};\n\n']

            elif BOARD == 'Apollo2' or BOARD == 'STM32L4R5ZI-P':
                lines_body = ['// Datasets\n',
                              'const int8_t Input_data_s123[SAMPLE_COUNT][', data_len_merged, '] = {\n',
                              '{', data_merged, '}\n',
                              '};\n\n',

                              'const int8_t Input_data_s3[SAMPLE_COUNT][', data_len_single, '] = {\n',
                              '{', data_s3, '}\n',
                              '};\n\n']

        lines_head = ['// Test data\n\n',
                      '// Name of datasets\n',
                      '#define DATASETS_NAME "dataset"\n\n',

                      '// Number of datasets\n',
                      '#define SAMPLE_COUNT ', SampleCount, '\n\n']

        lines_tail = ['//Labels: ', categories, ' categories 0 to ', str(int(categories) - 1), '\n',
                      'const int Output_expect[SAMPLE_COUNT] = {', labels, '};\n\n']

        # Save data to .h file
        file_name = BOARD + '_' + ROOTS_Tools.CURRENT_MODEL + '_s123_' + str(INPUT_MODE) + 'IN.h'
        open(os.path.join(ROOTS_Tools.output_dataset_path, file_name), 'w+').write(''.join(lines_head + lines_body + lines_tail))

    else:
        data_len_single = str(len(generate_data[0].reshape(-1)))

        # convert data from type of ndarray to str
        data = []
        for i in range(len(data_No)):
            sub_data = generate_data[data_No[i]].reshape(-1)  # stretch to one dimension
            data.append(','.join(str(sub_data).strip('[]').split()))
        data = '},\n{'.join(data)

        # convert labels from type of ndarray to str
        labels = []
        for i in range(len(data_No)):
            sub_label = generate_labels[data_No[i]]
            labels.append(str(sub_label))
        labels = ','.join(labels)

        if BOARD == 'SparkFun':
            lines = ['// Test data\n\n',
                     '// Name of datasets\n',
                     '#define DATASETS_NAME "dataset"\n\n',

                     '// Number of datasets\n',
                     '#define SAMPLE_COUNT ', SampleCount, '\n\n',

                     '// Datasets\n',
                     'const int Input[SAMPLE_COUNT][', data_len_single, '] = {\n',
                     '{', data, '}\n',
                     '};\n\n',

                     '//Labels: ', categories, ' categories 0 to ', str(int(categories) - 1), '\n',
                     'const int Output[SAMPLE_COUNT] = {', labels, '};\n\n',
                     ]
        elif BOARD == 'ECM3532':
            lines = ['// Test data\n\n',
                     '// Name of datasets\n',
                     '#define DATASETS_NAME "dataset"\n\n',

                     '// Number of datasets\n',
                     '#define SAMPLE_COUNT ', SampleCount, '\n\n',

                     '// Datasets\n',
                     'const q7_t pIn0[SAMPLE_COUNT][', data_len_single, '] = {\n',
                     '{', data, '}\n',
                     '};\n\n',

                     '//Labels: ', categories, ' categories 0 to ', str(int(categories) - 1), '\n',
                     'const q7_t pExpect[SAMPLE_COUNT] = {', labels, '};\n\n',
                     ]
        else:
            lines = []

        # Save data to .h file
        file_name = BOARD + '_' + ROOTS_Tools.CURRENT_MODEL + '_' + ROOTS_Tools.SENSOR + '.h'
        open(os.path.join(ROOTS_Tools.output_dataset_path, file_name), 'w+').write(''.join(lines))
