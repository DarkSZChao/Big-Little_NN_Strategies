# -- coding: utf-8 --

import os

import Roots
from Tools.load_dataset import load_data

output_path = Roots.output_dataset_path

train_data_s3, train_labels, _, _ = load_data(sensor='s3',
                                              quantization_to_0_255='on',
                                              label_categorical='off',
                                              current_activity='all')
data_No = [100, 200, 300, 400, 500, 600]

print(train_data_s3[data_No, :, 0])
print(train_data_s3[data_No, :, 1])
print(train_data_s3[data_No, :, 2])
print(train_labels[data_No])

for i in range(len(data_No)):
    # Save data to txt
    a = train_data_s3[data_No[i], :, :]
    b = a.transpose(1, 0)
    # savetxt(os.path.join(output_path, 'train_data_s3.txt'), b, delimiter=',', fmt='%d')
    b.tofile(os.path.join(output_path, ''.join([str(data_No[i]), '.txt'])))
