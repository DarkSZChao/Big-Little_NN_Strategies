# -- coding: utf-8 --
import os

import numpy as np
from tensorflow.keras.models import load_model

from Tools import ROOTS_Tools, load_dataset
from Tools.NNoM import nnom

model_h5 = os.path.join(ROOTS_Tools.output_model_path, 'Best_model_NNoM/B_s123_1input_91.h5')
model_h = os.path.join(ROOTS_Tools.output_model_path, 'Best_model_NNoM/.h_NNoM/B_s123_1input_91.h')

model = load_model(model_h5)
model.summary()

data, _ = load_dataset.load_data_for_Tools()
if ROOTS_Tools.SENSOR == 'all':
    data = np.concatenate([data['s1'], data['s2'], data['s3']], 2)

nnom.generate_model(model, data, name=model_h)
