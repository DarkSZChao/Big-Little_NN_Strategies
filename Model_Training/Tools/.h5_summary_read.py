# -- coding: utf-8 --
import os

from tensorflow.keras.models import load_model

from Tools import ROOTS_Tools

model_h5 = os.path.join(ROOTS_Tools.output_model_path, 'Best_model_-128127/dual_motions.h5')

model = load_model(model_h5)
model.summary()
