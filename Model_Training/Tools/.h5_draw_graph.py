# -- coding: utf-8 --
import os

import tensorflow as tf
from tensorflow.keras.models import load_model

from Tools import ROOTS_Tools

model = 'dual_motions'
model_h5 = os.path.join(ROOTS_Tools.output_model_path, model + '.h5')
model_png = os.path.join(ROOTS_Tools.output_model_path, model + '.png')

model = load_model(model_h5)
model.summary()

tf.keras.utils.plot_model(
    model,
    to_file=model_png,
    show_shapes=True,
    # show_layer_names=True,
    # rankdir='TB',
    # expand_nested=False,
    # dpi=96,
)
