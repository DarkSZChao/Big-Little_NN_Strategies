import os

import tensorflow as tf

from Tools import ROOTS_Tools
from MOTION_Detector.S_s3_forNNoM import model_structure

checkpoint_No = '45'

model_h5_checkpoint = os.path.join(ROOTS_Tools.output_model_path, 'checkpoint', 'cp-' + checkpoint_No + '.ckpt')
model_h5 = os.path.join(ROOTS_Tools.output_model_path, 'S4_s3.h5')

model = model_structure()
model.summary()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights(model_h5_checkpoint)

model.save(model_h5)
