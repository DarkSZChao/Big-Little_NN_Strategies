import numpy
import tensorflow as tf

X = numpy.zeros([157, 128])
Y = numpy.zeros([157], dtype=numpy.int32)
example_id = numpy.array(['%d' % i for i in range(len(Y))])

x_column_name = 'x'
example_id_column_name = 'example_id'

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={x_column_name: X, example_id_column_name: example_id},
    y=Y,
    num_epochs=None,
    shuffle=True)

svm = tf.contrib.learn.SVM(
    example_id_column=example_id_column_name,
    feature_columns=(tf.contrib.layers.real_valued_column(column_name=x_column_name, dimension=128),),
    l2_regularization=0.1)

svm.fit(input_fn=train_input_fn, steps=10, )

