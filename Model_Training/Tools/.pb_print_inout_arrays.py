# -- coding: utf-8 --

import os

import tensorflow as tf

from Tools import ROOTS_Tools


def create_graph(model_path):
    with tf.compat.v1.gfile.FastGFile(os.path.join(model_path), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def print_io_arrays(pb):
    gf = tf.compat.v1.GraphDef()
    m_file = open(pb, 'rb')
    gf.ParseFromString(m_file.read())

    with open('gfnode.txt', 'a') as the_file:
        for n in gf.node:
            the_file.write(n.name + '\n')

    file = open('gfnode.txt', 'r')
    data = file.readlines()
    print("Input name = ")
    file.seek(0)
    print(file.readline())
    print("output name = ")
    print(data[len(data) - 1])


if __name__ == "__main__":
    pd_file_path = os.path.join(ROOTS_Tools.output_model_path, 'mnist.pb')
    print_io_arrays(pd_file_path)
