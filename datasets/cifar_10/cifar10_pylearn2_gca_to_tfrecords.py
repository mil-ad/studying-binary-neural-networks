from __future__ import division, print_function, absolute_import

import os
import sys
import re
import random
from glob import glob
import argparse

import tensorflow as tf
import numpy as np

from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.utils import serial


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    # if not isinstance(value, list):
    #     value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    """Wrapper for inserting int64 features into Example proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    """Wrapper for inserting byte features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _convert_to_example_proto(label, image):
    """
    Build an Example proto for an example.
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'image': _bytes_feature(image)}))

    return example


def create_tfrecords(name, dataset, output_dir):

    output_filename = os.path.join(output_dir, '{}.tfrecords'.format(name))

    with tf.python_io.TFRecordWriter(output_filename) as writer:

        for item in zip(dataset.y, dataset.X):

            example = _convert_to_example_proto(np.squeeze(item[0]),
                                                item[1].tobytes())

            writer.write(example.SerializeToString())


if __name__ == '__main__':

    print("Generating .tfrecords files ...")

    preprocessor = serial.load("/datasets/pylearn2_gcn_whitened/preprocessor.pkl")

    train_set = ZCA_Dataset(
        preprocessed_dataset=serial.load("/datasets/pylearn2_gcn_whitened/train.pkl"),
        preprocessor=preprocessor,
        start=0, stop=45000)
    valid_set = ZCA_Dataset(
        preprocessed_dataset=serial.load("/datasets/pylearn2_gcn_whitened/train.pkl"),
        preprocessor=preprocessor,
        start=45000, stop=50000)
    test_set = ZCA_Dataset(
        preprocessed_dataset=serial.load("/datasets/pylearn2_gcn_whitened/test.pkl"),
        preprocessor=preprocessor)

    output_dir = '/datasets/cifar_10/pylearn2_tfrecords'

    create_tfrecords('train', train_set, output_dir)
    create_tfrecords('val', valid_set, output_dir)
    create_tfrecords('test', test_set, output_dir)
