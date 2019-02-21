"""
Presents MNIST data as a tf.data.Dataset object similar to what we do for
(Tiny) ImageNet so that we can quickly switch to using MNIST for development
and debugging purposes.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# This is a copy from TF's official models. It's supposed to become available
# as a pip package at some point :
# See [this](https://github.com/tensorflow/models/issues/917)
from . import download_mnist_dataset
from datasets.utils import DatasetBase

NUM_THREADS = 28


class HiddenPrints:
    """Used to disable TF's annoying prints when reading MNIST data"""
    def __enter__(self):
        self._stdout_old = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout_old


class MNIST(DatasetBase):
    def __init__(self, num_epochs, batch_size):

        DatasetBase.__init__(self,
                             name='MNIST',
                             num_classes=10,
                             num_train_samples=60000,
                             num_val_samples=0,
                             num_test_samples=10000,
                             image_size=28,
                             channels=1)

        with HiddenPrints():
            train_dataset = download_mnist_dataset.train('./_MNIST_DATA/')
            test_dataset = download_mnist_dataset.test('./_MNIST_DATA/')

        # Set aside 5000 images for validation
        train_dataset, val_dataset = self._split_for_validation(
                                        train_dataset, 5000)

        train_dataset = self._prepare_dataset(
                            train_dataset, batch_size, num_epochs)
        val_dataset = self._prepare_dataset(val_dataset, batch_size)
        test_dataset = self._prepare_dataset(test_dataset, batch_size)

        self._create_iterators(train_dataset, val_dataset, test_dataset)

    def _prepare_dataset(self, dataset, batch_size, repeat=1):

        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(10000, repeat))

        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(self._mapper,
                                          batch_size,
                                          num_parallel_batches=NUM_THREADS))

        return dataset

    @staticmethod
    def _mapper(image, label):
        image = tf.reshape(image, [28, 28, 1])
        image = tf.image.per_image_standardization(image)
        label = tf.cast(label, tf.int32)
        one_hot_label = tf.one_hot(label, 10)
        one_hot_label = tf.squeeze(one_hot_label)

        return tf.convert_to_tensor(''), image, one_hot_label
