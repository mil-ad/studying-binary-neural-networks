from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from datasets.utils import DatasetBase

NUM_THREADS = 28


class CIFAR10_GCN_WHITENED(DatasetBase):
    def __init__(self, num_epochs, batch_size):

        DatasetBase.__init__(self,
                             name='CIFAR-10-GCN-WHITENED',
                             num_classes=10,
                             num_train_samples=45000,
                             num_val_samples=5000,
                             num_test_samples=10000,
                             image_size=32,
                             channels=3)

        train_dataset = self._prepare_dataset(
            '/datasets/cifar10_gca_whitened/train.tfrecords',
            batch_size, num_epochs)

        val_dataset = self._prepare_dataset(
            '/datasets/cifar10_gca_whitened/val.tfrecords', batch_size)

        test_dataset = self._prepare_dataset(
            '/datasets/cifar10_gca_whitened/test.tfrecords', batch_size)

        self._create_iterators(train_dataset, val_dataset, test_dataset)

    def _prepare_dataset(self, tfrecord_path, batch_size, repeat=1):

        dataset = tf.data.TFRecordDataset(tfrecord_path)

        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(10000, repeat))

        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(self._cifar10_mapper,
                                          batch_size,
                                          num_parallel_batches=NUM_THREADS))

        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    @staticmethod
    def _cifar10_mapper(dataset):

        feature_map = {
            'image': tf.FixedLenFeature([], dtype=tf.string),
            'label': tf.FixedLenFeature([], dtype=tf.int64)
        }

        parsed_features = tf.parse_single_example(dataset, feature_map)

        image = tf.decode_raw(parsed_features['image'], tf.float64)
        image = tf.reshape(image, [3, 32, 32])
        image = tf.transpose(image, [1, 2, 0])

        label = tf.cast(parsed_features['label'], tf.int32)
        one_hot_label = tf.one_hot(label, 10)
        one_hot_label = tf.squeeze(one_hot_label)

        # This is due to a bug in TF:
        # https://github.com/tensorflow/tensorflow/issues/18355
        return tf.convert_to_tensor(''), image, one_hot_label


class CIFAR10(DatasetBase):
    def __init__(self, num_epochs, batch_size, validation_size=0):

        DatasetBase.__init__(self,
                             name='CIFAR10',
                             num_classes=10,
                             num_train_samples=50000,
                             num_val_samples=0,
                             num_test_samples=10000,
                             image_size=32,
                             channels=3)

        train_dataset = tf.data.TFRecordDataset(
                            './datasets/cifar_10/by_javier/train.tfrecord')
        test_dataset = tf.data.TFRecordDataset(
                            './datasets/cifar_10/by_javier/test.tfrecord')

        train_dataset, val_dataset = self._split_for_validation(
            train_dataset, validation_size)

        train_dataset = self._prepare_dataset(train_dataset)
        test_dataset = self._prepare_dataset(test_dataset)

        train_dataset = train_dataset.repeat(num_epochs)

        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

        self._create_iterators(train_dataset, val_dataset, test_dataset)

    def _prepare_dataset(self, dataset):

        dataset = dataset.map(self._cifar10_mapper)

        dataset = dataset.shuffle(buffer_size=100)

        return dataset

    @staticmethod
    def _cifar10_mapper(dataset):

        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64)}

        parsed_features = tf.parse_single_example(dataset, feature)

        image = tf.decode_raw(parsed_features['image'], tf.float32)
        image = tf.reshape(image, [32, 32, 3])

        label = tf.cast(parsed_features['label'], tf.int32)
        one_hot_label = tf.one_hot(label, 10)
        one_hot_label = tf.squeeze(one_hot_label)

        return '', image, one_hot_label
