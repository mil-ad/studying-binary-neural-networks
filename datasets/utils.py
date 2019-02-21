# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf


class DatasetBase:
    """
    Usage:

        Before a dataset object can be used it needs to evaluate dataset
        handles inside session:

        dataset.evaluate_handles(sess)

        Once the handles are evaluated you can get elements from the training
        dataset by evaluating dataset.next_element and passing the dataset
        selector:

        _, images, labels = sess.run(
                                dataset.next_element,
                                feed_dict=dataset.from_training_set())

        The validation and testing datasets are re-initialisable so that you
        can go through them multiple times during training. In order to switch
        to validation dataset to run some evaluation first initialize it and
        then get elements:

        dataset.initialize_validation(sess)
        _, images, labels = sess.run(
                                dataset.next_element,
                                feed_dict=dataset.from_validation_set())

        To switch back to the next element from training dataset simply use
        from_training_set() selector. """

    def __init__(self, name, num_classes, num_train_samples,
                 num_val_samples, num_test_samples, image_size, channels):
        self._name = name
        self._num_classes = num_classes
        self._num_train_samples = num_train_samples
        self._num_val_samples = num_val_samples
        self._num_test_samples = num_test_samples
        self._image_size = image_size
        self._channels = channels

    def _split_for_validation(self, train_dataset, val_size):

        assert(self._num_val_samples == 0)

        self._num_train_samples -= val_size
        self._num_val_samples = val_size

        val_dataset = train_dataset.skip(self.num_train_samples)
        train_dataset = train_dataset.take(self.num_train_samples)

        return train_dataset, val_dataset

    def _create_iterators(self, train_dataset, val_dataset, test_dataset):

        self._handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
                                self._handle,
                                train_dataset.output_types,
                                train_dataset.output_shapes)
        self._next_element = iterator.get_next()

        self._training_iterator = train_dataset.make_one_shot_iterator()
        self._validation_iterator = val_dataset.make_initializable_iterator()
        self._testing_iterator = test_dataset.make_initializable_iterator()

    def evaluate_handles(self, sess):
        """
        Evaluate the tensors that are used to feed the `handle` placeholder to
        switch between datasets.
        """
        self._training_handle = sess.run(
                                    self._training_iterator.string_handle())
        self._validation_handle = sess.run(
                                    self._validation_iterator.string_handle())
        self._testing_handle = sess.run(
                                    self._testing_iterator.string_handle())

    def initialize_validation(self, sess):
        sess.run(self._validation_iterator.initializer)

    def initialize_testing(self, sess):
        sess.run(self._testing_iterator.initializer)

    def from_training_set(self):
        return {self._handle: self._training_handle}

    def from_validation_set(self):
        return {self._handle: self._validation_handle}

    def from_testing_set(self):
        return {self._handle: self._testing_handle}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_train_samples(self):
        return self._num_train_samples

    @property
    def num_val_samples(self):
        return self._num_val_samples

    @property
    def num_test_samples(self):
        return self._num_test_samples

    @property
    def image_size(self):
        return self._image_size

    @property
    def channels(self):
        return self._channels

    @property
    def next_element(self):
        return self._next_element
