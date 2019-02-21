# -*- coding: utf-8 -*-

"""
Courbariaux, Matthieu, Yoshua Bengio, and Jean-Pierre David. "Binaryconnect:
Training deep neural networks with binary weights during propagations."
Advances in neural information processing systems. 2015.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import math


def lr_mult(alpha):
    @tf.custom_gradient
    def _lr_mult(x):
        def grad(dy):
            return dy * alpha * tf.ones_like(x)
        return x, grad
    return _lr_mult


class BinaryConnect(object):
    """The base class defining core binary operations used in BinaryConnect.
    """
    def __init__(self, is_training, BN_momentum, binary=True, stochastic=False,
                 weight_decay=0.0,
                 disable_weight_constraint=False, disable_gradient_clipping=False,
                 enable_glorot_scaling=False):

        self.regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.is_binary = binary
        self.is_stochastic = stochastic
        self.is_training = is_training
        self.BN_momentum = BN_momentum
        self.enable_glorot_scaling = enable_glorot_scaling

        if disable_weight_constraint or not binary:
            self._weight_constraint = None
        else:
            self._weight_constraint = lambda x: tf.clip_by_value(x, -1, 1)

        self.disable_gradient_clipping = disable_gradient_clipping

    @property
    def output(self):
        return self._output

    @staticmethod
    def glorot_LR_scale(x):

        shape = x.get_shape().as_list()

        fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
        fan_out = float(shape[-1])
        n = (fan_in + fan_out) / 2.0
        limit = math.sqrt(1.5 / n)

        return lr_mult(1.0/limit)

    @staticmethod
    def deterministic_binary_op(input_op):
        g = tf.get_default_graph()
        with g.gradient_override_map({"Sign": "Identity"}):
            x = tf.clip_by_value(input_op, -1.0, 1.0)
            return tf.sign(x)

    @staticmethod
    def deterministic_binary_op_pure_ste(input_op):
        """No gradient clipping"""
        g = tf.get_default_graph()
        with g.gradient_override_map({"Sign": "Identity"}):
            return tf.sign(input_op)

    @staticmethod
    def stochastic_binary_op(input_op):
        p = tf.clip_by_value((input_op+1.0)/2, 0, 1)  # Hard sigmoid

        forward_path = (2. * tf.cast(
            tf.greater(p, tf.random_uniform(tf.shape(p))), tf.float32)) - 1.

        backward_path = tf.clip_by_value((input_op), -1.0, 1.0)

        return backward_path + tf.stop_gradient(forward_path - backward_path)

    def binarize(self, input_op):
        """Binarizes weights in the forward pass and uses Straight-Through
        Estimator in the backwards pass (with hard limits to stop gradients
        flowing backwards when the input is too large)
        """
        assert (self.is_binary is True)  # sanity check

        if self.is_stochastic:
            # In stochastic scenario, BinaryConnect only uses binarization to
            # achieve faster training and during test time the real-valued
            # weights are used.
            out_op = tf.where(self.is_training, self.stochastic_binary_op(input_op), input_op)
        elif self.disable_gradient_clipping:
            out_op = self.deterministic_binary_op_pure_ste(input_op)
        else:
            out_op = self.deterministic_binary_op(input_op)

        if self.enable_glorot_scaling:
            return self.glorot_LR_scale(out_op)(out_op)
        else:
            return out_op

    def _get_weights(self, shape, name):

        w_full = tf.get_variable(name=name,
                                 shape=shape,
                                 initializer=self.initializer,
                                 regularizer=self.regularizer,
                                 constraint=self._weight_constraint)

        # tf.summary.histogram('weights full-precision', w_full)

        if self.is_binary:
            w_bin = self.binarize(w_full)
            # tf.summary.histogram('weights binary', w_bin)
            return w_bin
        else:
            return w_full

    def _batch_norm_layer(self, x):
        if self.BN_momentum is None:
            return tf.identity(x, 'batch_norm_bypass')
        else:
            return tf.layers.batch_normalization(
                        x, axis=-1, epsilon=1e-4, center=True, scale=True,
                        momentum=self.BN_momentum, training=self.is_training)


class MLP(BinaryConnect):
    """Multi-Layer Perceptron used for MNIST. No convolution layers.
    """
    def __init__(self, input_op, is_training, keep_prob, num_classes,
                 binary, stochastic, units_per_layer=2048, weight_decay=0.0,
                 BN_momentum=0.85,
                 disable_weight_constraint=False, disable_gradient_clipping=False,
                 enable_glorot_scaling=False):

        assert(keep_prob == 1.0)

        BinaryConnect.__init__(self, is_training, BN_momentum, binary,
                               stochastic, weight_decay,
                               disable_weight_constraint, disable_gradient_clipping,
                               enable_glorot_scaling)

        self._output = self._build_model(input_op, units_per_layer,
                                         num_classes, keep_prob)

    @property
    def name(self):
        return 'BinaryConnect_MLP'

    def _build_model(self, input_op, units_per_layer, num_classes, keep_prob):

        model = tf.layers.flatten(input_op)  # preserves the batch axis

        model = tf.nn.dropout(model, keep_prob)

        model = self._dense("fc_layer1", model, units_per_layer, keep_prob)
        model = self._dense("fc_layer2", model, units_per_layer, keep_prob)
        model = self._dense("fc_layer3", model, units_per_layer, keep_prob)

        model = self._last_layer(model, num_classes)

        model = tf.identity(model, 'model_output')  # just to give it a name

        return model

    def _dense(self, name, input_op, num_units, keep_prob):

        input_dim = input_op.get_shape().as_list()[-1]

        with tf.variable_scope(name) as scope:

            layer = self._get_weights([input_dim, num_units], 'weights')
            # tf.summary.histogram('weights binary', w_binary)
            layer = tf.matmul(input_op, layer)
            layer = self._batch_norm_layer(layer)
            layer = tf.nn.relu(layer)
            layer = tf.nn.dropout(layer, keep_prob)

            return layer

    def _last_layer(self, input_op, num_classes):

        input_dim = input_op.get_shape().as_list()[-1]

        with tf.variable_scope('fc_final') as scope:

            weights = self._get_weights(
                        [input_dim, num_classes], name='weights')
            layer = tf.matmul(input_op, weights)
            layer = self._batch_norm_layer(layer)

        return layer


class CNN(BinaryConnect):
    def __init__(self, input_op, is_training, num_classes,
                 binary, stochastic, weight_decay=0.0, BN_momentum=0.9,
                 disable_weight_constraint=False, disable_gradient_clipping=False,
                 enable_glorot_scaling=False):

        BinaryConnect.__init__(self,
                               is_training,
                               BN_momentum,
                               binary,
                               stochastic,
                               weight_decay,
                               disable_weight_constraint,
                               disable_gradient_clipping,
                               enable_glorot_scaling)

        self._output = self._build_model(input_op, num_classes)

    @property
    def name(self):
        return 'BinaryConnect_CNN'

    def _build_model(self, input_op, num_classes):

        model = self._conv_layer('conv_1', input_op, 128)
        model = self._conv_layer('conv_2', model, 128, pool=True)
        model = self._conv_layer('conv_3', model, 256)
        model = self._conv_layer('conv_4', model, 256, pool=True)
        model = self._conv_layer('conv_5', model, 512)
        model = self._conv_layer('conv_6', model, 512, pool=True)

        model = tf.layers.flatten(model)  # preserves the batch axis

        model = self._dense('fc1', model, 1024)
        model = self._dense('fc2', model, 1024)
        model = self._dense('fc3', model, num_classes, relu=False)

        model = tf.identity(model, 'model_output')  # Just to give it a name

        # tf.summary.histogram('final_activations', model)

        return model

    def _conv_layer(self, name, input_op, out_channels, pool=False):

        in_channels = input_op.get_shape().as_list()[-1]

        with tf.variable_scope(name) as scope:

            weights = self._get_weights([3, 3, in_channels, out_channels],
                                        'weights')

            layer = tf.nn.conv2d(input=input_op,
                                 filter=weights,
                                 strides=[1, 1, 1, 1],
                                 padding='SAME')

            if pool:
                layer = tf.nn.max_pool(layer,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='VALID')

            layer = self._batch_norm_layer(layer)

            layer = tf.nn.relu(layer)

            return layer

    def _dense(self, name, input_op, num_units, relu=True):

        input_dim = input_op.get_shape().as_list()[-1]

        with tf.variable_scope(name) as scope:

            layer = self._get_weights([input_dim, num_units], 'weights')
            # tf.summary.histogram('weights_binary', w_binary)
            layer = tf.matmul(input_op, layer)
            layer = self._batch_norm_layer(layer)

            if relu:
                layer = tf.nn.relu(layer)

            return layer
