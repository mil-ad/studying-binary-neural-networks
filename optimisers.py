# -*- coding: utf-8 -*-
"""


"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf


def square_hinge_loss(labels, predictions):
    """TF has builtin hinge loss function but not the squared version.
    There are multiple definitions for multi-class hinge loss; this one is
    based on the implementation in BinaryConnect/BinaryNets papers.
    """
    polar_labels = tf.cast((labels*2)-1, tf.float32)  # [0,1] -> [-1,1]

    hinge_loss = tf.maximum(0.0, 1.0-tf.multiply(polar_labels, predictions))

    return tf.reduce_mean(tf.square(hinge_loss))


def binary_connect_optimiser(global_step_op, NUM_EPOCHS, steps_per_epoch,
                             labels, model_output, start_lr, finish_lr):

    # Apply exponential LR decay at the end of each epoch
    decaye_rate = (finish_lr/start_lr)**(1.0/NUM_EPOCHS)
    learning_rate = tf.train.exponential_decay(start_lr, global_step_op,
                                               decay_steps=steps_per_epoch,
                                               decay_rate=decaye_rate,
                                               staircase=True)

    # No explicit weight regularization so the only loss is square hinge loss
    total_loss = square_hinge_loss(labels, model_output)

    # This is necessary because of tf.layers.batch_normalization()
    # See https://tensorflow.org/api_docs/python/tf/layers/batch_normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step_op)

    tf.summary.scalar('Total loss', total_loss)
    tf.summary.scalar('Learning Rate', learning_rate)

    return train_op, total_loss


def alexnet_optimiser(global_step_op, labels, model_output):
    # NUM_EPOCHS = 90
    # BATCH_SIZE = 128
    # AlexNet: Divide learning rate by 10 when the validation error rate stops
    # improving.
    # TODO: We're not following above here because I don't know when validation
    # error rate stops imporving. For now let's just decay by %25 every 100k steps
    INITIAL_LEARNING_RATE = 0.01
    learning_rate = tf.train.exponential_decay(
                    INITIAL_LEARNING_RATE, global_step_op,
                    decay_steps=100000, decay_rate=0.75, staircase=True)

    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    loss = tf.losses.softmax_cross_entropy(labels, model_output)
    reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = loss + reg_loss

    optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
    train_op = optimizer.minimize(total_loss, global_step_op)

    tf.summary.scalar('classifier loss', loss)
    tf.summary.scalar('reg loss', reg_loss)
    tf.summary.scalar('total loss', total_loss)
    tf.summary.scalar('Learning Rate', learning_rate)

    return train_op, total_loss
