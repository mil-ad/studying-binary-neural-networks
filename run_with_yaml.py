from __future__ import division, print_function, absolute_import

import yaml
import sys

import tensorflow as tf

from datasets import CIFAR10, CIFAR10_GCN_WHITENED
from datasets import MNIST
from datasets import TinyImageNet

from models import AlexNet
from models import BWN  # Binary-Weight Network (from XNOR-Net)
import models.binaryconnect as binaryconnect
import models.binarynet as binarynet   # BinaryConnect + binary activation
from models import DebugNet  # Small CNN for training on MNIST

from train_utils import train

_RANDOM_SEED = 1234


def get_dataset(dataset_name, num_epochs, batch_size):

    if dataset_name == 'mnist':
        dataset = MNIST(num_epochs, batch_size)
    elif dataset_name == 'cifar10':
        # dataset = CIFAR10(num_epochs, batch_size, validation_size=5000)
        dataset = CIFAR10_GCN_WHITENED(num_epochs, batch_size)
    elif dataset_name == 'tinyimagenet':
        dataset = TinyImageNet(num_epochs, batch_size, validation_size=10000)
    else:
        raise ValueError('Dataset option not valid.')

    return dataset


def get_model_fn(model_name, binarization, disable_batch_norm, disable_weight_constraint, disable_gradient_clipping,
                 enable_glorot_scaling):

    if binarization == 'deterministic-binary':
        binary = True
        stochastic = False
    elif binarization == 'stochastic-binary':
        binary = True
        stochastic = True
    elif binarization == 'disabled':
        binary = False
        stochastic = False
    else:
        print('ERROR!')  # TODO

    kwargs = {}

    if disable_batch_norm is True:
        kwargs['BN_momentum'] = None

    kwargs['disable_weight_constraint'] = disable_weight_constraint
    kwargs['disable_gradient_clipping'] = disable_gradient_clipping
    kwargs['enable_glorot_scaling'] = enable_glorot_scaling

    def model_fn(input_images, num_classes, is_training, keep_prob):
        if model_name == 'alexnet':
            model = AlexNet(input_images, keep_prob, num_classes,
                            weight_decay=0.0005)

        elif model_name == 'binary_connect_mlp':
            model = binaryconnect.MLP(input_images, is_training,
                                      1.0, num_classes,
                                      binary, stochastic, **kwargs)

        elif model_name == 'binary_connect_cnn':
            # Paper settings: 500 epochs, batch size 50
            model = binaryconnect.CNN(input_images, is_training, num_classes,
                                      binary, stochastic, **kwargs)

        elif model_name == 'binarynet_mlp':
            model = binarynet.MLP(input_images, is_training,
                                  keep_prob, num_classes,
                                  binary, stochastic)  # TODO

        elif model_name == 'bwn':
            model = BWN(input_images, keep_prob, num_classes, weight_decay=0.0)

        elif model_name == 'debugnet':
            model = DebugNet(input_images, keep_prob)

        return model

    return model_fn


def get_learning_rate_fn(config, dataset):

    if config['learning_rate']['type'] == 'exponential-decay':

        start_lr = float(config['learning_rate']['start'])
        finish_lr = float(config['learning_rate']['finish'])
        num_epochs = int(config['epochs'])
        steps_per_epoch = dataset.num_train_samples // config['batch_size']

        def exponential_decay_lr():

            global_step_op = tf.train.get_global_step()

            decaye_rate = (finish_lr/start_lr)**(1.0/num_epochs)

            learning_rate = tf.train.exponential_decay(start_lr, global_step_op,
                                                       decay_steps=steps_per_epoch,
                                                       decay_rate=decaye_rate,
                                                       staircase=True)

            return(learning_rate)

        return exponential_decay_lr

    elif config['learning_rate']['type'] == 'piecewise_constant':

        steps_per_epoch = dataset.num_train_samples // config['batch_size']

        def piecewise_lr():

            global_step_op = tf.train.get_global_step()

            epoch_boundaries = eval(config['learning_rate']['epoch_boundaries'])
            step_boundaries = [epoch * steps_per_epoch for epoch in epoch_boundaries]

            values = eval(config['learning_rate']['values'])

            learning_rate = tf.train.piecewise_constant(global_step_op, step_boundaries, values)

            return learning_rate

        return piecewise_lr

    else:
        assert(0)


def get_loss_fn(config):

    # Make sure it gives loss operator the name: total_loss

    if config['loss'] == 'square_hinge_loss':

        def square_hinge_loss(labels, predictions):
            """TF has builtin hinge loss function but not the squared version.
            There are multiple definitions for multi-class hinge loss; this one is
            based on the implementation in BinaryConnect/BinaryNets papers.
            """
            polar_labels = tf.cast((labels*2)-1, tf.float32)  # [0,1] -> [-1,1]

            hinge_loss = tf.maximum(0.0, 1.0-tf.multiply(polar_labels, predictions))

            return tf.reduce_mean(tf.square(hinge_loss), name='total_loss')

        return square_hinge_loss

    elif config['loss'] == 'softmax_cross_entropy':

        def loss_fn(labels, predictions):
            return tf.losses.softmax_cross_entropy(labels, predictions)

        return loss_fn

    else:
        assert(0)


def get_optimiser_fn(config, lr_fn, loss_fn):

    def optimiser_fn(actual_labels, model_output):

        learning_rate = lr_fn()
        total_loss = loss_fn(actual_labels, model_output)
        optimiser_op = eval(config['optimiser']['function'])(learning_rate, **eval(config['optimiser']['kwargs']))

        # This is necessary because of tf.layers.batch_normalization()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimiser_op.minimize(total_loss, tf.train.get_global_step())

        tf.summary.scalar('Total loss', total_loss)
        tf.summary.scalar('Learning Rate', learning_rate)

        return train_op, total_loss

    return optimiser_fn


if __name__ == '__main__':

    with open(sys.argv[1], 'r', ) as f:
        config = yaml.load(f)

    if config.get('fixed_seed', True):
        tf.set_random_seed(_RANDOM_SEED)

    dataset = get_dataset(config['dataset'], config['epochs'], config['batch_size'])
    model_fn = get_model_fn(
                config['model'], config['binarization'],
                config.get('disable_batch_norm', False),
                config.get('disable_weight_constraint', False),
                config.get('disable_gradient_clipping', False),
                config.get('enable_glorot_scaling', False),
                )

    lr_fn = get_learning_rate_fn(config, dataset)
    loss_fn = get_loss_fn(config)
    optimiser_fn = get_optimiser_fn(config, lr_fn, loss_fn)

    train(config['epochs'],
          config['batch_size'],
          dataset,
          model_fn,
          optimiser_fn,
          False,
          config['experiment-name'],
          False)
