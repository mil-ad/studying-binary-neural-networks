# -*- coding: utf-8 -*-
"""
Top-level file. Parses the passed arguments to select the model, dataset and
optimisation and passes them to the training routine.
"""
from __future__ import division, print_function, absolute_import

import sys
import argparse
import tensorflow as tf

from datasets import CIFAR10, CIFAR10_GCN_WHITENED
from datasets import MNIST
from datasets import TinyImageNet

from models import AlexNet
from models import BWN  # Binary-Weight Network (from XNOR-Net)
import models.binaryconnect as binaryconnect
import models.binarynet as binarynet   # BinaryConnect + binary activation
from models import DebugNet  # Small CNN for training on MNIST

from optimisers import binary_connect_optimiser, alexnet_optimiser

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


def get_model_fn(model_name, binarization):

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

    def model_fn(input_images, num_classes, is_training, keep_prob):
        if model_name == 'alexnet':
            model = AlexNet(input_images, keep_prob, num_classes,
                            weight_decay=0.0005)

        elif model_name == 'binary_connect_mlp':
            model = binaryconnect.MLP(input_images, is_training,
                                      1.0, num_classes,
                                      binary, stochastic)

        elif model_name == 'binary_connect_cnn':
            # Paper settings: 500 epochs, batch size 50
            model = binaryconnect.CNN(input_images, is_training, num_classes,
                                      binary, stochastic)

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


def get_optimiser_fn(model_name):

    def optimiser_fn(global_step_op, num_epochs, steps_per_epoch, labels, model_output):

        if model_name == 'alexnet':
            train_op = alexnet_optimiser(global_step_op,
                                         labels,
                                         model_output)

        elif model_name in ['binary_connect_mlp', 'binarynet_mlp']:
            train_op = binary_connect_optimiser(global_step_op,
                                                num_epochs,
                                                steps_per_epoch,
                                                labels,
                                                model_output, 1e-3, 3e-6)

        elif model_name == 'binary_connect_cnn':
            train_op = binary_connect_optimiser(global_step_op,
                                                num_epochs,
                                                steps_per_epoch,
                                                labels,
                                                model_output, 3e-3, 2e-6)

        else:
            print("Error!")

        return train_op

    return optimiser_fn


def args_parser(args):
    parser = argparse.ArgumentParser(description='TODO')

    parser.add_argument('-m', '--model', choices=[
                        'alexnet', 'debugnet', 'bwn', 'xnornet',
                        'binary_connect_mlp', 'binary_connect_cnn',
                        'binarynet_mlp'],
                        action='store', required=True, help='TODO')

    parser.add_argument('-d', '--dataset', choices=['mnist', 'cifar10',
                        'cifar100', 'tinyimagenet', 'imagenet'],
                        action='store', required=True, help='TODO')

    parser.add_argument('-e', '--epochs', action='store', default=250,
                        type=int, help='Number of Epochs (Default: 250)')

    parser.add_argument('-b', '--batch-size', action='store', default=100,
                        type=int, help='Batch Size (Default: 100')

    parser.add_argument('-r', '--resume-from-latest-checkpoint',
                        action='store_true', required=False, help='TODO')

    parser.add_argument('-t', '--tag', action='store', required=False,
                        help='Set a tag for the test run. Overrides default unique name')

    parser.add_argument('-f', '--freeze', action='store_true', required=False,
                        help='Freeze the model after training.')

    parser.add_argument('--binarization',
                        choices=['deterministic-binary',
                                 'stochastic-binary',
                                 'disabled'],
                        action='store',
                        required=False,
                        default='deterministic-binary',
                        help='binarization mode')

    return parser.parse_args()


if __name__ == '__main__':

    tf.set_random_seed(_RANDOM_SEED)

    parsed_args = args_parser(sys.argv)

    train(parsed_args.epochs,
          parsed_args.batch_size,
          get_dataset(parsed_args.dataset, parsed_args.epochs, parsed_args.batch_size),
          get_model_fn(parsed_args.model, parsed_args.binarization),
          get_optimiser_fn(parsed_args.model),
          parsed_args.resume_from_latest_checkpoint,
          parsed_args.tag,
          parsed_args.freeze)
