# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os
import sys
import subprocess
from datetime import datetime
from pprint import pprint, pformat
import logging
from glob import glob
from collections import namedtuple

import numpy as np
import tensorflow as tf

_DEFAULT_SESSIONS_PATH = './_sessions/'
_PRINT_SUMMARY_FREQ = 50

logger = logging.getLogger(__name__)


###############################################################################
# Some helper functions
###############################################################################
def compute_accuracy(oneshot_labels, predictions, k):
    """Computes Top-k accuracy. Note that the behaviour of in_top_k differs from
    the top_k op in its handling of ties; if multiple classes have the same
    prediction value and straddle the top-k boundary, all of those classes are
    considered to be in the top k."""
    correct_mask = tf.nn.in_top_k(predictions, tf.argmax(oneshot_labels, 1),
                                  k, name="top_{}_correct_mask".format(k))
    return tf.reduce_mean(tf.cast(correct_mask, tf.float32)) * 100.00


def get_num_trainable_params():
    return sum(np.prod(p.get_shape().as_list())
               for p in tf.trainable_variables())


def get_timedelta(start_time):
    delta = datetime.now() - start_time

    # timedelta objects don't have hours and seconds!
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return('[{:01d} day(s), {:02d} hr(s), {:02d} min(s)]'.
           format(delta.days, hours, minutes))


def time_per_step(start_time, num_steps):
    delta = datetime.now() - start_time

    return (delta / num_steps).total_seconds() * 1000


def get_resettable_mean_metric(values, scope):
    """Used for validation and test datasets where we're interested in computing
    accuracy over the entire dataset but due to their size still have to loop
    through them via batches. Since we do this multiple times during training
    we'd like the metric to be resettable."""

    with tf.variable_scope(scope) as s:
        mean_op, update_mean_op = tf.metrics.mean(values)

    # The .* regex makes filtering work in nested scopes
    variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, '.*'+scope)

    if (len(variables) == 0):
        logger.error("Couldn't collect resettable mean \"{}\"".format(scope))
        sys.exit()

    reset_mean_op = tf.variables_initializer(variables)

    return {'value_op': mean_op,
            'update_op': update_mean_op,
            'reset_op': reset_mean_op}


def find_latest_chkpt(search_dir):
    try:

        sessions = glob(os.path.join(search_dir, '*/'))
        sessions.sort(key=os.path.getmtime, reverse=True)

        # We pick up index 2 from the sorted list instead of 0 or 1 because
        # we've already created the session directory for this run plus the
        # 'latest' symlink
        latest_session = sessions[2]

        latest_checkpoint = tf.train.latest_checkpoint(
                    os.path.join(latest_session, 'train_checkpoints'))

        logger.info('Latest session found: {}'.format(latest_session))
        logger.info('Resuming from checkpoint: {}'.format(latest_checkpoint))

        return latest_checkpoint
    except IndexError:
        logger.error("Couldn't find the checkpoint.")
        sys.exit()


def make_symbolic_link(sessions_dir, session_name):
    latest_symlink = os.path.join(sessions_dir, 'latest')

    try:
        os.symlink(session_name, latest_symlink)
    except FileExistsError:
        logger.debug("Replaced old symbolic link.")
        os.unlink(latest_symlink)
        os.symlink(session_name, latest_symlink)


def save_frozen_model(sess, output_node_names, output_path,
                      placeholder_values=None):
    """Args:
        placeholder_values: dictionary for injecting values into placeholders
    """
    vars_removed_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, tf.get_default_graph().as_graph_def(), output_node_names)

    # We'd like to convert unrelated placeholders to consts at this stage.
    # Examples of such placeholders are "keep_prob" for Dropout layers or
    # "is_training" for BatchNorm layers. Unfortunately there is not an easy way
    # to do this when freezing the model. The only official API I'm aware of is
    # the "map_dict" argument in "tf.import_graph_def" method which means
    # placeholders values can be provided when loading the frozen models. That's
    # why we're doing a dummy load here to use that API and will store the
    # result. The downside is that calling import_graph_def() adds an extra
    # prefix which cannot be removed.

    with tf.Graph().as_default() as output_graph:
        tf.import_graph_def(
            vars_removed_graph_def,
            input_map=placeholder_values,
            name="frozen")

    tf.train.write_graph(
        output_graph,
        output_path,
        'frozen_model.pb',
        as_text=False)


def load_frozen_graph(frozen_graph):

    with open(frozen_graph, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    # temporarily override the current default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(restored_graph_def, name="")

    return graph


def configure_logger(log_file):

    # Make sure we're not adding duplicate handlers if train() is called
    # multiple times.
    if not logger.handlers:

        logger.setLevel(logging.DEBUG)

        file_log_format = logging.Formatter(
                        fmt='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

        console_log_format = file_log_format

        console_log_handler = logging.StreamHandler()
        console_log_handler.setFormatter(console_log_format)
        console_log_handler.setLevel(logging.INFO)

        logger.addHandler(console_log_handler)

        # Also store logs to a file
        file_log_handler = logging.FileHandler(log_file, 'w')
        file_log_handler.setFormatter(file_log_format)
        file_log_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_log_handler)


def train(num_epochs, batch_size, dataset, model_fn, optimiser_fn,
          resume_from_latest_checkpoint=False, tag=None, freeze_model=False):

    # Hopefully the caller has already set the the graph-level random seed for
    # reproducibility.

    # Generate a unique name for this run. Used to name directories for
    # TensorBoard and stored checkpoints.
    this_session_name = datetime.now().strftime("%a-%d-%b-%I%M%p")
    if tag is not None:
        this_session_name = tag

    this_session_path = os.path.join(_DEFAULT_SESSIONS_PATH, this_session_name)
    os.makedirs(this_session_path)

    # Create a 'latest' symlink when necessary
    if len(os.listdir(_DEFAULT_SESSIONS_PATH)) > 1:
        make_symbolic_link(_DEFAULT_SESSIONS_PATH, this_session_name)

    SUMMARIES_PATH = os.path.join(this_session_path, 'summaries')
    TRAIN_CHECKPOINTS_PATH = os.path.join(
                this_session_path, 'train_checkpoints/train_checkpoint')
    BEST_VAL_CHECKPOINT_PATH = os.path.join(
                this_session_path, 'val_checkpoints/best_val_checkpoint')

    configure_logger(os.path.join(this_session_path, 'log.txt'))

    ###########################################################################
    # Placeholders
    ###########################################################################
    batch_images = tf.placeholder(
            tf.float32,
            [None, dataset.image_size, dataset.image_size, dataset.channels],
            'model_input')
    batch_labels = tf.placeholder(tf.float32, [None, dataset.num_classes])

    tf.summary.image('images', batch_images, max_outputs=6)

    is_training = tf.placeholder(tf.bool, name='is_training')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    ###########################################################################
    # The Model
    ###########################################################################
    model = model_fn(batch_images, dataset.num_classes, is_training, keep_prob)

    ###########################################################################
    # Accuracy
    ###########################################################################
    batch_top1_accuracy_op = compute_accuracy(batch_labels, model.output, 1)
    batch_top5_accuracy_op = compute_accuracy(batch_labels, model.output, 5)

    top1_mean_acc_ops = get_resettable_mean_metric(batch_top1_accuracy_op,
                                                   'val_top1_mean')
    top5_mean_acc_ops = get_resettable_mean_metric(batch_top5_accuracy_op,
                                                   'val_top5_mean')

    tf.summary.scalar('Top-1 Training Accuracy', batch_top1_accuracy_op)
    tf.summary.scalar('Top-5 Training Accuracy', batch_top5_accuracy_op)

    # Validation-related summaries are added to a separate collection so that
    # they can be evaluated at separate time from training summaries.
    tf.summary.scalar('Validation Top-1 Accuracy',
                      top1_mean_acc_ops['value_op'],
                      collections=['VALIDATION_SUMMARIES'])

    ###########################################################################
    # Optimiser
    ###########################################################################
    global_step_op = tf.train.create_global_step()
    steps_per_epoch = dataset.num_train_samples // batch_size

    train_op, total_loss_op = optimiser_fn(batch_labels, model.output)

    ###########################################################################
    # TensorBoard Summaries/Checkpoints
    ###########################################################################
    train_summary_op = tf.summary.merge_all()
    val_summary_op = tf.summary.merge_all('VALIDATION_SUMMARIES')

    train_chkpt_op = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
    best_val_chkpt_op = tf.train.Saver(max_to_keep=1)

    ###########################################################################
    # Session
    ###########################################################################

    def evaluate(dataset_init_fn, dataset_select_fn):
        """A helper function used to compute accuracy on validation and
        training dataset. Ideally we'd like to do this in one go but often the
        entire validation set is too big to fit in memory and therefore we
        still need to use batches for validation as well.

        Args:

        Returns:
            Top 1% and Top 5% accuracies
        """
        dataset_init_fn(sess)
        sess.run([top1_mean_acc_ops['reset_op'],
                  top5_mean_acc_ops['reset_op']])

        while True:
            try:
                _, images, labels = sess.run(dataset.next_element,
                                             feed_dict=dataset_select_fn())

                sess.run([top1_mean_acc_ops['update_op'],
                          top5_mean_acc_ops['update_op']],
                         feed_dict={batch_images: images,
                                    batch_labels: labels,
                                    is_training: False,
                                    keep_prob: 1.0})

            except tf.errors.OutOfRangeError:
                top1, top5 = sess.run([top1_mean_acc_ops['value_op'],
                                       top5_mean_acc_ops['value_op']])
                break

        return top1, top5

    def main_training_loop(sess, starting_epoch, num_epochs):

        #################################
        #    The Actual Training Loop
        #################################
        summary_writer = tf.summary.FileWriter(SUMMARIES_PATH, sess.graph)

        dataset.evaluate_handles(sess)

        train_start_time = summary_start_time = datetime.now()
        logger.info('Training start time: {}'.format(train_start_time))

        current_step = starting_epoch * steps_per_epoch
        best_epoch = 0
        best_validation_acc = 0.0
        testing_acc = 0.0

        # TODO: Put these in the right place
        # --------8<----------
        epoch_loss = tf.Variable(0, trainable=False, dtype=tf.float32)
        update_epoch_loss = tf.assign_add(epoch_loss, total_loss_op)
        reset_epoch_loss = tf.variables_initializer([epoch_loss])
        tf.summary.scalar('Epoch Training Loss', epoch_loss, collections=['EPOCH_SUMMARIES'])
        epoch_summary_op = tf.summary.merge_all('EPOCH_SUMMARIES')
        # --------8<----------

        for epoch_num in range(starting_epoch, num_epochs):

            sess.run(reset_epoch_loss)

            for batch_num in range(steps_per_epoch):

                _, images, labels = sess.run(
                                dataset.next_element,
                                feed_dict=dataset.from_training_set())

                sess.run([train_op, update_epoch_loss], feed_dict={batch_images: images,
                                              batch_labels: labels,
                                              is_training: True,
                                              keep_prob: 0.5})

                if batch_num % _PRINT_SUMMARY_FREQ == 0:

                    top1, top5, batch_summary = sess.run(
                            [batch_top1_accuracy_op, batch_top5_accuracy_op,
                             train_summary_op],
                            feed_dict={batch_images: images,
                                       batch_labels: labels,
                                       is_training: False,
                                       keep_prob: 1.0})

                    logger.debug('time/step: {:.2f} ms'.
                                 format(time_per_step(summary_start_time,
                                                      _PRINT_SUMMARY_FREQ)))

                    logger.info(
                        '{} Epoch {:>3} - Batch {:>4} - '
                        'Batch accuracy: top-1 {:5.2f}% - top-5: {:5.2f}%'
                        .format(get_timedelta(train_start_time), epoch_num,
                                batch_num, top1, top5))

                    summary_writer.add_summary(batch_summary, current_step)
                    summary_start_time = datetime.now()

                current_step += 1

            ########################
            #     End of Epoch
            ########################
            train_chkpt_op.save(sess, TRAIN_CHECKPOINTS_PATH, global_step_op)

            # Evaluate accuracy on the validation dataset (and testing dataset
            # if observed validation is the best seen)
            val_acc, _ = evaluate(dataset.initialize_validation,
                                  dataset.from_validation_set)

            summary_writer.add_summary(sess.run(val_summary_op), current_step)
            summary_writer.add_summary(sess.run(epoch_summary_op), current_step)

            logger.info('Epoch training loss {} - Validation accuracy: top-1 {:5.2f}%'.format(
                    epoch_loss.eval(), val_acc))

            if val_acc > best_validation_acc:
                logger.info('New Best Validation!')

                best_validation_acc = val_acc
                best_epoch = epoch_num

                testing_acc, _ = evaluate(dataset.initialize_testing,
                                          dataset.from_testing_set)

                best_val_chkpt_op.save(sess, BEST_VAL_CHECKPOINT_PATH, global_step_op)

            logger.info(
                'Testing accuracy to report: {:5.2f}% (error: {:5.2f}%) - '
                'Seen in epoch {}'
                .format(testing_acc, 100.0 - testing_acc, best_epoch))

        ########################
        #   End of training!
        ########################
        total_train_time = datetime.now() - train_start_time

        logger.info('End of training! Overall training time: {}'
                    .format(total_train_time))

        logger.info('Best observed validation accuracy: {:5.2f}%'
                    .format(best_validation_acc))
        logger.info('Testing accuracy to report: {:5.2f}% (error: {:5.2f}%) - '
                    'Seen in epoch {}'
                    .format(testing_acc, 100-testing_acc, best_epoch))

        summary_writer.close()

        return testing_acc, best_epoch

    def _print_env_details():
        logger.info("=====================================")
        """ A helper function to print env details. """
        logger.info("Timestamp: {}".format(datetime.now()))
        logger.info("TensorFlow Version: {}".format(tf.VERSION))
        logger.info('Session Name: "{}"'.format(this_session_name))
        logger.info("Total number of trainable parameters: {:,}"
                    .format(get_num_trainable_params()))
        logger.info("{} Epochs - Batch Size {}".format(num_epochs, batch_size))
        logger.debug(pformat(globals()))
        logger.debug(pformat(locals()))
        logger.debug("All Trainable Parameters:")
        for var in tf.trainable_variables():
            logger.debug('{} {}'.format(var.name, var.shape))
        logger.info("=====================================")

    _print_env_details()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        if resume_from_latest_checkpoint:
            train_chkpt_op.restore(sess, find_latest_chkpt(_DEFAULT_SESSIONS_PATH))
            starting_epoch = global_step_op.eval() // steps_per_epoch
            # TODO: Restore best validation details
        else:
            starting_epoch = 0

        #######################################################################
        # Let's train!
        #######################################################################
        acc, best_epoch = main_training_loop(sess, starting_epoch, num_epochs)

        if freeze_model:
            logger.info("Feezing the model ...")
            save_frozen_model(sess, ['model_output'], this_session_path,
                              {'is_training': False})
            logger.info("Frozen model saved.")

        sess.close()

    TestResults = namedtuple('TestResults', ['session_name', 'accuracy',
                                             'best_epoch', 'trained_model'])

    return TestResults(this_session_name, acc, best_epoch, model)
