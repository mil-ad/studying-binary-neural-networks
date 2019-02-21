import tensorflow as tf
import numpy as np
import glob
import sys

from six.moves import urllib
import tarfile
import shutil

_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
_FILE_NAME = 'cifar-10-python.tar.gz'
_FOLDER_NAME_AFTER_UNTAR = 'cifar-10-batches-py'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataset_dir', '.',
    'The directory where you want the dataset to be downloaded to and converted.')


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def loadData(path_to_downloaded_dataset):

    print("Loading data...")
    num_train_data_batches = 5

    for i in range(num_train_data_batches):
        data_dict = unpickle(path_to_downloaded_dataset + 'data_batch_' + str(i+1))
        if i == 0:
            images_train = data_dict['data']
            labels_train = np.asarray(data_dict['labels'])
        else:
            images_train = np.append(images_train, data_dict['data'], axis=0)
            labels_train = np.append(labels_train, data_dict['labels'])

    print("Image_train data shape:", images_train.shape)
    print("Labels_train data shape:", labels_train.shape)

    # now we do the same for the test set
    data_dict = unpickle(path_to_downloaded_dataset + 'test_batch')
    images_test = data_dict['data']
    labels_test = np.asarray(data_dict['labels'])
    print("Image_test data shape:", images_test.shape)
    print("Labels_test data shape:", labels_test.shape)

    return images_train, labels_train, images_test, labels_test


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def reshapeImage(img_flat):
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    return np.dstack((img_R, img_G, img_B))


def generateTfRecordFile(fileName, images, labels):

    print("Generating Tfrecord file...")

    if images.shape[0] != labels.shape[0]:
        print(" %d and %d" % (images.shape[0], labels.shape[0]))
        raise ValueError("dimensions mismatch!!")

    writer = tf.python_io.TFRecordWriter(fileName)

    for i in range(images.shape[0]):

        img = reshapeImage(images[i].astype('f'))

        feature = {'label': _int64_feature(labels[i]),
                   'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()
    print("Generated: %s" % fileName)


def download_dataset(path):

    file_path = path + _FILE_NAME

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    print(_DATA_URL)
    print(file_path)
    filepath, _ = urllib.request.urlretrieve(_DATA_URL, file_path, _progress)
    print()

    with tf.gfile.GFile(filepath) as f:
        size = f.size()
    print('Successfully downloaded', _FILE_NAME, size, 'bytes.')


def convert(path_to_dataset):

    if not tf.gfile.Exists(path_to_dataset):
        tf.gfile.MakeDirs(path_to_dataset)

    train_filename = path_to_dataset + 'train.tfrecord'
    test_filename = path_to_dataset + 'test.tfrecord'

    if tf.gfile.Exists(train_filename) and tf.gfile.Exists(test_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    file_name = path_to_dataset + _FILE_NAME
    if not tf.gfile.Exists(file_name):
        download_dataset(path_to_dataset)
        print("Uncompressing dataset file...")
        tarfile.open(file_name, 'r:gz').extractall(path_to_dataset)
    else:
        print("Downloaded dataset found: %s" % file_name)
        print("Uncompressing dataset file...")
        tarfile.open(file_name, 'r:gz').extractall(path_to_dataset)

    uncompressed_data_dir = path_to_dataset+_FOLDER_NAME_AFTER_UNTAR+'/'
    images_train, labels_train, images_test, labels_test = loadData(uncompressed_data_dir)

    generateTfRecordFile(train_filename, images_train, labels_train)
    generateTfRecordFile(test_filename, images_test, labels_test)

    print("Deleting directory with uncompressed dataset from URL: %s" % uncompressed_data_dir)
    shutil.rmtree(uncompressed_data_dir)


if __name__ == '__main__':
    print("Downloading and converting CIFAR-10 dataset to/in: %s" % FLAGS.dataset_dir)
    convert(FLAGS.dataset_dir)
