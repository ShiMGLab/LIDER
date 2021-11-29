from __future__ import division
from __future__ import print_function


import numpy as np
import scipy.io as sio
import pandas as pd


class DataSet(object):

  def __init__(self, data, labels):

    self._num_examples = labels.shape[0]

    self._data = data
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples
  @property
  def start_index(self):
      return self._index_in_epoch


  def next_batch(self, batch_size, UNSUPERVISED = False):
    """Return the next `batch_size` examples from this data set."""
    n_sample = self.num_examples
    start = self._index_in_epoch
    end = self._index_in_epoch + batch_size
    end = min(end, n_sample)
    id = range(start, end)
    data_input = self._data[id, :]
    if ~UNSUPERVISED:
        target_input = self._labels[id, :]
    else: target_input = []

    self._index_in_epoch = end

    if end == n_sample:
        self._index_in_epoch = 0

    return data_input, target_input



def read_data_sets(FLAGS):




    class DataSets(object):
        pass
    data_sets = DataSets()
    data_train, data_test,  targets_train, targets_test = \
    load_biology_data(FLAGS)
    data_sets.train = DataSet(data_train, targets_train)
    data_sets.test= DataSet(data_test, targets_test)


    return data_sets





def fill_feed_dict_ae_for_hidden(data_set, input_pl, FLAGS):
    input_feed = data_set.data
    feed_dict = {
        input_pl: input_feed}

    return feed_dict


def fill_feed_dict_ae_test(data_set, input_pl, target_pl, FLAGS):
    input_feed = data_set.data
    target_feed = data_set.labels
    feed_dict = {
        input_pl: input_feed,
        target_pl: target_feed}

    return feed_dict


def load_biology_data(FLAGS):

    train_size = FLAGS.train_size

    test_size = FLAGS.test_size
    dimension = FLAGS.dimension
    data = np.loadtxt(".../zeisel_features.txt")
    data = data.T
    # data = data.values


    [n_dim, n_sample] = data.shape


    targets = np.loadtxt('.../zeisel_label.txt')
    np.random.rand(2)
    index = np.random.permutation(n_sample)
    data = data[:, index]

    targets = targets[index]
    n_label = len(np.unique(targets))
    Y = np.zeros([n_sample, n_label])

    targets = targets - 1
    targets = targets.astype('uint8')
    for target in np.unique(targets):
        id = (targets == target).nonzero()[0]
        Y[id, target] = 1
    X = data.T
    X = X[:, 0:dimension]


    data_train = X[0:train_size, :]
    targets_train = np.float32(Y[0:train_size, ])

    data_test = X[train_size:train_size+test_size, :]
    targets_test = np.float32(Y[train_size:train_size+test_size, ])

    # np.savetxt("/Users/apple/Desktop/初步处理的一些数据/ zeisel/macparlandtrue_label.txt",targets_test)





    return data_train, data_test, targets_train, targets_test






