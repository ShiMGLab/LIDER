from __future__ import division
import os
from os.path import join as pjoin
import sys
import numpy as np
import tensorflow as tf
import numpy as np


def home_out(path):
    local_dir = os.getcwd()
    return pjoin(local_dir, path)


def set_flags():
    NUM_GENES_1 = 1000
    NUM_HIDDEN = 2
    NUM_CLASSES =9
    NUM_NODES = [528,256]

    NUM_TRAIN_SIZE = 2404

    NUM_TEST_SIZE = 601

    NUM_SAMPLE_SIZE = 3005
    NUM_BATCH_SIZE = 256
    LEARNING_RATE = 1.e-3
    NUM_SUPERVISED_BATCHES = 1200 # shorter training process to save time
    NUM_TRAIN_BATCHES = 10000 # shorter training process to save time
    LAMBDA = 0.006
    ALPHA = 1.2


    RESULT_DIR = 'results/'

    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # Autoencoder Architecture Specific Flags
    flags.DEFINE_integer('num_hidden_layers', NUM_HIDDEN, 'Number of hidden layers')
    flags.DEFINE_list('NN_dims_1', [NUM_GENES_1] + NUM_NODES + [NUM_CLASSES], 'Size of NN')

    flags.DEFINE_integer('hidden_layer_dim', 256,
                         'Number of units in the final hidden layer.')

    flags.DEFINE_integer('num_classes', NUM_CLASSES,
                         'Number of prior known classes.')

    flags.DEFINE_integer('dimension', NUM_GENES_1, 'Number units in input layers.')  # single view
    flags.DEFINE_integer('train_size', NUM_TRAIN_SIZE, 'Number of samples in train set')

    flags.DEFINE_integer('test_size', NUM_TEST_SIZE, 'Number of samples in test set')
    flags.DEFINE_integer('sample_size', NUM_SAMPLE_SIZE, 'Number of whole samples')



    # Constants

    flags.DEFINE_integer('batch_size', NUM_BATCH_SIZE,
                         'Batch size. Must divide evenly into the dataset sizes.')
    flags.DEFINE_float('learning_rate', LEARNING_RATE,
                       'Initial learning rate.')
    flags.DEFINE_integer('supervised_train_steps', NUM_SUPERVISED_BATCHES,
                         'Number of training steps for supervised training')
    flags.DEFINE_integer('train_steps', NUM_TRAIN_BATCHES,
                         'Number of training steps in one epoch for supervised-unsupervised training')
    flags.DEFINE_integer('display_steps', 20,
                         'Display the middle results.')
    flags.DEFINE_boolean('initialize', False, 'whether use initialization')
    flags.DEFINE_boolean('visualization', False, 'visualization middle results') # Use Matlab tsne toolbox for better visualization

    flags.DEFINE_float('beta', ALPHA, 'K-means loss coefficient.')
    flags.DEFINE_float('alpha', LAMBDA, 'sparsity penalty.')
    flags.DEFINE_boolean('parameter_tune', False, 'Tune parameters or not.')


    flags.DEFINE_string('results_dir', home_out(RESULT_DIR),
                        'Directory to put the results.')



    # Python
    flags.DEFINE_string('python', sys.executable,
                        'Path to python executable')
    return FLAGS

