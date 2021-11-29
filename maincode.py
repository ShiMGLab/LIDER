import tensorflow as tf
from training import *
from flags import set_flags

import pickle
import os


if __name__ == '__main__':
    FLAGS = set_flags()
    np.random.seed(0)
    tf.set_random_seed(0)

    # Create folders
    if not os.path.exists(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir)


    # create autoencoder and perform training


    _, AE, sess= main_supervised_1view(FLAGS)


