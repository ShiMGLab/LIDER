from sklearn.cluster import KMeans

from model import *
import tensorflow as tf
from data import *
from eval import loss_supervised, evaluation, \
    loss_supervised_unsupervised, do_get_hidden, do_testdata
# from utils import *
from collections import deque
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'




def main_supervised_1view(FLAGS):
    """
    Perform supervised training with sparsity penalty.
    :return acc: the accuracy on trainng set.
    :return ae_supervised: the trained autoencoder.
    :return sess: the current session.
    """
    print('Supervised training...')
    data = read_data_sets(FLAGS)

    # combine training and test

    do_test = True

    sess, acc, ae_supervised = supervised_1view(data, FLAGS, do_test)




    return acc, ae_supervised, sess


def supervised_1view(data, FLAGS, do_test=True):
    """
    :param data: input data
    :param do_val: whether do validation.
    :return sess: the current session.
    :return ae: the trained autoencoder.
    :return acc: the accuracy on validation/training batch.
    """
    if FLAGS.initialize == True:
        file_dir = FLAGS.data_dir + 'initialize_encoder.mat'
        matContents = sio.loadmat(file_dir)
        AE_initialize = matContents

    sess = tf.Session()
    acc_record = deque([])
    N_record = 5
    with sess.graph.as_default():
        # here we need to perform the parameter selection

        ae_shape = FLAGS.NN_dims_1

        ae = AutoEncoder(ae_shape)

        if FLAGS.initialize:
            ae._setup_variables(FLAGS.initialize, AE_initialize)

        input_pl = tf.placeholder(tf.float32, shape=(None, FLAGS.dimension), name='input_pl')

        logits = ae.supervised_net(input_pl, FLAGS.num_hidden_layers + 1)

        labels_placeholder = tf.placeholder(tf.float32,
                                            shape=(None, FLAGS.num_classes), name='target_pl')
        # alpha_placeholder = tf.placeholder(tf.float32, None, name='alpha_pl')

        loss,  cr = loss_supervised(logits, labels_placeholder, ae)
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())

        for step in range(FLAGS.supervised_train_steps):  # gradually increase alpha, for fast convergence
            # alpha = 0. + FLAGS.alpha * (1 - np.exp(-(step + 0.) / 300.))
            # if alpha >= FLAGS.alpha * (1.0 - 1e-2):
            #     alpha = FLAGS.alpha
            input_feed, target_feed = data.train.next_batch(FLAGS.batch_size)
            feed_dict_supervised = {input_pl: input_feed,
                                    labels_placeholder: target_feed}

            accuracy, est = evaluation(logits, labels_placeholder)
            sess.run(train_op, feed_dict=feed_dict_supervised)

            acc_train, loss_value,  cr_value, estimation = \
                sess.run([accuracy, loss, cr, est], feed_dict=feed_dict_supervised)

            # Print training process.
            if (step + 1) % FLAGS.display_steps == 0 or step + 1 == FLAGS.supervised_train_steps or step == 0:
                # print(alpha)
                output = 'Train step ' + str(step + 1) + ' minibatch loss: ' + str(loss_value) + ' accuracy: ' + str(acc_train)
                print(output)
                acc = acc_train
                if do_test:
                    if do_test:
                        acc_test, target_prediction = do_testdata(sess, ae, data.test, FLAGS)
                else:
                    acc_test = acc_train

                acc_record.append(acc_test)
                if len(acc_record) > N_record:
                    acc_record.popleft()
                else:
                    pass

                acc_test_show = max(acc_record)
                output = 'accuracy on test: ' + str(acc_test_show)
                print(output)
                acc = acc_test

    return sess, acc, ae




