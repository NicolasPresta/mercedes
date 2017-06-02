# coding=utf-8

# ==============================================================================

"""Armado del modelo"""

# ==============================================================================

import tensorflow as tf
import numpy as np

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


def inference(x):

    weights = {
        'h1': tf.Variable(tf.random_normal([FLAGS.model_n_input, FLAGS.model_n_hidden_1], mean=0.0, stddev=1.0)),
        'h2': tf.Variable(tf.random_normal([FLAGS.model_n_hidden_1, FLAGS.model_n_hidden_2], mean=0.0, stddev=1.0)),
        'out': tf.Variable(tf.random_normal([FLAGS.model_n_hidden_2, 1], mean=0.0, stddev=1.0))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([FLAGS.model_n_hidden_1], mean=0.0, stddev=1.0)),
        'b2': tf.Variable(tf.random_normal([FLAGS.model_n_hidden_2], mean=0.0, stddev=1.0)),
        'out': tf.Variable(tf.random_normal([1]))
    }

    # Hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    regularizers = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2'])

    return out_layer, regularizers
