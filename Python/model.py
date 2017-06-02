# coding=utf-8

# ==============================================================================

"""Armado del modelo"""

# ==============================================================================

import tensorflow as tf
import numpy as np

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================

weights = {
    'h1': tf.Variable(tf.random_normal([FLAGS.model_n_input, FLAGS.model_n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([FLAGS.model_n_hidden_1, FLAGS.model_n_hidden_2])),
    'out': tf.Variable(tf.random_normal([FLAGS.model_n_hidden_2, 1]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([FLAGS.model_n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([FLAGS.model_n_hidden_2])),
    'out': tf.Variable(tf.random_normal([FLAGS.model_n_classes]))
}


def inference(x):
    # Hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer
