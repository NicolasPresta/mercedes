# coding=utf-8

# ==============================================================================

"""Armado del modelo"""

# ==============================================================================

import tensorflow as tf
import numpy as np

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


def crear_capa(input, size_in, size_out, name, act, keep_prob):
    with tf.name_scope(name):
        w = tf.Variable(tf.random_normal([size_in, size_out],
                                         mean=FLAGS.model_w_init_mean,
                                         stddev=FLAGS.model_w_init_stddev))
        b = tf.Variable(tf.random_normal([size_out],
                                         mean=FLAGS.model_b_init_mean,
                                         stddev=FLAGS.model_b_init_stddev))

        layer = tf.add(tf.matmul(input, w), b)

        if act:
            layer = tf.nn.sigmoid(layer)
            layer = tf.nn.dropout(layer, keep_prob)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", layer)

        return layer, w


def inference(x, keep_prob):

    # CAPA 1
    layer_1, w_1 = crear_capa(x, FLAGS.model_n_input, FLAGS.model_n_hidden_1, "CAPA-1", True, keep_prob)

    # CAPA 2
    layer_2, w_2 = crear_capa(layer_1, FLAGS.model_n_hidden_1, FLAGS.model_n_hidden_2, "CAPA-2", True, keep_prob)

    # CAPA 3
    layer_3, w_3 = crear_capa(layer_2, FLAGS.model_n_hidden_2, FLAGS.model_n_hidden_3, "CAPA-3", True, keep_prob)

    # CAPA OUT
    out_layer, w_out = crear_capa(layer_3, FLAGS.model_n_hidden_3, 1, "CAPA-OUT", False, 1)

    with tf.name_scope("REGULARIZACION-L2"):
        regularizers = tf.nn.l2_loss(w_1) + tf.nn.l2_loss(w_2) + tf.nn.l2_loss(w_3) + tf.nn.l2_loss(w_out)

    return out_layer, regularizers



