# coding=utf-8

# ==============================================================================

"""Inputs del modelo"""

# ==============================================================================

import tensorflow as tf
import numpy as np
import pandas as pd

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


def get_inputs():
    data_train = pd.read_csv(FLAGS.dir_data_train)
    data_test = pd.read_csv(FLAGS.dir_data_test)

    print("Train data dims:", data_train.shape)
    print("Test data dims:", data_test.shape)

    del data_train['ID']

    data_train = pd.get_dummies(data_train, prefix=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'])

    msk = np.random.rand(len(data_train)) < FLAGS.dataset_porcentaje_entrenamiento
    train = data_train[msk]
    val = data_train[~msk]

    x_train = train[train.columns[1:600]]
    y_train = train[train.columns[0:1]]
    x_val = val[val.columns[1:600]]
    y_val = val[val.columns[0:1]]

    return x_train, y_train, x_val, y_val
































