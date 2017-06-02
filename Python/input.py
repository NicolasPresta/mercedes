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
    id_test = data_test[data_test.columns[0:1]]
    del data_test['ID']

    data_train = pd.get_dummies(data_train, prefix=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'])
    data_test = pd.get_dummies(data_test, prefix=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'])

    # Completamos las columnas que tiene el set de test que no tiene el set de train
    # La completamos con una columna en blanco
    data_train['X0_ae'] = data_train['X11']
    data_train['X0_ag'] = data_train['X11']
    data_train['X0_an'] = data_train['X11']
    data_train['X0_av'] = data_train['X11']
    data_train['X0_bb'] = data_train['X11']
    data_train['X0_p'] = data_train['X11']
    data_train['X2_ab'] = data_train['X11']
    data_train['X2_ad'] = data_train['X11']
    data_train['X2_aj'] = data_train['X11']
    data_train['X2_ax'] = data_train['X11']
    data_train['X2_u'] = data_train['X11']
    data_train['X2_w'] = data_train['X11']
    data_train['X5_a'] = data_train['X11']
    data_train['X5_b'] = data_train['X11']
    data_train['X5_t'] = data_train['X11']
    data_train['X5_z'] = data_train['X11']

    # Completamos las columnas que tiene el set de train que no tiene el set de test
    # La completamos con una columna en blanco
    data_test['X0_aa'] = data_test['X11']
    data_test['X0_ab'] = data_test['X11']
    data_test['X0_ac'] = data_test['X11']
    data_test['X0_q'] = data_test['X11']
    data_test['X2_aa'] = data_test['X11']
    data_test['X2_ar'] = data_test['X11']
    data_test['X2_c'] = data_test['X11']
    data_test['X2_l'] = data_test['X11']
    data_test['X2_o'] = data_test['X11']
    data_test['X5_u'] = data_test['X11']

    # Dividimos el set de train, dejamos una parte aparte para validación del aprendizaje
    # Nos va a servir para evitar sobreajuste del modelo
    msk = np.random.rand(len(data_train)) < (FLAGS.dataset_porcentaje_entrenamiento / 100)
    train = data_train[msk]
    val = data_train[~msk]

    x_train = train[train.columns[1:600]]
    y_train = train[train.columns[0:1]]

    x_val = val[val.columns[1:600]]
    y_val = val[val.columns[0:1]]

    x_test = data_test[data_test.columns[0:600]]

    return x_train, y_train, x_val, y_val, x_test, id_test


































