# coding=utf-8

# ==============================================================================

"""Evaluación del modelo"""

# ==============================================================================

import tensorflow as tf
import model
import config
import input
import pandas as pd
import numpy as np
import scipy
from sklearn import metrics
from glob import glob
import os

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


def rsquared(y_true, y_pred):
    """ Return R^2 where x and y are array-like."""

    #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    #return r_value**2
    return metrics.r2_score(y_true, y_pred)


def evaluar(checkpoint_dir, consola=True):
    with tf.Graph().as_default():

        # tf Graph input
        x = tf.placeholder("float", [None, FLAGS.model_n_input])
        keep_prob = tf.placeholder("float")

        # Construimos el modelo ( y = h(x) )
        pred = model.inference(x, keep_prob)

        # Obtenemos inputs
        x_train, y_train, x_val, y_val, x_test, id_test = input.get_inputs()

        saver = tf.train.Saver()

        r2_train = 0
        r2_val = 0

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:

                # Cargamos el modelo entrenado
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                if consola:
                    print("Model restored. global_step: " + str(global_step))

                # Predecimos
                y_pred = sess.run([pred], feed_dict={x: x_test, keep_prob : 1})

                # Guardamos el csv
                resul = pd.DataFrame(data=y_pred[0][0], index=id_test, columns=['y'])
                resul['y'] = input.rescalar_datos(resul['y'])
                resul.index.name = 'ID'
                resul.reset_index().to_csv("./data/submission.csv", sep=',', index=False)

                # R2 train
                y_pred_train = sess.run([pred], feed_dict={x: x_train, keep_prob: 1 })
                y_pred_train = input.rescalar_datos(y_pred_train[0][0])
                r2_train = rsquared(input.rescalar_datos(y_train), y_pred_train)
                if consola:
                    print("r2_train: " + str(r2_train))

                # R2 val
                y_pred_val = sess.run([pred], feed_dict={x: x_val, keep_prob: 1})
                y_pred_val = input.rescalar_datos(y_pred_val[0][0])
                r2_val = rsquared(input.rescalar_datos(y_val), y_pred_val)
                if consola:
                    print("r2_val: " + str(r2_val))

    return r2_train, r2_val


def main(_):
    dir_checkpoint = glob(FLAGS.dir_checkpoint + "/*")[-1]
    evaluar(dir_checkpoint)

if __name__ == '__main__':
    tf.app.run()