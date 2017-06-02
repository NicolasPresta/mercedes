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

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


def rsquared(y_true, y_pred):
    """ Return R^2 where x and y are array-like."""

    #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    #return r_value**2
    return metrics.r2_score(y_true, y_pred)


def evaluar():
    with tf.Graph().as_default():

        # tf Graph input
        x = tf.placeholder("float", [None, FLAGS.model_n_input])

        # Construimos el modelo ( y = h(x) )
        pred = model.inference(x)

        # Obtenemos inputs
        x_train, y_train, x_val, y_val, x_test, id_test = input.get_inputs()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.dir_checkpoint)
            if ckpt and ckpt.model_checkpoint_path:

                # Cargamos el modelo entrenado
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print("Model restored. global_step: " + str(global_step))

                # Predecimos
                y_pred = sess.run([pred], feed_dict={x: x_test})

                # Guardamos el csv
                resul = pd.DataFrame(data=y_pred[0][0], index=id_test, columns=['y'])
                resul['y'] = input.rescalar_datos(resul['y'])
                resul.index.name = 'ID'
                resul.reset_index().to_csv("./data/submission.csv", sep=',', index=False)

                # R2 train
                y_pred_train = sess.run([pred], feed_dict={x: x_train})
                y_pred_train = input.rescalar_datos(y_pred_train[0][0])
                r2_train = rsquared(input.rescalar_datos(y_train), y_pred_train)
                print("r2_train: " + str(r2_train))

                # R2 val
                y_pred_val = sess.run([pred], feed_dict={x: x_val})
                y_pred_val = input.rescalar_datos(y_pred_val[0][0])
                r2_val = rsquared(input.rescalar_datos(y_val), y_pred_val)
                print("r2_val: " + str(r2_val))

    return 0


def main(_):
    evaluar()

if __name__ == '__main__':
    tf.app.run()