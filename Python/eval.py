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

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


def evaluar():
    with tf.Graph().as_default():

        # tf Graph input
        x = tf.placeholder("float", [None, FLAGS.model_n_input])

        # Construimos el modelo ( y = h(x) )
        pred = model.inference(x)

        # Obtenemos inputs
        _, _, _, _, x_test, id_test = input.get_inputs()

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
                resul = pd.DataFrame(data=y_pred[0], index=id_test, columns=['y'])
                resul.index.name = 'ID'
                resul.reset_index().to_csv("submission.csv", sep=',', index=False)


    return 0


def main(_):
    evaluar()

if __name__ == '__main__':
    tf.app.run()