# coding=utf-8

# ==============================================================================

"""Entrenamiento del modelo"""

# ==============================================================================

import tensorflow as tf
import config
import input
import model

from datetime import datetime
import os.path
import time
import numpy as np
import shutil

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


def train():

    with tf.Graph().as_default():

        # tf Graph input
        x = tf.placeholder("float", [None, FLAGS.model_n_input])
        y = tf.placeholder("float", [None, 1])

        # Construimos el modelo ( y = h(x) )
        pred = model.inference(x)

        # Definimos costo
        cost = tf.reduce_mean(tf.square(pred - y))

        # Definimos el optimizador (para reducir el costo)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.train_learning_rate).minimize(cost)

        # Traemos los datos de entrada
        x_train, y_train, x_val, y_val, _, _ = input.get_inputs()

        # Creamos el saver que va a guardar nuestro modelo
        saver = tf.train.Saver(tf.global_variables())

        # Inicializamos todas las variables
        init = tf.global_variables_initializer()

        # Definimos la configuración general de la sesion
        config = tf.ConfigProto()

        config.log_device_placement = FLAGS.log_device_placement
        config.allow_soft_placement = FLAGS.allow_soft_placement

        # Creamos la sesión
        sess = tf.Session(config=config)
        sess.run(init)

        # Creamos la operación que va a guardar el resumen para luego visualizarlo desde tensorboard
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.dir_summary_train, sess.graph)

        # Training cycle
        for epoch in range(FLAGS.train_epochs):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, train_cost, p = sess.run([optimizer, cost, pred],
                                        feed_dict={x: x_train, y: y_train},
                                        run_metadata=run_metadata,
                                        options=run_options)

            # Display logs per epoch step
            if epoch % FLAGS.train_display_step == 0:
                print("Epoch:", (epoch + 1), "train_cost=", train_cost)

            if epoch % FLAGS.train_steps_to_guardar_checkpoint == 0 or (epoch + 1) == FLAGS.train_epochs:
                checkpoint_path = os.path.join(FLAGS.dir_checkpoint, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
                print("---> Guardado Checkpoint")

            val_cost, val_p = sess.run([cost, pred],
                                       feed_dict={x: x_val, y: y_val},
                                       run_metadata=run_metadata,
                                       options=run_options)

            if epoch % FLAGS.train_display_step == 0:
                print("Epoch:", (epoch + 1), "val_cost=", val_cost)


        print("Optimization Finished!")


def main(_):
    # creamos el directorio de dir_summary_train si no existe, y si existe lo borramos y creamos de nuevo
    if not os.path.exists(FLAGS.dir_summary_train):
        os.mkdir(FLAGS.dir_summary_train)
    else:
        shutil.rmtree(FLAGS.dir_summary_train)
        os.mkdir(FLAGS.dir_summary_train)

    # creamos el directorio de dir_checkpoint si no existe, y si existe lo borramos y creamos de nuevo
    if not os.path.exists(FLAGS.dir_checkpoint):
        os.mkdir(FLAGS.dir_checkpoint)
    else:
        shutil.rmtree(FLAGS.dir_checkpoint)
        os.mkdir(FLAGS.dir_checkpoint)

    train()

if __name__ == '__main__':
    tf.app.run()
