# coding=utf-8

# ==============================================================================

"""Entrenamiento del modelo"""

# ==============================================================================

import tensorflow as tf
import config
import input
import model
import eval

from datetime import datetime
import os.path
import time
import numpy as np
import shutil

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================


def train(summary_dir, checkpoint_dir):

    with tf.Graph().as_default():

        # tf Graph input
        x = tf.placeholder("float", [None, FLAGS.model_n_input])
        y = tf.placeholder("float", [None, 1])
        keep_prob = tf.placeholder("float")

        # Construimos el modelo ( y = h(x) )
        pred, regularizers = model.inference(x, keep_prob)

        # Definimos costo
        with tf.name_scope("COSTO"):
            # cost = tf.reduce_mean(tf.square(pred - y) + 0.01 * regularizers)
            cost = tf.reduce_mean(tf.square(pred - y))
            tf.summary.scalar("cost", cost)

        with tf.name_scope("R2-SCORE"):
            total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
            unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, pred)))
            R_squared = tf.subtract(1.0, tf.div(total_error, unexplained_error))
            tf.summary.scalar("R_squared", R_squared)

        # Definimos el optimizador (para reducir el costo)
        with tf.name_scope("OPTIMIZADOR"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.train_learning_rate).minimize(cost)

        # Traemos los datos de entrada
        with tf.name_scope("INPUT"):
            x_train, y_train, x_val, y_val, x_test, id_test = input.get_inputs()

        # Creamos el saver que va a guardar nuestro modelo
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

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
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

        r2_val_ant = -10000
        cant_epoch_malas = 0

        # Entrenamiento
        for epoch in range(FLAGS.train_epochs):

            # Optimización
            _, train_cost, p, s = sess.run([optimizer, cost, pred, summary_op],
                                           feed_dict={x: x_train, y: y_train,  keep_prob: 0.7},
                                           run_metadata=run_metadata,
                                           options=run_options)

            #summary_writer.add_run_metadata(run_metadata, 'epoch%d' % epoch)
            #summary_writer.add_summary(s, epoch)

            # Mostrar avance por consola
            if epoch % FLAGS.train_display_step == 0:
                val_cost, val_p = sess.run([cost, pred],
                                           feed_dict={x: x_val, y: y_val, keep_prob: 1},
                                           run_metadata=run_metadata,
                                           options=run_options)

                if epoch % FLAGS.train_display_step == 0:
                    print("Epoch:", (epoch + 1), " - train_cost=", train_cost, " - val_cost=", val_cost)

            # Resguardar el modelo
            if epoch % FLAGS.train_steps_to_guardar_checkpoint == 0 or (epoch + 1) == FLAGS.train_epochs:
                checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
                r2_train, r2_val = eval.evaluar(checkpoint_dir, False)
                if r2_val_ant > r2_val:
                    cant_epoch_malas = cant_epoch_malas + 1
                else:
                    cant_epoch_malas = 0
                r2_val_ant = r2_val
                print("check save -  \t \t \t \t \t \t \t \t \t \t \t \t r2_train: ",
                      str(r2_train), " - r2_val: ", str(r2_val), " - cant_epoch_malas", cant_epoch_malas)

            if cant_epoch_malas > FLAGS.max_cant_epoch_malas:
                print("------------------ ENTRENAMIENTO ABORTADO ------------------")
                break

        print("------------------ ENTRENAMIENTO FINALIZADO ------------------")


def main(_):
    # creamos el directorio de dir_summary_train si no existe, y si existe lo borramos y creamos de nuevo
    version = datetime.now().strftime("%Y-%m-%d_%H-%M")
    actual_summary_dir = FLAGS.dir_summary_train + "/" + version
    actual_checkpoint_dir = FLAGS.dir_checkpoint + "/" + version

    if not os.path.exists(FLAGS.dir_summary_train):
        os.mkdir(FLAGS.dir_summary_train)
    else:
        if FLAGS.borrar_summary:
            shutil.rmtree(FLAGS.dir_summary_train)
            os.mkdir(FLAGS.dir_summary_train)

    if os.path.exists(actual_summary_dir):
        shutil.rmtree(actual_summary_dir)
    os.mkdir(actual_summary_dir)

    # creamos el directorio de dir_checkpoint si no existe, y si existe lo borramos y creamos de nuevo
    if not os.path.exists(FLAGS.dir_checkpoint):
        os.mkdir(FLAGS.dir_checkpoint)
    else:
        if FLAGS.borrar_checkpoint:
            shutil.rmtree(FLAGS.dir_checkpoint)
            os.mkdir(FLAGS.dir_checkpoint)

    if os.path.exists(actual_checkpoint_dir):
        shutil.rmtree(actual_checkpoint_dir)
    os.mkdir(actual_checkpoint_dir)

    train(actual_summary_dir, actual_checkpoint_dir)

if __name__ == '__main__':
    tf.app.run()
