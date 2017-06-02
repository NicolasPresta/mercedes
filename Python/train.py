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
        x_train, y_train, x_val, y_val = input.get_inputs()

        # Calculamos el tamaño del dataset (cantidad de ejemplos)
        total_len = x_train.shape[0]

        # Inicializamos todas las variables
        init = tf.global_variables_initializer()

        # Definimos la configuración general de la sesion
        config = tf.ConfigProto()

        config.log_device_placement = FLAGS.log_device_placement
        config.allow_soft_placement = FLAGS.allow_soft_placement

        # Creamos la sesión
        sess = tf.Session(config=config)
        sess.run(init)

        # Training cycle
        for epoch in range(FLAGS.train_epochs):
            avg_cost = 0.
            total_batch = int(total_len / FLAGS.train_batch_size)

            # Loop over all batches
            for i in range(total_batch - 1):
                batch_x = x_train[i * FLAGS.train_batch_size:(i + 1) * FLAGS.train_batch_size]
                batch_y = y_train[i * FLAGS.train_batch_size:(i + 1) * FLAGS.train_batch_size]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})

                # Compute average loss
                avg_cost += c / total_batch

            # sample prediction
            label_value = batch_y
            estimate = p
            err = label_value - estimate
            print("num batch:", total_batch)

            # Display logs per epoch step
            if epoch % FLAGS.train_display_step == 0:
                print("Epoch:", (epoch + 1), "cost=", avg_cost)
                print("[*]----------------------------")
                for i in range(3):
                    print("label value:" + label_value[i] + "estimated value:" + estimate[i])
                print("[*]============================")

            print("Optimization Finished!")

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({x: x_val, y: y_val}))


def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()
