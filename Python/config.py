# coding=utf-8

# ==============================================================================

import tensorflow as tf

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================

# -------------------------------- DIRECTORIOS --------------------------------
tf.app.flags.DEFINE_string('dir_data_train', './data/train.csv', "csv con datos de entrenamiento (con label)")
tf.app.flags.DEFINE_string('dir_data_test', './data/test.csv', "csv con datos a evaluar (sin label)")
tf.app.flags.DEFINE_string('dir_summary_eval', './summary_eval', "Logs de proceso de evaluación")
tf.app.flags.DEFINE_string('dir_summary_train', './summary_train', "Logs de proceso de entrenamiento")
tf.app.flags.DEFINE_string('dir_checkpoint', './checkpoints', "Resguardo del modelo a utilizar")
tf.app.flags.DEFINE_boolean('borrar_summary', False, "Borrar o no los summarys antes de empezar")
tf.app.flags.DEFINE_boolean('borrar_checkpoint', False, "Borrar o no los checkpoint antes de empezar")

# -------------------------------- DATASET --------------------------------
tf.app.flags.DEFINE_string('dataset_porcentaje_entrenamiento', 80, 'Porcentaje de ejemplos que van al set de train')


# -------------------------------- INPUT --------------------------------


# -------------------------------- MODELO --------------------------------
tf.app.flags.DEFINE_integer('model_n_input', 553, 'Cantidad de neuronas entrada')
tf.app.flags.DEFINE_integer('model_n_hidden_1', 30, 'Cantidad de neuronas en la capa 1')
tf.app.flags.DEFINE_integer('model_n_hidden_2', 20, 'Cantidad de neuronas en la capa 2')
tf.app.flags.DEFINE_integer('model_n_hidden_3', 10, 'Cantidad de neuronas en la capa 2')

tf.app.flags.DEFINE_integer('model_w_init_mean', 0.0, 'Inicialización media w')
tf.app.flags.DEFINE_integer('model_w_init_stddev', 1.0, 'Desvio estandar w')
tf.app.flags.DEFINE_integer('model_b_init_mean', 0.0, 'Inicialización media b')
tf.app.flags.DEFINE_integer('model_b_init_stddev', 1.0, 'Desvio estandar b')

# -------------------------------- ENTRENAMIENTO --------------------------------
tf.app.flags.DEFINE_integer("train_learning_rate", 0.001, "Learning rate")
tf.app.flags.DEFINE_integer("train_epochs", 1000000, "Cantidad de pasos de entrenamiento")
tf.app.flags.DEFINE_integer("train_display_step", 100, "Cada cuanto muestra por consola el avance")
tf.app.flags.DEFINE_integer("train_steps_to_guardar_checkpoint", 200, "Cada cuanto muestra por consola el avance")
tf.app.flags.DEFINE_boolean('log_device_placement', False, "Si logea la ubicación de variables al inciar la ejecución")
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, "Si permite una asignación de variables flexible")
tf.app.flags.DEFINE_integer("max_cant_epoch_malas", 20, "Cantidad máxima de epocas malas (peor r2 score)")


# -------------------------------- EVALUACION --------------------------------






