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


# -------------------------------- DATASET --------------------------------
tf.app.flags.DEFINE_string('dataset_porcentaje_entrenamiento', 75, 'Porcentaje de ejemplos que van al set de validation')


# -------------------------------- INPUT --------------------------------
tf.app.flags.DEFINE_integer('input_num_preprocess_threads', 2, 'Numero de hilos que hacen el preprocesado')
tf.app.flags.DEFINE_integer('input_num_readers', 2, 'Numero de readers')
tf.app.flags.DEFINE_integer('image_height', 40, 'Alto imagen')
tf.app.flags.DEFINE_integer('image_width', 40, 'Ancho imagen')


# -------------------------------- MODELO --------------------------------
tf.app.flags.DEFINE_integer('model_n_input', 579, 'Cantidad de neuronas entrada')
tf.app.flags.DEFINE_integer('model_n_hidden_1', 256, 'Cantidad de neuronas en la capa 1')
tf.app.flags.DEFINE_integer('model_n_hidden_2', 128, 'Cantidad de neuronas en la capa 2')


# -------------------------------- ENTRENAMIENTO --------------------------------
tf.app.flags.DEFINE_integer("train_learning_rate", 0.001, "Learning rate")
tf.app.flags.DEFINE_integer("train_epochs", 100000, "Cantidad de pasos de entrenamiento")
tf.app.flags.DEFINE_integer("train_display_step", 1, "Cada cuanto muestra por consola el avance")
tf.app.flags.DEFINE_integer("train_steps_to_guardar_checkpoint", 100, "Cada cuanto muestra por consola el avance")
tf.app.flags.DEFINE_boolean('log_device_placement', False, "Si logea la ubicación de variables al inciar la ejecución")
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, "Si permite una asignación de variables flexible")


# -------------------------------- EVALUACION --------------------------------






