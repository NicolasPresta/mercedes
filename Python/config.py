# coding=utf-8

# ==============================================================================

import tensorflow as tf

# ==============================================================================

FLAGS = tf.app.flags.FLAGS

# ==============================================================================

# -------------------------------- DIRECTORIOS --------------------------------
tf.app.flags.DEFINE_string('dir_data_train', './data/train.csv', "csv con datos de entrenamiento (con label)")
tf.app.flags.DEFINE_string('dir_data_test', './data/test.csv', "csv con datos a evaluar (sin label)")

# -------------------------------- DATASET --------------------------------
tf.app.flags.DEFINE_string('dataset_porcentaje_entrenamiento', 75, 'Porcentaje de ejemplos que van al set de validation')

tf.app.flags.DEFINE_integer('cantidad_clases', 20, 'Cantidad de clases a reconocer')

tf.app.flags.DEFINE_boolean('modo_procesar', False, 'Al ejecutar analize_jpg realizar emparejamiento')
tf.app.flags.DEFINE_integer('img_por_captura', 110, 'Cantidad de imagenes a conservar por captura')

tf.app.flags.DEFINE_string('capturas_id', 'A,B,C,D,E,F', 'Ids de todas las capturas realizadas')
tf.app.flags.DEFINE_string('capturasEntrenamiento_id', 'A,B,C,D,E', 'Ids de todas las capturas para entrenar')
tf.app.flags.DEFINE_string('capturasTest_id', 'F', 'Ids de todas las capturas para test')

tf.app.flags.DEFINE_integer('train_shards', 1, 'Numero de particiones del dataset de entrenamiento')
tf.app.flags.DEFINE_integer('validation_shards', 1, 'Numero de particiones del dataset de validación')
tf.app.flags.DEFINE_integer('test_shards', 1, 'Numero de particiones del dataset de entrenamiento')
tf.app.flags.DEFINE_integer('dataset_num_threads', 1, 'Numero de hilos de ejecución para armar el dataset')


# -------------------------------- INPUT --------------------------------
tf.app.flags.DEFINE_integer('input_num_preprocess_threads', 2, 'Numero de hilos que hacen el preprocesado')
tf.app.flags.DEFINE_integer('input_num_readers', 2, 'Numero de readers')
tf.app.flags.DEFINE_integer('image_height', 40, 'Alto imagen')
tf.app.flags.DEFINE_integer('image_width', 40, 'Ancho imagen')


# -------------------------------- MODELO --------------------------------
tf.app.flags.DEFINE_integer('model_n_input', 563, 'Cantidad de neuronas entrada')
tf.app.flags.DEFINE_integer('model_n_hidden_1', 256, 'Cantidad de neuronas en la capa 1')
tf.app.flags.DEFINE_integer('model_n_hidden_2', 128, 'Cantidad de neuronas en la capa 2')


# -------------------------------- ENTRENAMIENTO --------------------------------
tf.app.flags.DEFINE_integer("train_learning_rate", 0.001, "Learning rate")
tf.app.flags.DEFINE_integer("train_epochs", 500, "Cantidad de pasos de entrenamiento")
tf.app.flags.DEFINE_integer("train_batch_size", 3314, "Tamaño del batch")
tf.app.flags.DEFINE_integer("train_display_step", 1, "Cada cuanto muestra por consola el avance")


# -------------------------------- EVALUACION --------------------------------
tf.app.flags.DEFINE_boolean('eval_unique', False, "Ejecutar revisión imagen por imagen")
tf.app.flags.DEFINE_boolean('eval_unique_from_dataset', True, "Evaluar imagen por imagen desde dataset")
tf.app.flags.DEFINE_integer('eval_unique_cantidad_img', 3, "Cantidad de imagenes a evaluar si eval_unique = true")
tf.app.flags.DEFINE_string('eval_dataset', 'test', 'Data set usado para validacion (train, validation o test')

tf.app.flags.DEFINE_boolean('eval_distort', False, "Distorcionar imagenes al evaluar")
tf.app.flags.DEFINE_boolean('eval_crop', True, "Distorcionar imagenes al evaluar")

tf.app.flags.DEFINE_integer('eval_num_examples', 2200, "Número de imagenes a evaluar")
tf.app.flags.DEFINE_integer('eval_num_examples_mini', 1000, "Número de imagenes a evaluar durante el entrenamiento")

tf.app.flags.DEFINE_integer("top_k_prediction", 1, "La predicción correcta si esta entre los k primeros resultados")

tf.app.flags.DEFINE_boolean('visualice_conv1_kernels', True, "Hacer Summary de kernels")

titulosStr = ("Fisica universita,"
              "Patrones de diseño,"
              "Introducción a Mineria de datos,"
              "Mineria de datos a traves de ejemplos,"
              "Sistemas expertos,"
              "Sistemas inteligentes,"
              "Big data,"
              "Analisis matematico (vol 3 / Azul),"
              "Einstein,"
              "Analisis matematico (vol 2 / Amarillo),"
              "Teoria de control,"
              "Empresas de consultoría,"
              "Legislación,"
              "En cambio,"
              "Liderazgo Guardiola,"
              "Constitución Argentina,"
              "El arte de conversar,"
              "El señor de las moscas,"
              "Revista: Epigenetica,"
              "Revista: Lado oscuro del cosmos")

tf.app.flags.DEFINE_string('titulos', titulosStr, 'Titulos de los libros')







