import tensorflow as tf

model = tf.keras.models.load_model('/data/api/tensorflow/saved_model/1/model.h5')

model.save('/data/api/tensorflow/saved_model/1/model')