import tensorflow as tf
from tensorflow.keras.models import Model
def my_model():
    model_input = tf.keras.layers.Input(shape=(16, 16, 16, 3))
    x = tf.keras.layers.Conv3D(filters=8, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(model_input)
    x = tf.keras.layers.BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv3D(filters=8, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=8192, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    model_output = tf.keras.layers.Dense(units=10, activation='softmax')(x)
    return  Model(model_input ,model_output)

