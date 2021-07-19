import numpy as np
import pandas as pd
import os 
import tensorflow as tf
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np
import h5py

def process():
    for dirname,_, filenames in os.walk('data'):
        for filename in filenames:
            print(os.path.join(dirname,filename))

    with h5py.File('archive/full_dataset_vectors.h5', 'r') as dataset:
        x_train = dataset["X_train"][:]
        x_test = dataset["X_test"][:]
        y_train = dataset["y_train"][:]
        y_test = dataset["y_test"][:]

    print ("x_train shape: ", x_train.shape)
    print ("y_train shape: ", y_train.shape)
    print ("x_test shape:  ", x_test.shape)
    print ("y_test shape:  ", y_test.shape)

    xtrain = np.ndarray((x_train.shape[0], 4096, 3))
    xtest = np.ndarray((x_test.shape[0], 4096, 3))

    def add_rgb_dimention(array):
        scaler_map = cm.ScalarMappable(cmap="Oranges")
        array = scaler_map.to_rgba(array)[:, : -1]
        return array

    for i in range(x_train.shape[0]):
        xtrain[i] = add_rgb_dimention(x_train[i])
    for i in range(x_test.shape[0]):
        xtest[i] = add_rgb_dimention(x_test[i])


    xtrain = xtrain.reshape(x_train.shape[0], 16, 16, 16, 3)
    xtest = xtest.reshape(x_test.shape[0], 16, 16, 16, 3)

    print(xtrain.shape)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return xtrain,xtest,y_train,y_test