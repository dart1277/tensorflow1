import math
import os, datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

import keras
from keras import layers, Model
from pathlib import Path

# unsupervised learning
# kmeans clustering, autoencoding (eg. PCA)

# autoencoders reconstructs outputs based on inputs
# in all autoencoders input and output layers need to have the same dimensionality (to be able to reconstruct output from input)
# in autoencoders hidden, lower dimensionality layers are used to reduce features in data

def create_autoencoder(x_train):
    encoder = tf.keras.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        ]
    )
    decoder = tf.keras.Sequential([
        layers.Dense(32, input_shape=(16), activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(28*28, activation="relu"),
        layers.Reshape([28, 28]),
        ]
    )

    # cnn encoder
    # model.add(Reshape([28,28,1], input_shape=[28,28]))
    # model.add(Conv2D(16, kernel_size=3,  padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(32, kernel_size=3,  padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, kernel_size=3 padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # decoder input size has last dimension of 64, since last encoder layer has input of 64
    # model.add(Conv2DTranspose(32, kernel_size=3, input_shape=(3, 3, 64), strides=2, padding='valid', activation='relu'))
    # model.add(Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='relu'))
    # model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='sigmoid'))
    # model.add(Reshape([28,28]))
    # use binarycrossentropu SGD(lt=1.0), metrics=mse

    ae_model = tf.keras.Sequential([encoder, decoder])
    ae_model.summary()

    ae_model.compile(
        loss="mse",
        metrics=["mae", "mse"],
        optimizer=tf.keras.optimizers.RMSprop()
    )


    return ae_model

def test1():
    alphabets_data = pd.read_csv('datasets/A_Z_handwritten_datta_kaggle.csv', header=None)
    print(alphabets_data[0].unique()) # represent numeric labels for letters

    # no labels are present during learning (output labels are exactly the input data)
    # ae_model.fit(train_imgs, train_imgs, epochs=20, verbose=True)
    ...


if __name__ == '__main__':
    test1()