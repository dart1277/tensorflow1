import keras
from tensorflow import keras as k
from keras import layers
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot  as plt
import matplotlib
matplotlib.use('TkAgg')


def test1():
    W_true = 2
    b_true = 0.5
    x = np.linspace(0,3,130)
    y = W_true * x + b_true + np.random.randn(*x.shape) * 0.5
    x = pd.DataFrame(x, columns=['x'])
    y = pd.DataFrame(y, columns=['y'])
    print(x.head())
    print(y.head())
    m = keras.Sequential([layers.Dense(1, input_shape=(1,), activation='linear')])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    m.compile(loss = "mse", metrics = ['mse'], optimizer=optimizer)
    m.fit(x,y, epochs=100)
    print(m.loss)
    z = m.predict(x)
    print(z)
    plt.figure(figsize=(8,8))
    plt.scatter(x,y)
    plt.plot(x, z, 'r--')
    plt.show()

    ...

if __name__ == '__main__':
    test1()