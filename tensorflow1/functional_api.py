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

matplotlib.use('QtAgg') #  MacOSX, QtAgg, GTK4Agg, Gtk3Agg, TkAgg, WxAgg


# model can be also created using model subclassing
class WineClassificationModel(Model):

    # input_shape=xtrain.shape[1]
    # loss=tf.keras.losses.CategoricalCrossentropy() # for multiple category outputs
    # metrics=['accuracy']
    # bath_size=48
    def __init__(self, input_shape):
        super(WineClassificationModel, self).__init__()
        self.d1 = layers.Dense(128, activation="relu", input_shape=input_shape)
        self.d2 = layers.Dense(64, activation="relu")
        self.d3 = layers.Dense(3, activation="softmax") # used for multi class classification

    def call(self, inputs, training=None, mask=None):
        x=self.d1(inputs)
        x=self.d2(x)
        x=self.d3(x)
        return x


def build_model_fun(x_train):
    inputs = tf.keras.Input(shape=(x_train.shape[1],))
    dense_layer_1 = layers.Dense(12, activation="relu")
    x = dense_layer_1(inputs)
    dropout_layer = layers.Dropout(0.3) # minimizes overfitting, turns off 30% of neurons on each epoch
    x = dropout_layer(x)
    dense_layer_2 = layers.Dense(8, activation="relu")
    x = dense_layer_2(x)
    predictions_layer = layers.Dense(1, activation="sigmoid") # probability score output
    predinctions = predictions_layer(x)

    model = tf.keras.Model(inputs=inputs, outputs=predinctions)

    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), # RMSprop can be used as well
                  loss=tf.keras.losses.BinaryCrossentropy(), # since the model is a binary classifier
                  metrics=['accuracy',
                           # idempotent operation that simply divides true_positives by the sum of true_positives and false_positives.
                           tf.keras.metrics.Precision(0.5), # how many positive identifications of the model have been correctly rpedicted
                           # idempotent operation that simply divides true_positives by the sum of true_positives and false_negatives.
                           tf.keras.metrics.Recall(0.5), # how many positives in dataset were correctly indetified by the model
                           ]
                  )

    return model

def test1():
    # scikit-learn has built in datatsets
    # wine_data = datasets.load_wine()
    # print(wine_data['DESCR'])
    #data = pd.DataFrame(data=wine_data['data'], columns = wine_data['feature_names'])
    #data['target'] = wine_data['target']

    data = pd.read_csv(Path(__file__).parent / "data" / "heart_cleveland_upload.csv")
    print(data.head())
    print(data.isna().sum())
    data = data.dropna()
    print(data.describe().T)
    print(data['sex'].value_counts())
    print(data['cp'].value_counts())
    # plt.figure(figsize=(8,10))
    # sns.countplot(x='sex', hue='condition', data = data.astype('string'), hue_order=data['condition'].astype('string').value_counts(ascending=False).index,  order=data['sex'].astype('string').value_counts(ascending=True).index)
    # plt.xlabel('gender 0 - F, 1 - M')
    # plt.ylabel('freq')
    # plt.show()

    print(data.loc[:, ['sex', 'condition']].loc[data['sex'] == 1].value_counts())
    print(data.loc[:, ['sex', 'condition']].loc[data['sex'] == 0].value_counts())

    # splitting data into training sets was skipped

    # if model does not learn properly and predictions are close to zero, then this step might be missing
    standard_scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance.

    features = data.drop(columns=['condition'])
    target = data.loc[:, 'condition']

    # if target has multiple columns use:
    # target = tf.keras.utils.to_categorical(target, 3)

    num_features = pd.DataFrame(standard_scaler.fit_transform(features), columns=features.columns, index=features.index)


    x_train, x_test, y_train, y_test = train_test_split(num_features, target, test_size=0.2, random_state=1)

    print((x_train.shape, y_train.shape, x_test.shape, y_test.shape,))

    # use Dataset to take advantage of tf parallel processing
    # in production Dataset MUST be used instead of plain numpy arrays
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values))
    dataset_train = dataset_train.batch(16)
    dataset_train.shuffle(128)

    dataset_val = tf.data.Dataset.from_tensor_slices((x_test.values, y_test.values))
    dataset_val = dataset_val.batch(16)

    model = build_model_fun(x_train)

    print(model.summary())
    tf.keras.utils.plot_model(model, show_shapes=True) # used pydot and graphviz libs, prints to model.png file

    num_epochs = 100
    training_history = model.fit(dataset_train, epochs=num_epochs,
                                 #validation_split=0.2, # unsupported for dataset_train object
                                 #batch_size=100, # default 32, don't use with generators
                                 verbose=True)


    plt.figure(figsize=(16,16))
    params = [_ for _ in training_history.history.keys() if not _.startswith("val_")]
    for idx, param in enumerate(params):
        plt.subplot(2, math.ceil(len(params)/2), idx + 1)
        plt.plot(training_history.history[param])
        plt.title(f'Model {param}')
        plt.xlabel('epoch')
        plt.ylabel(param)
        plt.legend(['train', "val"])
    plt.show()

    print("eval")
    score = model.evaluate(dataset_val)
    print(pd.Series(score, index=model.metrics_names))
    y_pred = model.predict(x_test)
    y_pred_th = np.where(y_pred>=0.5, 1, y_pred)
    y_pred_th = np.where(y_pred<0.5, 0, y_pred_th)

    res_df = pd.DataFrame({'y_test': y_test.values.flatten(), "y_pred": y_pred_th.flatten().astype('int32')},
                         index=range(len(y_pred)))
    print(res_df)

    # print confusion matrix
    print(pd.crosstab(res_df.y_pred, res_df.y_test))

    print("r2")
    r2= r2_score(y_test, y_pred_th) # higher is better
    print(r2)
    print("accuracy")
    print(sklearn.metrics.accuracy_score(y_test, y_pred_th))


if __name__ == '__main__':
    test1()
