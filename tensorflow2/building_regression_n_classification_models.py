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

# import tensorflow_docs as tfdocs
# import tensorflow_docs.modeling
# import tensorflow_docs.plots

matplotlib.use('QtAgg')


def test_data_prep1():
    pd.set_option('display.max_columns', None)
    data = pd.read_csv(Path(__file__).parent / "data" / "insurance.csv")
    print(data.head())
    print(data.shape)
    print(data.isna().sum())
    print(data[['age', 'bmi', 'charges']].describe().T)


    # plot kernel density estimation
    #data['charges'].plot.kde() # gives probability distribution of data
    #plt.show()
    features = data.drop('charges', axis=1)
    target = data[['charges']]
    categorical_features = features[['sex', 'smoker', 'region']].copy()
    categorical_features['sex'].replace({'female':0, 'male':1}, inplace=True) # use for label encoding
    numeric_features = features.drop(['sex', 'smoker', 'region'], axis=1)
    categorical_features['smoker'].replace({'no':0, 'yes':1}, inplace=True)
    categorical_features = pd.get_dummies(categorical_features, columns=['region']).astype('int32')
    print(categorical_features.head())

    standard_scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance.
    num_features = pd.DataFrame(standard_scaler.fit_transform(numeric_features), columns=numeric_features.columns, index=numeric_features.index)
    processed_features = pd.concat([num_features, categorical_features], axis=1, sort=False)

    print(processed_features.head())

    pd.reset_option('display.max_columns')
    ...


def test1():
    ...


if __name__ == '__main__':
    test1()