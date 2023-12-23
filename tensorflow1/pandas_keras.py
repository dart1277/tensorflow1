import json
import os, datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

import keras
from keras import layers

matplotlib.use('QtAgg') #  MacOSX, QtAgg, GTK4Agg, Gtk3Agg, TkAgg, WxAgg

from pathlib import Path

from functional_api import build_model_fun

# import tensorflow_docs as tfdocs
# import tensorflow_docs.modeling
# import tensorflow_docs.plots


def build_single_layer_model(x_train):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, input_shape=(x_train.shape[1],),activation="relu")) # activation="sigmoid"
    model.add(tf.keras.layers.Dropout(rate=0.1)) # avoids overfitting
    model.add(tf.keras.layers.Dense(16,activation="relu")) # activation="sigmoid", "elu" - mitigates issue of saturating neuron, converges faster
    model.add(tf.keras.layers.Dense(16,activation="relu")) # activation="sigmoid"
    model.add(tf.keras.layers.Dense(16,activation="relu")) # activation="sigmoid"
    model.add(tf.keras.layers.Dense(16,activation="relu")) # activation="sigmoid"
    model.add(tf.keras.layers.Dense(1))
    # or use SGD optimizer
    # or use RMSprop optimizer - optimizes gradients, use with elu activation function
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.

    model.compile(
        loss="mse",
        metrics=["mae", "mse"],
        optimizer=optimizer
    )
    return model

def test1():
    data = pd.read_csv(Path(__file__).parent / "data" / "life_exp_data.csv")
    #print(data.head())
    #print(data.sample(5))
    #print(data.isna().sum()) # null or missing cnt
    countries = data['Country'].unique()
    na_columns = data.isna().columns
    for col in na_columns:
        for country in countries:
            if pd.api.types.is_numeric_dtype(data.dtypes[col]):
                val = data.loc[data['Country'] == country, col]
                data.loc[data['Country'] == country, col] = val.fillna(val.mean())
    data = data.dropna()
    #print(data.shape)
    #print(data.isna().sum())
    #print(data.index[data.isna().any(axis=1)])
    print(type(data.isna().any(axis=1)))
    #print(data['Status'].value_counts())

    #plt.figure(figsize=(10,8))
    #data.boxplot('Life expectancy ')
    #plt.show()

    # plt.figure(figsize=(8, 6))
    # data['Life expectancy '].hist(bins=20, color='skyblue', edgecolor='black')
    # plt.title('Histogram of Values')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.show()

    # grouped_df = data.loc[:,['Country', 'Life expectancy ']].groupby(['Country']).agg({'Life expectancy ': 'mean'}).sort_values(by=["Life expectancy "], ascending=[False])
    # print(grouped_df.head(20))
    # print()
    # print(grouped_df.tail(20))

    # plt.figure(figsize=(8,6))
    # sns.boxplot(x='Status', y='Life expectancy ', data=data)
    # plt.xlabel('Status', fontsize=16)
    # plt.ylabel('Total expenditure', fontsize=16)
    # plt.show()

    # plt.figure(figsize=(8,6))
    # sns.boxplot(x='Status', y='Total expenditure', data=data)
    # plt.xlabel('Status', fontsize=16)
    # plt.ylabel('Total expenditure', fontsize=16)
    # plt.show()

    # find correlations between data columns
    # pd.set_option('display.max_columns', None)
    # corr = data[['Life expectancy ', 'GDP', 'Adult Mortality', 'Total expenditure', 'Population', 'Schooling']].corr()
    # print(corr)
    # pd.reset_option('display.max_columns')
    #
    # sns.set_palette(palette="Spectral")
    # sns.set_palette(palette="tab10")
    #
    # fig, ax = plt.subplots(figsize=(12, 8))
    # sns.set(font_scale=0.8)
    # sns.heatmap(corr, annot=True, cmap="copper", fmt='.1g',  xticklabels=True, yticklabels=True) #gist_gray cividis plasma viridis, tab10, husl, Set2, Spectral, flare, # dont work: blend:#7AB,#EDA, ch:s=.25,rot=-.25
    # plt.show()

    pd.set_option('display.max_columns', None)
    features = data.drop('Life expectancy ', axis=1)
    features = features.drop('Country', axis=1)
    category_features = features['Status'].copy()
    category_features = pd.get_dummies(category_features)
    target = data[['Life expectancy ']]
    numeric_features = features.drop("Status", axis=1)
    #print(category_features)
    #print(numeric_features.describe().T)

    # perform standardisation
    standard_scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance.

    num_features = pd.DataFrame(standard_scaler.fit_transform(numeric_features), columns=numeric_features.columns, index=numeric_features.index)

    print(num_features.describe().T) # transpose

    processed_features = pd.concat([num_features, category_features.astype("float32")], axis=1, sort=False)

    # when concatenating on axis=0 make index start at 0 and monotonically increase
    # processed_features.reset_index(drop=True, inplace=True)

    # data["some_date_column"] = pd.to_datetime(data["some_date_column"])

    # data["some categorical var column"].map({"cat1":!, "cat2":2}) # map column labels to categories, then can use one hot encoding
    # for ex. target = tf.keras.utils.to_categorical(target, 3)
    # or
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()
    # data["country"] = le.fit_transform(data["country"])
    # one hot encoding
    # cat_data = pd.get_dummies(data=data, columns=["col1", "col2"])
    # drop categorical data
    # num_data = data.drop(columns=cat_data.columns, axis=1)

    # find correlation matrix
    # thresh = 0.85
    # c_mat = final_df.corr().abs()
    # upper_triangle_corr_matrix = c_mat.where(np.triu(np.ones(c_mat.shape), k=1).astype(bool))
    #
    # drop highly correlated columns
    # to_drop = []
    # for i in range(len(corr_matrix.columns)):
    #     for j in range(i):
    #         if abs(corr_matrix.iloc[i,j]) >= thresh
    #             colname = corr_matrix.columns[i]
    #             to_drop.append(colname)
    # final_df.drop(columns=to_drop, inplace=True)
    # final_df["agent"] = final_df["agent"].fillna(0)

    print(processed_features.shape)

    pd.reset_option('display.max_columns')

    x_train, x_test, y_train, y_test = train_test_split(processed_features, target, test_size=0.2, random_state=1)
    print((x_train.shape, y_train.shape, x_test.shape, y_test.shape, ))
    model = build_single_layer_model(x_train)
    # model = build_model_fun(x_train)
    print(model.summary())
    tf.keras.utils.plot_model(model, show_shapes=True) # used pydot and graphviz libs, prints to model.png file

    logdir = os.path.join("seq_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    num_epochs = 1000
    training_history = model.fit(x_train, y_train, epochs=num_epochs,
                                 validation_split=0.2, # if using numpy arrays
                                 batch_size=100, # default 32, don't use with generators
                                 verbose=True, callbacks=[tensorboard_callback,
                                                          # tfdocs.modeling.EpochDots()
                                                          tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) # stop early if no progress is made for 5 epochs
                                                          ])

    # plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    # plt.figure(figsize=(16,8))
    # plotter.plot({'Basic': training_history}, metric="mae")
    # plt.show()

    # use callback output,
    # in jupyter:
    # load_ext tensorboard
    # tensorboard --logdir ./seq_logs --port 6050

    #print(training_history)

    # plt.figure(figsize=(16,8))
    # plt.subplot(1,2,1)
    # plt.plot(training_history.history["mae"])
    # plt.plot(training_history.history["val_mae"])
    # plt.title('Model MAE')
    # plt.xlabel('epoch')
    # plt.ylabel('mae')
    # plt.legend(['train', "val"])
    #
    # plt.subplot(1,2,2)
    # plt.plot(training_history.history["loss"])
    # plt.plot(training_history.history["val_loss"])
    # plt.title('Model loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(['train', "val"])
    #
    # plt.show()

    print("eval")
    model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    print("r2")
    r2= r2_score(y_test, y_pred) # higher is better, used for regression models
    print(r2)

    pred_res = pd.DataFrame({
        'y_test': y_test.values.flatten(),
        'y_pred': y_pred.flatten()
    },
        index=range(len(y_pred))
    )
    print(pred_res.sample(10))

    plt.figure(figsize=(10,8))
    plt.scatter(y_test, y_pred, s=2, c='blue') # is is the dot size
    plt.xlabel('Actual life expectancy')
    plt.ylabel('Predicted life expectancy')
    plt.show()

    # Keras has 2 file formats
    # TF SavedModel (recommended)
    # h5 - external losses and metrics are not saved
    # - computation graph of custom objects not saved

    # saving whole model
    # - model.save
    # - tf.keras.models.save_model
    # - tf.keras.models.load_model
    model.save('./my_models/relu_1/model_all')
    # model = tf.keras.models.load_model('./my_models/relu_1/model_all')


    # saving model architecture
    # - get_config / from_config
    # tf.keras.models.model_to_json
    # tf.keras.models.model_from_json
    # custom objects/layers must override get_config and from_config
    # custom functions need not override get_config
    print(json.dumps(json.loads(model.to_json()), indent=2))
    # with open('./my_models/reulu.json') as infile:
    #     model_json = json.load(infile)
    #     model = tf.keras.models.model_from_json(json.dumps(model_json))
    #     model.summary()

    # checkpointing (saving weights)
    # formats
    # - tensorflow checkpoint
    # - HDF5
    # apis (only works with non-custom models)
    # tf.keras.layers.Layer.get_weights
    # tf.keras.layers.Layer.set_weights
    model.save_weights('./my_models/relu_1',
                       #save_format='h5' # can be used with different backends
                       )

    # model.load_weights('./my_models/relu_1')

    ...

if __name__ == '__main__':
    test1()