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
    for index, row in y.iterrows():
        print(f"{index}: {row}")
    # reindex columns and rows
    yy = pd.DataFrame(y, columns=['y'], index=[i for i in range(y.shape[0])])

    # replace missing values with interpolation
    yy['y'].interpolate()#axis=1)
    yym = yy.fillna({'y': yy['y'].median()})

    # turn to pivot table
    # Date, Product_A_Sales, Product_B_Sales, Product_C_Sales,...
    # converts to
    # Date, Product, Sales
    # df = pd.melt(yy, id_vars=['Date'], var_name='Product', value_name='Sales')
    # back to pivot table
    # df.pivot(index='Date', columns='Product', values='Sales')
    # y.sort_values(by=['Age', 'Name'], ascending=[False, True])
    # y.sort_index(ascending=False)
    # pd.date_range(start='2023-01-01', end='2023-01-01', freq='D')
    #  np.random.randint(50,100,size=len(y))
    # slice = df['2023-01-01': '2023-02-01']
    # weekly = df['Temp'].resample('W').mean()
    # rolling = df['Temp'].rolling(window=7).mean()
    # shift = df['Temp'].shift(1)

    # df.to_csv('data.csv', index=False)
    # df.to_excel('data.xlsx', index=False, sheet_name='Sheet1')
    # df.to_json('data.json', orient='records')
    # df.to_pickle('data.pkl')
    # df.to_hdf('data.h5', key='df', mode='w')
    # df.to_parquet('data.parquet', index=False)

    # binning in pandas
    # pd.cut(ages, bins=[20,30,40,60,80, float('inf')], labels=["20-30","30-40", '40-60', '60-80', '80-'], right=True, inlude_lowest=True)
    # pd.qcut(ages, q=[0, 0.33, 0.66, 1], labels=['low', 'mid', 'hi'])

    # join operations
    # inner = pd.merge(df1, df2, on='ID', how='inner') # 'right', 'outer'

    # crosstab function in pandas
    # Gender(M/F), Education(A,B), Satisfaction(1-5)
    # is converted to:
    # Satisfaction          1 2 3 4 5 Total
    # (Gender Education)
    # Female A              0 0 0 1 5 6
    #        B
    # Male   A
    #        B
    # Total                           6
    # pd.crosstab(index=[survey['Gender'], survey['Education']],
    #             columns=survey['Satisfaction'],
    #             margins=True, # Add row/column margins (subtotals)
    #             margins_name='Total'
    #             )


    # grouped_data = y.groupby('y').sum()
    # y.reset_index(inplace=True, drop=True)
    # y['y'].rolling(window=3).mean()
    # y['y'].mean()
    # y['y'].median()
    # y['y'].std()
    # y['y'].quantile(0.25)
    # y['y'].quantile(0.75)


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