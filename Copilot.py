import pandas as pd
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler 
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
import seaborn as sns
sns.set_theme() 
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
train_data = []
test_data = []
train_y = []
test_y = []
    
'''read file from disk and create dataframe'''
def read_file_df(filename):
    with open(filename, 'r') as f:
        return pd.read_csv(f)
    

    
def create_neural_network(num_hidden_layers, output_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(31,)))
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(output_dim, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error') 
    return model
    
def neural_net(n_hidden=3, n_neurons=256, learning_rate=0.005):
    model = keras.models.Sequential()
    model.add(keras.Input(train_data.shape[1],))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')
    model.summary()
    history = model.fit(train_data, train_y, 
        validation_data=(test_data, test_y), epochs=3, verbose=2)
    return model, history




    
    
    
    
    
    
    
    