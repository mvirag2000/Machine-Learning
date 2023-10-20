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

train_df = pd.read_csv("train.csv")

all_vars = []
for col in train_df.columns:
    all_vars.append(train_df[col].name)
cat_vars = all_vars[10:17]
num_vars = all_vars[1:10]
print(cat_vars)
print(num_vars)

train_set, test_set = train_test_split(train_df, test_size=0.20, random_state=42)
train_temp = train_set.drop('Default', axis=1)
train_y = train_set['Default'].copy() 
test_temp = test_set.drop('Default', axis=1)
test_y = test_set['Default'].copy() 

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', MinMaxScaler())
    ])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_vars),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_vars) 
    ]) 

train_data = full_pipeline.fit_transform(train_temp)
test_data = full_pipeline.transform(test_temp) # Do NOT re-fit 
feature_names = num_vars + full_pipeline.named_transformers_['cat'].get_feature_names_out(input_features=cat_vars).tolist()

### MODELS START HERE ###

def result_roc(name, pred_y):
    fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
    display.plot()
    plt.show()
    
def neural_net(n_hidden=3, n_neurons=256, learning_rate=0.001):
    print("\n*** Hidden: "+ str(n_hidden))
    print("*** Neurons: " + str(n_neurons))
    print("*** Learning Rate: " + str(learning_rate))               
    # train_data_dense = np.asarray(train_data) Not sparse 
    # test_data_dense = np.asarray(test_data)
    model = keras.models.Sequential()
    model.add(keras.Input(train_data.shape[1],))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1)) 
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')
    model.summary()
    history = model.fit(train_data, train_y, validation_data=(test_data, test_y), epochs=3, verbose=2)
    return model, history

def history_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model MSE')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

forest_class = RandomForestClassifier(max_depth=2, n_estimators=1000, criterion="entropy").fit(train_data, train_y) # 0.73
pred_y = forest_class.predict_proba(test_data)[:, 1]
result_roc("Forest Classifier", pred_y) 

lin_reg = LinearRegression().fit(train_data, train_y)  # 0.75 
pred_y = lin_reg.predict(test_data)
result_roc("Linear Regression", pred_y)

forest_reg = RandomForestRegressor(max_depth=4).fit(train_data, train_y) # 0.72
pred_y = forest_reg.predict(test_data)
result_roc("Random Forest", pred_y)

nn_model, history = neural_net() # 0.76
pred_y = nn_model.predict(test_data)
result_roc("Neural Net", pred_y)
history_plot(history)