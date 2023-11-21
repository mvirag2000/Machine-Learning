import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler 
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm  
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
y_train = train_set['Default'].copy() 
test_temp = test_set.drop('Default', axis=1)
y_test = test_set['Default'].copy() 

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', MinMaxScaler())
    ])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_vars),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_vars) 
    ]) 

X_train = full_pipeline.fit_transform(train_temp)
X_test = full_pipeline.transform(test_temp) # Do NOT re-fit 
feature_names = num_vars + full_pipeline.named_transformers_['cat'].get_feature_names_out(input_features=cat_vars).tolist()
print(feature_names)

### MODELS START HERE ###
   
def neural_net(n_hidden=4, n_neurons=256, learning_rate=0.005):
    print("\n*** Hidden: "+ str(n_hidden))
    print("*** Neurons: " + str(n_neurons))
    print("*** Learning Rate: " + str(learning_rate))               
    # X_train_dense = np.asarray(X_train) Not sparse 
    # X_test_dense = np.asarray(X_test)
    model = keras.models.Sequential()
    model.add(keras.Input(X_train.shape[1],))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
        # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1, activation='linear'))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')
    model.summary()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, verbose=2)
    return model, history

def history_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model MSE')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

nn_model, history = neural_net(3, 256) # 0.76
y_pred = nn_model.predict(X_test)
history_plot(history)

'''
forest_class = RandomForestClassifier(max_depth=4, n_jobs=-1, verbose=1).fit(X_train, y_train) # 0.73
y_pred = forest_class.predict_proba(X_test)[:, 1]

forest_reg = RandomForestRegressor(max_depth=4).fit(X_train, y_train) # 0.72
y_pred = forest_reg.predict(X_test)
result_roc("Random Forest", y_pred)

nn_model, history = neural_net(3, 256) # 0.76
y_pred = nn_model.predict(X_test)
history_plot(history)

model = GradientBoostingClassifier(n_estimators=500, max_depth=2).fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]

# lin_reg = LinearRegression(n_jobs=-1).fit(X_train, y_train)  # 0.75 
model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_test)
print(model.summary())

'''

def result_roc(y_pred, y_test, name):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) for ' + name)
    plt.legend(loc='lower right')
    plt.show()

def confusion_chart(test_results):
    P = y_test.sum()
    N = y_test.count() - P
    specificity = []
    sensitivity = []
    cutoff = np.arange(.01, 1, .01)
    for t in cutoff:
        test_results["Bool"] = (test_results["Proba"] >= t) * 1
        test_results.eval("TP = ((Actual == 1) and (Bool == 1)) * 1", inplace=True)
        test_results.eval("TN = ((Actual == 0) and (Bool == 0)) * 1", inplace=True)
        TP = test_results["TP"].sum()
        TN = test_results["TN"].sum()
        TPR = TP / P if P > 0 else 1
        TNR = TN / N if N > 0 else 1 
        specificity.append(TNR)
        sensitivity.append(TPR)

    plt.figure()
    plt.plot(cutoff, specificity, color='darkorange', lw=2, label='Specificity (TNR)')
    plt.plot(cutoff, sensitivity, color='navy', lw=2, label='Sensitivity (TPR)')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Threshold Value')
    plt.ylabel('True Pos/Neg Rate')
    plt.title('Truth Rates v. Proba Cutoff')
    plt.legend(loc='center right')
    plt.show()

result_roc(y_pred, y_test, "Neural Net")    

test_results = pd.DataFrame({"Proba":y_pred.flatten(), "Actual":y_test})
test_results.to_csv("neural.csv", index=False)
confusion_chart(test_results) 

# Plot sample cases 
sample = train_set.sample(n=200, random_state=42)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
mask = sample["Default"] == 1 
ax.scatter(sample.loc[mask, 'Age'], sample.loc[mask, 'InterestRate'], sample.loc[mask, 'CreditScore'], c='red', label='1', marker='o')
ax.scatter(sample.loc[-mask, 'Age'], sample.loc[-mask, 'InterestRate'], sample.loc[-mask, 'CreditScore'], c='green', label='0', marker='o')
ax.set_xlabel('Age')
ax.set_ylabel('InterestRate')
ax.set_zlabel('CreditScore')
ax.set_title('3D Scatter Plot')
def rotate(angle): 
    ax.view_init(elev=25, azim=angle)
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 1))
plt.show()