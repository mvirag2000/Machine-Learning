from pyexpat import model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import tensorflow as tf

# Load your data from a CSV file into a DataFrame
data = pd.read_csv('train.csv')  # Replace 'your_data.csv' with your actual CSV file

# Define your features (independent variables) and the target variable
X = data.drop(columns=['Default'])  # Features
y = data['Default']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numeric and categorical feature columns
numeric_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']  # Replace with your numeric columns
categorical_features = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']  # Replace with your categorical columns

# Create data preprocessing transformers for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

def create_neural_network(num_hidden_layers, output_dim):
    # Define a Sequential model
    model = tf.keras.Sequential()

    # Add an input layer (input_dim will be inferred from the data)
    model.add(tf.keras.layers.Input(shape=(31,)))

    # Add the specified number of hidden layers with 256 units each
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(256, activation='relu'))
 
    # Add an output layer with the specified output dimension
    model.add(tf.keras.layers.Dense(output_dim, activation='linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error') 

    return model

# Create the full preprocessing and modeling pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', create_neural_network(num_hidden_layers=4, output_dim=1))  # Use the function from the previous answer
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
