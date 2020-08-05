import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

stock_prices = genfromtxt('prices.csv', delimiter=',')
x = stock_prices[1:, 4:]
y = stock_prices[1:, 3]

def shuffle(x, y):     # Randomize X and Y values together 
    m = x.shape[0]
    assert m == y.shape[0] # Must have same number of rows
    permutation = list(np.random.permutation(m))
    new_x = x[permutation, :]
    new_y = y[permutation]
    return new_x, new_y

def split(x, y, pct=20):    # Split X and Y into train and test sets
    m = x.shape[0]
    assert m == y.shape[0] # Must have same number of rows
    top = np.int(m * pct / 100)
    test_x = x[:top, :]
    test_y = y[:top]
    train_x = x[top:, :]
    train_y = y[top:]
    return train_x, train_y, test_x, test_y

x, y = shuffle(x, y)
train_x, train_y, test_x, test_y = split(x, y, 20)

train_x = train_x.reshape(-1, 10, 1)
test_x = test_x.reshape(-1, 10, 1)
train_y = train_y.reshape(-1, 1, 1)
test_y = test_y.reshape(-1, 1, 1)

print ("train_x shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x shape: " + str(test_x.shape))
print ("test_y shape: " + str(test_y.shape))

model = keras.Sequential([
    layers.LSTM(16, return_sequences = False, input_shape=(10,1)),
    layers.Dense(8),
    layers.Dense(1)
])
model.summary()
model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics='accuracy')
history = model.fit(train_x, train_y, epochs=100, verbose=2, validation_data=(test_x, test_y)) 

predicted = model.predict(test_x) * 100
print(predicted.shape)
test_y = test_y.reshape(-1, 1) * 100
plt.scatter(test_y, predicted)
plt.title('Daily Pct. Change in SPY')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()
