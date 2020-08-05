import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(linewidth=200)
print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0
test_m = test_labels.shape[0]
train_m = training_labels.shape[0]

# Explore the dataset
print("INPUT")
print ("train_x shape: " + str(training_images.shape))
print ("train_y shape: " + str(training_labels.shape))
print ("test_x shape: " + str(test_images.shape))
print ("test_y shape: " + str(test_labels.shape))

def build_model(n_hidden=1, n_neurons=256, learning_rate=0.001):
    print("*** Hidden: "+ str(n_hidden))
    print("*** Neurons: " + str(n_neurons))
    print("*** Learning Rate: " + str(learning_rate))
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation=tf.nn.relu))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model(1, 1024, 0.00005) 
history = model.fit(training_images, training_labels, validation_data=(test_images, test_labels), epochs=100, verbose=2)

# Old school results checking 
train_results = model.predict(training_images.reshape((-1,28,28))).argmax(axis=1)
train_correct = (train_results == training_labels)
train_pct = sum(train_correct) * 100 / train_m

test_results = model.predict(test_images.reshape((-1,28,28))).argmax(axis=1)
test_correct = (test_results == test_labels)
test_pct = sum(test_correct) * 100 / test_m

print("\nRESULTS")
print("Train accuracy: " + str(train_pct))
print("Test accuracy: " + str(test_pct))

# History plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()