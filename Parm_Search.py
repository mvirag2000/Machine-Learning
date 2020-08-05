import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.stats import reciprocal 
from sklearn.model_selection import GridSearchCV
np.set_printoptions(linewidth=200)
print(tf.__version__)
np.random.seed(1)
rng = default_rng()

DESIRED_ACCURACY = 0.90

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}): # Best to check at END of epoch 
    if(logs.get('accuracy') is not None and logs.get('accuracy') >= DESIRED_ACCURACY):
      print("\nReached " + str(DESIRED_ACCURACY) + " accuracy so cancelling training!") 
      self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

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

#model = build_model(1, 256, 0.001) 
#model.fit(training_images, training_labels, epochs=10, verbose=2, callbacks=[callbacks])

keras_wrap = keras.wrappers.scikit_learn.KerasClassifier(build_model)

parm_distro = {
    "n_hidden" : [1, 2],
    "n_neurons" : [256, 512, 1024], 
    "learning_rate" : [0.0001, 0.0005, 0.00005]
    }

search = GridSearchCV(keras_wrap, parm_distro, cv=3)
search.fit(training_images, training_labels, epochs=100, verbose=2, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3)])
# print(search.best_params_)

report(search.cv_results_)
# Model with rank: 1
# Mean validation score: 0.899 (std: 0.003)
# Parameters: {'learning_rate': 5e-05, 'n_hidden': 1, 'n_neurons': 1024}

pick = rng.integers(test_images.shape[0])
guess = search.predict(test_images[pick].reshape((1,28,28)))
print(guess)
print(test_labels[pick])
plt.imshow(test_images[pick], cmap='binary')
plt.show()