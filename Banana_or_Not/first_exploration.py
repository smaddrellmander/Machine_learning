# A python script to start exploring MNIST
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Number of data points to:\nTrain: {0}\nTest: {1}\nValidate: {2}"\
        .format(len(mnist.train.images), len(mnist.test.images), len(mnist.validation.images)))

def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt

# Get a batch of two random images and show in a pop-up window.
batch_xs, batch_ys = mnist.train.next_batch(1000)
X_val, y_val = mnist.validation.next_batch(200)
X_test, y_test = mnist.test.next_batch(1000)
# gen_image(batch_xs[0]).show()
# gen_image(batch_xs[1]).show()


import keras
from keras.models import Sequential
import keras.layers as ll

model = Sequential(name="cnn")

model.add(ll.InputLayer([1,28,28]))

model.add(ll.Flatten())

#network body
model.add(ll.Dense(25))
model.add(ll.Activation('linear'))

model.add(ll.Dropout(0.9))

model.add(ll.Dense(25))
model.add(ll.Activation('linear'))

#output layer: 10 neurons for each class with softmax
model.add(ll.Dense(10,activation='softmax'))

model.compile("adam","categorical_crossentropy",metrics=["accuracy"])

print(model.summary())

# Shaoe of data here is a problem. What form is it expecting?
# Allre shaped  to 28*28
model.fit(batch_xs, batch_ys, validation_data=(X_val,y_val), epochs=5)
model.predict_proba(X_val[:2])
print("\nLoss, Accuracy = ",model.evaluate(X_test,y_test))
