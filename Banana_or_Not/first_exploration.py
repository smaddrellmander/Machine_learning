# A python script to start exploring MNIST
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Number of data points to:\nTrain: {0}\nTest: {1}\nValidate: {2}"\
        .format(len(mnist.train.images), len(mnist.test.images), len(mnist.validation.images)))

def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt

# Get a batch of two random images and show in a pop-up window.
batch_xs, batch_ys = mnist.test.next_batch(2)
gen_image(batch_xs[0]).show()
gen_image(batch_xs[1]).show()
