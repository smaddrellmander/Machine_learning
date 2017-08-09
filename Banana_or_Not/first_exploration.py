# A python script to start exploring MNIST
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Number of data points to:\nTrain: {0}\nTest: {1}\nValidate: {2}"\
        .format(len(mnist.train.images), len(mnist.test.images), len(mnist.validation.images)))
