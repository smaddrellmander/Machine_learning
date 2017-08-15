import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json


def main():
    # Start with parameter values
    batch_size = 128
    num_classes = 10
    epochs = 1
    # Visualisation list
    Ns = np.random.randint(low=0, high=100, size=10)

    # nput image dimensions
    img_rows, img_cols = 28, 28
    # Pull the data straight from the KERAS module
    (x_train_, y_train_), (x_test_, y_test_) = mnist.load_data()
    # Here just reducing the amount of data when running on CPU
    # Still gets surprsingly good results

    y_train = y_train_[0:]
    y_test = y_test_[0:]

    # Extra validation for Visualisation
    x_val = x_test_[200:300]
    x_vis = x_val
    y_val = y_test_[200:300]

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_test = x_test.astype('float32')
    x_test /= 255
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                         optimizer = keras.optimizers.Adadelta(),
                         metrics = ['accuracy'])


    score = loaded_model.evaluate(x_test, y_test, verbose = 0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    for N in Ns:
        test = loaded_model.predict(x_val[N:N+1], batch_size=1)
        print('Prediction:', np.argmax(test), 'Correct answer:', y_val[N])
        plt.imshow(x_vis[N])
        plt.show()



if __name__ == '__main__':
    main()
