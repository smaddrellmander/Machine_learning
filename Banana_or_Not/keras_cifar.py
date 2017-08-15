import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras import backend as K
from keras.activations import relu
from keras.regularizers import l2

import matplotlib.pyplot as plt
import numpy as np

from fetch_cifar import fetch_cifar_dataset

def main():
    X_train, y_train, X_test, y_test, class_names = fetch_cifar_dataset()

    print(X_train.shape)
    print(y_train.shape)

    print(X_test.shape)
    print(y_test.shape)

    # Show some of the data
    cols = 10
    rows = 5
    batch_size = 32
    epochs = 1
    input_shape = (32, 32, 3)
    num_classes = 10

    fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
    for i in range(cols):
        for j in range(rows):
            k = np.random.randint(0, X_train.shape[0])

            ax = fig.add_subplot(rows, cols, i*rows+j+1)
            ax.grid('off')
            ax.axis('off')
            ax.set_title('%s' % (class_names[np.where(y_train[k] > 0.0)[0][0]]))
            im = ax.imshow(X_train[k])
    # plt.show()

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) # Reducing size by factor 2^2
    model.add(Dropout(0.25))
    model.add(Flatten()) # To allow our normal NN to work on the vector
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax')) # The output layer

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer = keras.optimizers.Adadelta(),
                  metrics = ['accuracy'])

    logger = model.fit(X_train, y_train,
                        batch_size = batch_size,
                        epochs = epochs,
                        verbose = 1,
                        validation_data = (X_test, y_test))

    y_predicted = cnn.predict(X_test)
    print('Accuracy:', np.mean( np.argmax(y_predicted, axis=1) ==  np.argmax(y_test, axis=1)))
    from sklearn.metrics import roc_curve, auc

    fprs, tprs = [None] * 10, [None] * 10
    aucs = [None] * 10

    for i in range(10):
        fprs[i], tprs[i], _ = roc_curve(y_test[:, i], y_predicted[:, i])
        aucs[i] = auc(fprs[i], tprs[i], reorder=True)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], '--', color='black')

    plt.title('One-vs-rest ROC curves', fontsize=16)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    for i in range(10):
        plt.plot(fprs[i], tprs[i], label='%s (AUC %.2lf)' % (class_names[i], aucs[i]))

    plt.legend(fontsize=14)
    plt.show()

    y_predicted_classes = np.argmax(y_predicted, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    from sklearn.metrics import confusion_matrix

    c_matrix = confusion_matrix(y_true_classes, y_predicted_classes)


    ### Diagonal excluded
    c_matrix[np.arange(10), np.arange(10)] = 0.0

    plt.figure(figsize=(10, 8))
    plt.title('Confusion matrix (diagonal excluded)', fontsize=16)
    plt.imshow(c_matrix)
    plt.xticks(np.arange(10), class_names, rotation=45, fontsize=14)
    plt.yticks(np.arange(10), class_names, fontsize=14)
    plt.colorbar()
    plt.show()

    cols = 7
    rows = 5
    fig = plt.figure(figsize=(3 * cols - 1, 4 * rows - 1))

    for i in range(cols):
        for j in range(rows):
            k = np.random.randint(0, X_test.shape[0])

            y_pred = y_predicted[k]
            predicted_class = np.argmax(y_pred)
            real_class = np.argmax(y_test[k])
            score = y_pred[predicted_class]

            ax = fig.add_subplot(rows, cols, i * rows + j + 1)
            ax.grid('off')
            ax.axis('off')
            if real_class == predicted_class:
                ax.set_title('%s\nscore: %.3lf' % (class_names[real_class], score))
            else:
                ax.set_title('real: %s;\npredicted: %s\nwith score: %.3lf' % (
                    class_names[real_class], class_names[predicted_class], score
                ))
            im = ax.imshow(X_test[k])
    plt.show()

if __name__ == '__main__':
    main()
