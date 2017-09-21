import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras import backend as K
from keras.activations import relu
from keras.regularizers import l2

import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def get_data():
    # How to read in the data
    bananas = []
    rooms = []
    for fn in os.listdir('bananas/'):
        # if os.path.isfile(fn):
        print (fn)
        img = load_img('bananas/'+fn)  # this is a PIL image
        a = img_to_array(img, data_format='channels_last')  # this is a Numpy array with shape (3, 150, 150)
        a = a.reshape((1,) + a.shape)
        bananas.append(a)
    bananas = np.vstack(bananas)
    print(len(bananas))
    for fn in os.listdir('rooms/'):
        # if os.path.isfile(fn):
        print (fn)
        img = load_img('rooms/'+fn)  # this is a PIL image
        a = img_to_array(img, data_format='channels_last')  # this is a Numpy array with shape (3, 150, 150)
        a = a.reshape((1,) + a.shape)
        rooms.append(a)
    rooms = np.vstack(rooms)
    print(len(rooms))
    X_train = bananas[0:50]+rooms[0:50]
    X_test = bananas[50:100]+rooms[50:100]
    y_train = 50*[1]+50*[0]
    y_test = 50*[1]+50*[0]
    return X_train, y_train, X_test, y_test
    pass

def main():
    get_data()
    X_train, y_train, X_test, y_test = get_data()

    print (X_train)
    print (y_train)

    print(X_test)
    print(y_test)

    # # First attempt at using the ImageDataGenerator
    # train_datagen = ImageDataGenerator(rescale=1./255)
    # test_datagen = ImageDataGenerator(rescale=1./255)
    #
    # train_generator = train_datagen.flow_from_directory(
    #                 'train',
    #                 target_size=(32,32),
    #                 batch_size=32,
    #                 class_mode='binary')
    # test_generator = test_datagen.flow_from_directory(
    #                 'test',
    #                 target_size=(32,32),
    #                 batch_size=32,
    #                 class_mode='binary')
    #
    # print(train_generator)
    # Show some of the data
    cols = 10
    rows = 5
    batch_size = 500
    epochs = 75
    input_shape = (32, 32, 3)
    num_classes = 2

    fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
    for i in range(cols):
        for j in range(rows):
            k = np.random.randint(0, X_train.shape[0])

            ax = fig.add_subplot(rows, cols, i*rows+j+1)
            ax.grid('off')
            ax.axis('off')
            ax.set_title('%s' % (class_names[np.where(y_train[k] > 0.0)[0][0]]))
            im = ax.imshow(X_train[k])
    plt.show()

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

    y_predicted = model.predict(X_test)
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
    # plt.show()
    plt.savefig('ACU_ROC_ban.png')

    y_predicted_classes = np.argmax(y_predicted, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)



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
    # plt.show()
    plt.savefig('test_pics_ban.png')
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_cifar_bana.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_cifar_bana.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    main()
