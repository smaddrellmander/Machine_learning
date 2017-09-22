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
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for fn in os.listdir('bananas/'):
        # if os.path.isfile(fn):
        # print (fn)
        img = load_img('bananas/'+fn)  # this is a PIL image
        a = img_to_array(img, data_format='channels_last')  # this is a Numpy array with shape (3, 150, 150)
        a = a.reshape((1,) + a.shape)
        bananas.append(a)
    # bananas = np.vstack(bananas)
    # bananas = bananas.reshape((1,) + bananas.shape)
    print(len(bananas))
    for fn in os.listdir('rooms/'):
        # if os.path.isfile(fn):
        # print (fn)
        img = load_img('rooms/'+fn)  # this is a PIL image
        a = img_to_array(img, data_format='channels_last')  # this is a Numpy array with shape (3, 150, 150)
        a = a.reshape((1,) + a.shape)
        rooms.append(a)
    # rooms = np.vstack(rooms)
    # rooms = rooms.reshape((1,) + rooms.shape)
    print(len(rooms))
    # TODO: Starting here with a very small sample size.
    X_train = np.vstack(bananas[0:500]+rooms[0:500])
    X_test = np.vstack(bananas[500:700]+rooms[500:700])
    y_train = np.vstack(500*[[0,1]]+500*[[1,0]])
    y_test = np.vstack(200*[[0,1]]+200*[[1,0]])
    return X_train, y_train, X_test, y_test
    pass

def get_image(PATHTOIMAGE):
    size = 32, 32
    img = load_img(PATHTOIMAGE)
    img = img.resize(size)
    a = img_to_array(img, data_format='channels_last')
    a = a.reshape((1,) + a.shape)
    return a

def test_on_image(model, image_path):
    class_names = ['rooms', 'bananas']
    i = get_image(image_path)
    fig2 = plt.figure()
    my_predicted = model.predict(i)
    my_pred = my_predicted[0]
    my_predicted_class = np.argmax(my_pred)
    my_real_class = 1 # Assume bananas
    my_score = my_pred[my_predicted_class]
    ax1 = fig2.add_subplot(111)
    if my_real_class == my_predicted_class:
        ax1.set_title('%s\nscore: %.3lf' % (class_names[my_real_class], my_score))
    else:
        ax1.set_title('real: %s;\npredicted: %s\nwith score: %.3lf' % (
            class_names[my_real_class], class_names[my_predicted_class], my_score
        ))
    im1 = ax1.imshow(i[0]/255)
    # outName = 'save_'+str(n)+'.png'
    # plt.savefig(outName)
    plt.show()



def main():
    get_data()
    X_train, y_train, X_test, y_test = get_data()
    # print (X_test)
    # print (y_test)

    print(len(X_test), 'length X')
    print(len(y_test), 'length y')

    # TODO:
    # What we want here is to on hot encode the data. With binary this is actually quite easy
    # How best to do this?
    # Simplest way to encode in the way I was classifying already
    # Now [0,1] and [1,0] this should work fine

    # Show some of the data
    cols = 8
    rows = 7
    batch_size = 150
    epochs = 500
    input_shape = (32, 32, 3)
    num_classes = 2

    fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
    for i in range(cols):
        for j in range(rows):
            k = np.random.randint(0, X_train.shape[0])

            ax = fig.add_subplot(rows, cols, i*rows+j+1)
            ax.grid('off')
            ax.axis('off')
            # ax.set_title('%s' % (class_names[np.where(y_train[k] > 0.0)[0][0]]))
            # THERE IS DEFINETLY SOME SCALING HERE NOT WORKING
            im = ax.imshow(X_train[k]/255)
    plt.show()

    # print(X_train[0])
    # print(X_test[0])
    print(y_train[0])
    print(y_test[0])
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

    history = model.fit(X_train, y_train,
                        batch_size = batch_size,
                        epochs = epochs,
                        verbose = 1,
                        validation_data = (X_test, y_test))

    y_predicted = model.predict(X_test)
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    print('Accuracy:', np.mean( np.argmax(y_predicted, axis=1) ==  np.argmax(y_test, axis=1)))
    from sklearn.metrics import roc_curve, auc

    fprs, tprs = [None] * 10, [None] * 10
    aucs = [None] * 10

    for i in range(1):
        fprs[i], tprs[i], _ = roc_curve(y_test[:, i], y_predicted[:, i])
        aucs[i] = auc(fprs[i], tprs[i], reorder=True)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], '--', color='black')

    plt.title('One-vs-rest ROC curves', fontsize=16)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    class_names = ['rooms', 'bananas']
    for i in range(1):
        plt.plot(fprs[i], tprs[i], label='%s (AUC %.2lf)' % (class_names[i], aucs[i]))

    plt.legend(fontsize=14)
    plt.show()
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
            im = ax.imshow(X_test[k]/255)
    # plt.show()
    plt.savefig('test_pics_ban.png')
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_cifar_bana.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_cifar_bana.h5")
    print("Saved model to disk")

    test_on_image(model, 'real_world_test.png')


if __name__ == '__main__':
    main()
