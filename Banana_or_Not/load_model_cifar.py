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
from keras.models import model_from_json
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def main():
    X_train, y_train, X_test, y_test, class_names = fetch_cifar_dataset()

    img = load_img('data/plane.png')  # this is a PIL image
    a = img_to_array(img, data_format='channels_last')  # this is a Numpy array with shape (3, 150, 150)
    a = a.reshape((1,) + a.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    img = load_img('data/dog.png')  # this is a PIL image
    b = img_to_array(img, data_format='channels_last')  # this is a Numpy array with shape (3, 150, 150)
    b = b.reshape((1,) + b.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    img = load_img('data/frog.png')  # this is a PIL image
    c = img_to_array(img, data_format='channels_last')  # this is a Numpy array with shape (3, 150, 150)
    c = c.reshape((1,) + c.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    x = []
    img = load_img('data/dog2.png')  # this is a PIL image
    d = img_to_array(img, data_format='channels_last')  # this is a Numpy array with shape (3, 150, 150)
    d = d.reshape((1,) + d.shape)  # this is a Numpy array with shape (1, 3, 150, 150)



    # temp_pic.append(temp_pics)
    temp_class = [0,5,6,5]
    # plt.imshow(a[0]/255)
    plt.show()
    cols = 10
    rows = 5
    batch_size = 500
    epochs = 1
    input_shape = (32, 32, 3)
    num_classes = 10

    # load json and create model
    json_file = open('model_cifar.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_cifar.h5")
    print("Loaded model from disk")

    loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer = keras.optimizers.Adadelta(),
                  metrics = ['accuracy'])

    # y_predicted = loaded_model.predict(X_test)
    # print('Accuracy:', np.mean( np.argmax(y_predicted, axis=1) ==  np.argmax(y_test, axis=1)))
    # from sklearn.metrics import roc_curve, auc
    #
    # fprs, tprs = [None] * 10, [None] * 10
    # aucs = [None] * 10
    #
    # for i in range(10):
    #     fprs[i], tprs[i], _ = roc_curve(y_test[:, i], y_predicted[:, i])
    #     aucs[i] = auc(fprs[i], tprs[i], reorder=True)
    #
    # plt.figure(figsize=(8, 8))
    # plt.plot([0, 1], [0, 1], '--', color='black')
    #
    # plt.title('One-vs-rest ROC curves', fontsize=16)
    #
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    #
    # for i in range(10):
    #     plt.plot(fprs[i], tprs[i], label='%s (AUC %.2lf)' % (class_names[i], aucs[i]))
    #
    # plt.legend(fontsize=14)
    # plt.savefig('ACU_ROC.png')
    # plt.show()
    #
    # y_predicted_classes = np.argmax(y_predicted, axis=1)
    # y_true_classes = np.argmax(y_test, axis=1)
    #
    #
    #
    # cols = 7
    # rows = 5
    # fig = plt.figure(figsize=(3 * cols - 1, 4 * rows - 1))
    #
    # for i in range(cols):
    #     for j in range(rows):
    #         k = np.random.randint(0, X_test.shape[0])
    #
    #         y_pred = y_predicted[k]
    #         predicted_class = np.argmax(y_pred)
    #         real_class = np.argmax(y_test[k])
    #         score = y_pred[predicted_class]
    #
    #         ax = fig.add_subplot(rows, cols, i * rows + j + 1)
    #         ax.grid('off')
    #         ax.axis('off')
    #         if real_class == predicted_class:
    #             ax.set_title('%s\nscore: %.3lf' % (class_names[real_class], score))
    #         else:
    #             ax.set_title('real: %s;\npredicted: %s\nwith score: %.3lf' % (
    #                 class_names[real_class], class_names[predicted_class], score
    #             ))
    #         im = ax.imshow(X_test[k])
    # plt.savefig('test_pics.png')
    # plt.show()


    for i,n in zip([a, b, c, d], [0,1,2,3]):
        fig2 = plt.figure()
        my_predicted = loaded_model.predict(i)
        my_pred = my_predicted[0]
        my_predicted_class = np.argmax(my_pred)
        my_real_class = temp_class[n]
        my_score = my_pred[my_predicted_class]
        ax1 = fig2.add_subplot(111)
        if my_real_class == my_predicted_class:
            ax1.set_title('%s\nscore: %.3lf' % (class_names[my_real_class], my_score))
        else:
            ax1.set_title('real: %s;\npredicted: %s\nwith score: %.3lf' % (
                class_names[my_real_class], class_names[my_predicted_class], my_score
            ))
        im1 = ax1.imshow(i[0]/255)
        outName = 'save_'+str(n)+'.png'
        plt.savefig(outName)
        plt.show()


    pass

if __name__ == '__main__':
    main()
