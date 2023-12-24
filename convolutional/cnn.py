import itertools
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import datasets, layers, models
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.python.ops.confusion_matrix import confusion_matrix

warnings.filterwarnings('ignore')

matplotlib.use('QtAgg')

print("Tensorflow version:", tf.__version__)
print("Keras version:", keras.__version__)
from keras.datasets import cifar10

# pip install opencv-python

plt.rcParams['figure.figsize'] = [10, 7]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# cnn avoid parameter explosion in deep learning while using fully connected dense layer networks
# deep networks also don't provide a way to make feature recognition within the image independent of position
# cnns are used mainly in image recognition
# convolutional layers probe the image locally
# pooling layers downsample the image
#
# sliding window applied to an image is often called a kernel function or a filter

# def brightness_adjustment(img):
#     # turn the image into the HSV space
#     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#     # creates a random bright
#     ratio = .5 + np.random.uniform()
#     # convert to int32, so you don't get uint8 overflow
#     # multiply the HSV Value channel by the ratio
#     # clips the result between 0 and 255
#     # convert again to uint8
#     hsv[:,:,2] =  np.clip(hsv[:,:,2].astype(np.int32) * ratio, 0, 255).astype(np.uint8)
#     # return the image int the BGR color space
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test1():
    # https://www.kaggle.com/code/ektasharma/simple-cifar10-cnn-keras-code-with-88-accuracy
    num_classes=10

    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Converting the pixels data to float type
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # Standardizing (255 is the total number of pixels an image can have)
    train_images = train_images / 255
    test_images = test_images / 255

    # One hot encoding the target class (labels)
    num_classes = 10
    train_labels = np_utils.to_categorical(train_labels, num_classes)
    test_labels = np_utils.to_categorical(test_labels, num_classes)

    # Creating a sequential model and adding layers to it

    model = Sequential()

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))  # num_classes = 10

    # Checking the model summary
    model.summary()

    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    history = model.fit(train_images, train_labels, batch_size=64, epochs=1,
                        validation_data=(test_images, test_labels))

    #(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    # X_train = X_train / 255.0
    # X_test = X_test / 255.0
    # # One-Hot-Encoding
    # Y_train_en = to_categorical(Y_train, 10)
    # Y_test_en = to_categorical(Y_test, 10)
    # #fig = plt.figure(figsize=(10, 10))
    # #plt.imshow(X_train[2])
    # #plt.show()
    #
    # # img_generator = ImageDataGenerator(preprocessing_function=brightness_adjustment,
    # #                                    rescale=1./255,
    # #                                    rotation_range=2, width_shift_range=0.01,
    # #                                    height_shift_range=0.01, shear_range=0.02,
    # #                                    zoom_range=0.03, channel_shift_range=4.,
    # #                                    horizontal_flip=True, vertical_flip=True,
    # #                                    fill_mode='nearest')
    # #
    # # train_generator = img_generator.flow_from_directory(
    # #     directory=r"./train/",
    # #     target_size=(32, 32),
    # #     color_mode="rgb",
    # #     batch_size=32,
    # #     # class_mode="categorical",
    # #     shuffle=True,
    # #     seed=42
    # # )
    #
    # y_train = tf.keras.utils.to_categorical(Y_train, num_classes)
    # y_test = tf.keras.utils.to_categorical(Y_test, num_classes)
    #
    # img_generator = ImageDataGenerator(rescale=1./255)
    # test_img_generator = ImageDataGenerator(rescale=1./255)
    #
    # train_generator = img_generator.flow(
    #     x=X_train,
    #     y=y_train,
    #     batch_size=32,
    #     shuffle=True,
    #     seed=42
    # )
    #
    # test_generator = test_img_generator.flow(
    #     x=X_test,
    #     y=y_test,
    #     batch_size=32,
    #     shuffle=True,
    #     seed=42
    # )
    #
    # model = Sequential()
    # model.add(Conv2D(16, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'))
    # #model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'))
    # #model.add(MaxPooling2D(pool_size=(2, 2)))
    # #model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'))
    # #model.add(Conv2D(128, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))
    # model.add(Flatten())
    # #model.add(Dense(512, activation='relu'))
    # #model.add(Dropout(0.1))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(10, activation='softmax'))
    # # sparese_categorical_crossentropy should be used if data is not 1-hot encoded, but labeled with numerical labels
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    # # train_data_gen can also be used as input to our neural network and as validation data param, steps_per_epoch=len(train_images), validation_steps=len(test_images)
    # history = model.fit( train_generator,
    #     #X_train, y_train,
    #                     epochs=1, verbose=1, validation_data=test_generator, # (X_test, y_test),
    #                     #steps_per_epoch=len(Y_train) // batch_size,
    #                     #validation_steps=len(Y_test) // batch_size,
    #                     )
    # print(history.params.keys())
    # print(model.evaluate())
    # p_test = model.predict(X_test).argmax(axis=1)
    # cm = confusion_matrix(Y_test_en, p_test)
    print(history.params.keys())
    print(model.evaluate(test_images, test_labels, batch_size=64))

    plt.figure(figsize=[6, 4])
    plt.plot(history.history['loss'], 'black', linewidth=2.0)
    plt.plot(history.history['val_loss'], 'green', linewidth=2.0)
    plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.title('Loss Curves', fontsize=12)

    # Accuracy curve
    plt.figure(figsize=[6, 4])
    plt.plot(history.history['accuracy'], 'black', linewidth=2.0)
    plt.plot(history.history['val_accuracy'], 'blue', linewidth=2.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.title('Accuracy Curves', fontsize=12)
    plt.show()

    p_test = model.predict(test_images).argmax(axis=1)
    print(p_test)
    # cm = confusion_matrix(test_labels, p_test)
    # print(cm)
    # plot_confusion_matrix(cm, list(range(10)))
    ...


if __name__ == '__main__':
    test1()
