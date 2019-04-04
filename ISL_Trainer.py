import pickle

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils import print_summary
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def loadFromPickle():
    with open("features_numbers_letters", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels_numbers_letters", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels


def keras_model(image_x, image_y):
    num_of_classes = 33
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "ISL.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list


def main():
    features, labels = loadFromPickle()
    # loadFromPicklefeatures, labels = augmentData(features, labels)
    features = features / 127.5 - 1.
    features, labels = shuffle(features, labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
    train_y = np_utils.to_categorical(train_y)
    test_y = np_utils.to_categorical(test_y)
    # train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])


    # test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
    train_x = train_x.reshape(train_x.shape[0], 50, 50, 1)
    test_x = test_x.reshape(test_x.shape[0], 50, 50, 1)
    model, callbacks_list = keras_model(50,50)
    print_summary(model)
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=1, batch_size=64,
              callbacks=callbacks_list)
    model.save('ISL.h5')


main()
