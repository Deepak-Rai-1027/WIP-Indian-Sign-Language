from __future__ import division
import cv2
import os
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
from itertools import islice

LIMIT = None

data_dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '7': 7,
                   '8': 8, '9': 9, 'a': 10, 'b': 11,
                   'c': 12,
                   'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'k': 18,
                   'l': 19,
                   'm': 20, 'n': 21, 'p': 22, 'q': 23,
                   'r': 24,
                   's': 25, 't': 26, 'u': 27, 'w': 28, 'x': 29, 'y': 30,
                   'z': 31}


GESTURE_FOLDER = 'gestures/'


def return_data():
    X = []
    y = []
    features = []

    for filename in os.listdir(os.path.join(GESTURE_FOLDER)):
        for file in os.listdir(os.path.join(GESTURE_FOLDER, filename)):
            if file.endswith('.jpg'):
                full_path = os.path.join(GESTURE_FOLDER, filename, file)
                X.append(full_path)
                y.append(data_dictionary[filename])

    for i in range(len(X)):
        img = plt.imread(X[i])
        features.append(img)

    features = np.array(features).astype('float32')
    labels = np.array(y).astype('float32')

    with open("features_numbers_letters", "wb") as f:
        pickle.dump(features, f, protocol=4)
    with open("labels_numbers_letters", "wb") as f:
        pickle.dump(labels, f, protocol=4)


return_data()
