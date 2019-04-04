import cv2
from keras.models import load_model
import numpy as np

model = load_model('ISL.h5')
data_dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '7': 7,
                   '8': 8, '9': 9, 'a': 10, 'b': 11,
                   'c': 12,
                   'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'k': 18,
                   'l': 19,
                   'm': 20, 'n': 21, 'p': 22, 'q': 23,
                   'r': 24,
                   's': 25, 't': 26, 'u': 27, 'w': 28, 'x': 29, 'y': 30,
                   'z': 31}


def main():
    cap = cv2.VideoCapture(0)
    pred_class = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask2 = cv2.inRange(hsv, np.array([110, 50, 50]), np.array([130, 255, 255]))
        res = cv2.bitwise_and(frame, frame, mask=mask2)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel_square = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(median, kernel_square, iterations=2)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)

        ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = thresh[y1:y1 + h1, x1:x1 + w1]
                newImage = cv2.resize(save_img, (50, 50))
                pred_probab, pred_class = keras_predict(model, newImage)
                print(pred_class, pred_probab)

        cv2.putText(frame,
                    "Conv Network :  " + list(data_dictionary.keys())[list(data_dictionary.values()).index(pred_class)],
                    (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break


def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 50
    image_y = 50
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


keras_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
main()
