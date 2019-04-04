import cv2
import numpy as np
import os
import dlib
from imutils import face_utils
import imutils

image_x, image_y = 50, 50

cap = cv2.VideoCapture(0)
fbag = cv2.createBackgroundSubtractorMOG2()
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(jawStart, jawEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["jaw"]
(noseStart, noseEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]
(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]


# Dat file is the crux of the code

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def main(g_id):
    total_pics = 600
    cap = cv2.VideoCapture(0)
    # x, y, w, h = 500, 100, 400, 350

    create_folder("gestures/" + str(g_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=700, height=700)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
            (xf, yf, wf, hf) = face_utils.rect_to_bb(subject)
            for (i, (x, y)) in enumerate(shape):
                # frame = cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
                leftEye = shape[leftEyeStart:leftEyeEnd]
                rightEye = shape[rightEyeStart:rightEyeEnd]
                mouth = shape[mouthStart:mouthEnd]
                nose = shape[noseStart:noseEnd]
                jaw = shape[jawStart:jawEnd]

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)
                noseHull = cv2.convexHull(nose)
                jawHull = cv2.convexHull(jaw)

                frame = cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), -1)
                frame = cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), -1)
                frame = cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), -1)
                frame = cv2.drawContours(frame, [noseHull], -1, (255, 0, 0), -1)
                # frame = cv2.drawContours(frame, [jawHull], -1, (255, 0, 0), -1)
                for l in range(1, len(jaw)):
                    ptA = tuple(jaw[l - 1])
                    ptB = tuple(jaw[l])
                    frame = cv2.line(frame, ptA, ptB, (255, 0, 0), 2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, np.array([110, 50, 50]), np.array([130, 255, 255]))
        res = cv2.bitwise_and(frame, frame, mask=mask2)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
        gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        blurred = cv2.blur(gradient, (2, 2))
        ret, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) > 0:
            c1 = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            if cv2.contourArea(c1) > 10000 and frames > 100:
                x1, y1, w1, h1 = cv2.boundingRect(c1)
                x = min(x1, xf)
                y = min(y1, yf)
                w = max(w1, wf)
                h = max(h1, hf)
                w_fin = (xf - x1) + wf
                h_fin = h
                pic_no += 1
                save_img = thresh[y:y + h, x1:x1 + (xf - x1) + wf]
                # if w1 > h1:
                #     save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                #                                   cv2.BORDER_CONSTANT, (0, 0, 0))
                # elif h1 > w1:
                #     save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                #                                   cv2.BORDER_CONSTANT, (0, 0, 0))
                save_img = cv2.copyMakeBorder(save_img, int((w_fin - h_fin) / 2), int((w_fin - h_fin) / 2), 0, 0,
                                              cv2.BORDER_CONSTANT, (0, 0, 0))
                save_img = cv2.resize(save_img, (image_x, image_y))
                cv2.putText(frame, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.imwrite("gestures/" + str(g_id) + "/" + str(pic_no) + ".jpg", save_img)

        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", frame)
        cv2.imshow("thresh", thresh)
        print(frames)
        keypress = cv2.waitKey(1)
        # if keypress == ord('c'):
        if frames <= 100:
            frames += 1
        if frames > 100:
            if flag_start_capturing == False:
                flag_start_capturing = True
            else:
                flag_start_capturing = False
                frames += 1
        if flag_start_capturing == True:
            frames += 1
        if pic_no == total_pics:
            break


g_id = input("Enter gesture number: ")
main(g_id)
