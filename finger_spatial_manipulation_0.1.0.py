import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

webcam_model = 0
cap = cv.VideoCapture(webcam_model)

cap.set(cv.CAP_PROP_FRAME_HEIGHT, 500)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 500)

base_options = mp.tasks.BaseOptions
face_detector = mp.tasks.vision.FaceDetector
face_detector_options = mp.tasks.vision.FaceDetectorOptions
vision_running_mode = mp.tasks.vision.RunningMode

while True:
    ret, frame = cap.read()

    img = frame.copy()

    cv.imshow('spatial manipluation 0.1.0', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()