import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

webcam_model = 0
cap = cv.VideoCapture(webcam_model)
fps = cap.get(cv.CAP_PROP_FPS)
delay = int(1000/fps)
timestamp = 0

cap.set(cv.CAP_PROP_FRAME_HEIGHT, 500)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 500)

model_path = 'finger_tracking/mediapipe_models/face_landmarker.task'

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = model_path),
    running_mode = VisionRunningMode.VIDEO,
    num_faces = 1
)

landmarker = FaceLandmarker.create_from_options(options)


def draw_landmarkers(video, result, height, width, colour_value):
    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:

            for landmark in face_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv.circle(video, (x,y), 1, colour_value, -1)

def main():
    landmarker_colour_value = (0,255,0)
    
    while True:
        ret, frame = cap.read()
    
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img
        )
    
        result = landmarker.detect_for_video(mp_image, timestamp)
    
        height, width = frame.shape[:2]
    
        draw_landmarkers(frame, result, height, width, landmarker_colour_value)
    
        cv.imshow('spatial manipluation 0.1.1', frame)
    
        timestamp += int(1000/fps)
    
        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    landmarker.close()
    cv.destroyAllWindows()

main()
