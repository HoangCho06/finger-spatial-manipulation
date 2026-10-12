import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

webcam_model = 0
cap = cv.VideoCapture(webcam_model)

cap.set(cv.CAP_PROP_FRAME_HEIGHT, 500)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 500)

face_model_path = 'finger_tracking/mediapipe_models/face_landmarker.task'
hand_model_path = 'finger_tracking/mediapipe_models/hand_landmarker.task'

BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode

FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions


face_options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = face_model_path),
    running_mode = VisionRunningMode.VIDEO,
    num_faces = 1
)

hand_options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = face_model_path),
    running_mode = VisionRunningMode.VIDEO,
    num_faces = 1
)

landmarker = FaceLandmarker.create_from_options(face_options)


def draw_face_landmarkers(video, result, height, width, colour_value):
    landmarks = result.face_landmarks
    if landmarks:
        for face_landmarks in landmarks:

            for landmark in face_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv.circle(video, (x,y), 1, colour_value, -1)


def main():
    landmarker_colour_value = (0,255,0)
    fps = cap.get(cv.CAP_PROP_FPS)

    bottom_left = (0,0)
    font = cv.FONT_HERSHEY_COMPLEX
    black = (0,0,0)

    timestamp = 0
    
    while True:
        ret, frame = cap.read()
    
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img
        )
    
        face_result = landmarker.detect_for_video(mp_image, timestamp)
        hand_result = landmarker.detect_for_video(mp_image, timestamp)
    
        height, width = frame.shape[:2]
        bottom_left = (0,height)
    
        draw_face_landmarkers(frame, face_result, height, width, landmarker_colour_value)
        cv.putText(frame,
                   str(fps),
                   bottom_left,
                   font,
                   fontScale=1,
                   color=black,
                   thickness=1,
                   )
    
        cv.imshow('spatial manipluation 0.1.1', frame)
    
        timestamp += int(1000/fps)
    
        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    landmarker.close()
    cv.destroyAllWindows()

main()
