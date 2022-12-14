import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
from playsound import playsound

mp_face_mesh = mp.solutions.face_mesh

#                    top left   bottom right
roi_eye_left = [276, 285, 343, 346]
roi_eye_right = [46, 55, 188, 111]

forehead = [10]

start_time = 0
drowsy_time = 0

model = tf.keras.models.load_model('models/CNN-163216-80k.h5')
alarm_sound = 'alarm_sound.mp3'

cv2.namedWindow('Driver Drowsiness Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Driver Drowsiness Detection', 640, 480)

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        cv2.putText(image, "Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                landmarks = face.landmark

                f = {}
                for index in forehead:
                    x = int(landmarks[index].x * image.shape[1])
                    y = int(landmarks[index].y * image.shape[0])
                    f[index] = (x, y)

                e = {}
                for index in roi_eye_left:
                    x = int(landmarks[index].x * image.shape[1])
                    y = int(landmarks[index].y * image.shape[0])
                    e[index] = (x, y)

                for index in roi_eye_right:
                    x = int(landmarks[index].x * image.shape[1])
                    y = int(landmarks[index].y * image.shape[0])
                    e[index] = (x, y)
                

                    """
                    -------------------------------------------
                    |                                         | 
                    |    (x1, y1)                             |
                    |      ------------------------           |
                    |      |                      |           |
                    |      |                      |           | 
                    |      |         ROI          |           |  
                    |      |                      |           |   
                    |      |                      |           |   
                    |      |                      |           |       
                    |      ------------------------           |   
                    |                           (x2, y2)      |    
                    |                                         |             
                    |                                         |             
                    |                                         |             
                    -------------------------------------------

                    ROI = image[y1:y2, x1:x2]
                    """

                # 285 top left 346 bottom right
                cropped_left_eye = image[e[285][1]
                    :e[346][1], e[285][0]:e[346][0]]

                if cropped_left_eye.size == 0:
                    continue
                eye_roi = cv2.resize(cropped_left_eye, (256, 256))
                eye_roi = eye_roi / 255.0
                eye_roi = np.expand_dims(eye_roi, axis=0)
                prediction = model.predict(eye_roi)

                if prediction > 0.5:
                    cv2.putText(image, f"Eyes Open {prediction[0][0]:.2f}", f[10],
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    start_time = time.perf_counter()
                    drowsy_time = 0
                else:
                    cv2.putText(image, f"Eyes Closed {prediction[0][0]:.2f}", f[10],
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    end_time = time.perf_counter()
                    drowsy_time += end_time - start_time
                    start_time = end_time

                if drowsy_time > 2:
                    cv2.putText(image, "DROWSINESS ALERT!", (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
                    # playsound(alarm_sound)
                    
                elif drowsy_time > 0 and drowsy_time < 2:
                    cv2.putText(image, "BLINK", (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

                cv2.putText(image, f"Drowsy Time: {drowsy_time:.2f}", (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                for index in roi_eye_left:
                    cv2.circle(image, e[index], 2, (255, 255, 255), -1)

                for index in roi_eye_right:
                    cv2.circle(image, e[index], 2, (255, 255, 255), -1)

        cv2.imshow('Driver Drowsiness Detection', image)
        if ord('q') == cv2.waitKey(1):
            break

cap.release()
cv2.destroyAllWindows()
