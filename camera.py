import cv2
import time
import mediapipe as mp
import numpy as np
import tensorflow as tf

mp_face_mesh = mp.solutions.face_mesh
forehead = [10]
#                    top left   bottom right
roi_eye_left = [276, 285, 343, 346]
roi_eye_right = [46, 55, 188, 111]

model = tf.keras.models.load_model('models/CNN-163216-80k.h5')

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.drowsy_time = 0
        self.start_time = 0
        self.end_time = 0

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        ret, frame = self.video.read()

        fps = self.video.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, "FPS: {}".format(fps), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:

            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    landmarks = face.landmark

                    f = {}
                    for index in forehead:
                        x = int(landmarks[index].x * frame.shape[1])
                        y = int(landmarks[index].y * frame.shape[0])
                        f[index] = (x, y)

                    e = {}
                    for index in roi_eye_left:
                        x = int(landmarks[index].x * frame.shape[1])
                        y = int(landmarks[index].y * frame.shape[0])
                        e[index] = (x, y)

                    for index in roi_eye_right:
                        x = int(landmarks[index].x * frame.shape[1])
                        y = int(landmarks[index].y * frame.shape[0])
                        e[index] = (x, y)
                    
                    # 285 top left 346 bottom right
                    cropped_left_eye = frame[e[285][1]
                        :e[346][1], e[285][0]:e[346][0]]

                    if cropped_left_eye.size == 0:
                        continue
                    eye_roi = cv2.resize(cropped_left_eye, (256, 256))
                    eye_roi = eye_roi / 255.0
                    eye_roi = np.expand_dims(eye_roi, axis=0)
                    prediction = model.predict(eye_roi)

                    if prediction > 0.5:
                        cv2.putText(frame, f"Eyes Open {prediction[0][0]:.2f}", f[10],
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        self.start_time = time.perf_counter()
                        self.drowsy_time = 0
                    else:
                        cv2.putText(frame, f"Eyes Closed {prediction[0][0]:.2f}", f[10],
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.end_time = time.perf_counter()
                        self.drowsy_time += self.end_time - self.start_time
                        self.start_time = self.end_time

                    if self.drowsy_time > 2:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
                        # playsound(alarm_sound)
                        
                    elif self.drowsy_time > 0 and self.drowsy_time < 2:
                        cv2.putText(frame, "BLINK", (10, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

                    cv2.putText(frame, f"Drowsy Time: {self.drowsy_time:.2f}", (10, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    for index in roi_eye_left:
                        cv2.circle(frame, e[index], 2, (255, 255, 255), -1)

                    for index in roi_eye_right:
                        cv2.circle(frame, e[index], 2, (255, 255, 255), -1)

        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()