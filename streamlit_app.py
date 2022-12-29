import cv2
import time
import numpy as np
import mediapipe as mp
import streamlit as st
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# 285: top left point of left eye / 346: bottom right of left eye
roi_eye_left = [276, 285, 343, 346]

forehead = [10]

start_time = 0
drowsy_time = 0
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

model = tf.keras.models.load_model('models/CNN-163216-80k.h5')
drowsy_time = 0

st.title("Drowsiness Detection")
st.write("This is a simple drowsiness detection app using mediapipe and tensorflow.")

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(-1)
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while run:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            time.sleep(1)
            continue

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
                elif drowsy_time > 0 and drowsy_time < 2:
                    cv2.putText(image, "BLINK", (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

                cv2.putText(image, f"Drowsy Time: {drowsy_time:.2f}", (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            
                for index in roi_eye_left:
                    cv2.circle(image, e[index], 2, (255, 255, 255), -1)

        FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if ord('q') == cv2.waitKey(1):
            break

cap.release()
cv2.destroyAllWindows()