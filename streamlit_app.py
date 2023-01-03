import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

roi_eye_left = [276, 285, 343, 346]

model = tf.keras.models.load_model('models/CNN-163216-80k.h5')

st.title("Uyku Hali Tespiti")

uploaded_file = st.file_uploader("Lütfen bir fotoğraf yükleyiniz", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Yüklenen fotoğraf', use_column_width=True)
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        results = face_mesh.process(image)

        annotated_image = image.copy()
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                landmarks = face.landmark

                e = {}
                for index in roi_eye_left:
                    x = int(landmarks[index].x * image.shape[1])
                    y = int(landmarks[index].y * image.shape[0])
                    e[index] = (x, y)

                cropped_eye_left = annotated_image[e[285][1]
                    :e[346][1], e[285][0]:e[346][0]]

                eye_roi = cv2.resize(cropped_eye_left, (256, 256))
                eye_roi = eye_roi / 255.0
                eye_roi = np.expand_dims(eye_roi, axis=0)
                prediction = model.predict(eye_roi)
                
                if prediction > 0.5:
                    status = "Uyku yok (Gözler Açık)"
                else:
                    status = "Uykulu (Gözler Kapalı)"
                st.write(f"Tahmin: {status}")

        else:
            st.write("Yüz tespit edilemedi. Yüz hatları belirgin bir fotoğraf yükleyiniz")
else:
    st.write("Lütfen bir fotoğraf yükleyiniz")


                
