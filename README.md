# Drowsiness Detection

### This repo contains 3 apps:
- `DDD_Main.py`: This app takes live input from webcam, tracks the face and eyes and sends an audible feedback if the user's eyes has been closed for more than a second. 
- `streamlit_app.py`: Streamlit app. In this app, you upload a file and the program detects whether the eyes in the image is closed or not. Or if any eyes detected at all
 - `DDD_Desktop.py`: Tkinter app. The logic goes like the web app, you just use it in your desktop environment

> ####  The Convolutional Neural Network trained for this project can be found in `CNNClassifier2.ipynb` 

> #### Under the models folder, you can see the output of the notebook; which is our trained model that is a file with a `.h5` extension
