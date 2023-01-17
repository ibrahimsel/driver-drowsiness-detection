# Drowsiness Detection

### To use the app: Clone the project. Open the terminal and `cd` into the project folder. Then run `python ./DDD_Main.py` or `python3 ./DDD_Main.py` depending on your OS and Python version
---
### This repo contains 3 apps:
- `DDD_Main.py`: This app takes live input from webcam, tracks the face and eyes and sends an audible feedback if the user's eyes has been closed for more than a second. 
- `streamlit_app.py`: Streamlit app. In this app, you upload a file and the program detects whether the eyes in the image is closed or not. Or if any eyes detected at all
 - `DDD_Desktop.py`: Tkinter app. The logic goes like the web app, you just use it in your desktop environment

> #### The Convolutional Neural Network trained for this project can be found in `CNNClassifier2.ipynb` 

### Folder Structure
- Under the models folder, you can see the output of the notebook; which is our trained model to classify if the eyes are closed or not
