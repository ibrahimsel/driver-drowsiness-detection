# Drowsiness Detection

### To use the app: Clone the project. Open the terminal and `cd` into the project folder. Then run `python ./DDD_Main.py` or `python3 ./DDD_Main.py` depending on your OS and Python version
---
### This repo contains 2 apps:
- `DDD_Main.py`: OpenCV app. This app takes live input from webcam, tracks the face and eyes and sends an audible feedback if the user's eyes has been closed for more than a second. 
 - `DDD_Desktop.py`: Tkinter app. You upload an image to this app and if there is a face and a pair of eyes in the image, it tells you if the eyes are closed or not

> #### The Convolutional Neural Network trained for this project can be found in `CNNClassifier2.ipynb` 

### Folder Structure
- Under the models folder, you can see the output of the notebook; which is our trained model to classify if the eyes are closed or not
