import tkinter
import tkinter.filedialog
import tkinter.messagebox
import tkinter.ttk
import tensorflow as tf
import mediapipe as mp
import cv2
import PIL
import numpy as np


class DesktopApp:
    def __init__(self):
        self.file_path = ''
        self.main_window = tkinter.Tk()
        self.main_window.title('Driver Drowsiness Detection')
        self.main_window.geometry('1000x700')
        self.main_window.resizable(width=True, height=True)
        self.create_button('Fotoğraf Yükle', width=11, height=1, bg='gray', fg='white', font=(
            'Times New Roman', 14), row=100, column=100, sticky='w')

        self.main_window.configure(background='white')
        self.main_window.mainloop()

    def create_button(self, text, command=None, width=10, height=2, bg='white', fg='black', font=('Arial', 10), row=0, column=0, sticky='w'):
        button = tkinter.Button(self.main_window, text=text, command=command,
                                width=width, height=height, bg=bg, fg=fg, font=font)
        button.grid(row=row, column=column, sticky=sticky)
        button.bind('<Button-1>', self.choose_file)
        button.lift()

    def choose_file(self, event):
        self.file_path = tkinter.filedialog.askopenfilename(
            filetypes=[('Image Files', '.png .jpg .jpeg')])
        if self.file_path == '':
            tkinter.messagebox.showerror(
                'Hata', 'Lütfen bir dosya seçiniz.')
        else:
            tkinter.messagebox.showinfo(
                'Bilgi', 'Dosya seçildi. Tahmin yapılıyor...')

            if self.file_path.endswith('.jpg') or self.file_path.endswith('.jpeg'):
                self.convert_to_png(self.file_path)
                self.file_path = 'DDD_Image.png'

        for widget in self.main_window.winfo_children():
            if widget.winfo_class() == 'Label':
                widget.destroy()
        self.display_image(self.file_path)
        self.predict()

    def display_image(self, image_path):
        image = tkinter.PhotoImage(file=image_path)
        image = image.subsample(2, 2)
        label = tkinter.Label(self.main_window, image=image)
        label.place(relx=0.5, rely=0.5, anchor='center')
        label.image = image

    def create_label(self, text, width=10, height=2, bg='white', fg='black', font=('Arial', 10), row=0, column=0, sticky='w'):
        label = tkinter.Label(self.main_window, text=text,
                              width=width, height=height, bg=bg, fg=fg, font=font)
        label.grid(row=row, column=column, sticky=sticky)

    def create_entry(self, width=10, bg='white', fg='black', font=('Arial', 10), row=0, column=0, sticky='w'):
        entry = tkinter.Entry(
            self.main_window, width=width, bg=bg, fg=fg, font=font)
        entry.grid(row=row, column=column, sticky=sticky)

    def predict(self):
        mp_face_mesh = mp.solutions.face_mesh
        model = tf.keras.models.load_model('models/CNN-163216-80k.h5')

        #                    top left   bottom right
        roi_eye_left = [276, 285, 343, 346]
        roi_eye_right = [46, 55, 188, 111]
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:

            image = cv2.imread(self.file_path)
            results = face_mesh.process(image)

            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    landmarks = face.landmark
                    f = {}
                    for index in roi_eye_right:
                        x = int(landmarks[index].x * image.shape[1])
                        y = int(landmarks[index].y * image.shape[0])
                        f[index] = (x, y)

                    e = {}
                    for index in roi_eye_left:
                        x = int(landmarks[index].x * image.shape[1])
                        y = int(landmarks[index].y * image.shape[0])
                        e[index] = (x, y)

                    eye_left = image[e[285][1]:e[346][1], e[285][0]:e[346][0]]
                    # eye_right = [e[46], e[55], e[188], e[111]]

                    left_eye_roi = cv2.resize(eye_left, (256, 256))
                    left_eye_roi = left_eye_roi / 255.0
                    left_eye_roi = np.expand_dims(left_eye_roi, axis=0)

            else:
                tkinter.messagebox.showerror(
                    'Hata', 'Fotoğrafta yüz algılanamadı. Yüz hatları belirgin bir fotoğraf seçiniz')
                return

            try:
                prediction = model.predict(left_eye_roi)
            except:
                tkinter.messagebox.showerror(
                    'Hata', 'Bilinmeyen bir hata oluştu')
                return

            if prediction[0][0] > 0.5:
                tkinter.messagebox.showinfo(
                    'Sonuç', 'Gözler açık.')
            else:
                tkinter.messagebox.showinfo(
                    'Sonuç', 'Gözler kapalı')
            return

    def convert_to_png(self, image_path):
        image = PIL.Image.open(image_path)
        image.save('DDD_Image.png')


DesktopApp()
