
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import tkinter as tk
from tkinter import filedialog
from tkinter import *

from sklearn import metrics

from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# donwload haarcascade_frontalface_default from here "https://github.com/opencv/opencv/tree/master/data/haarcascades"


def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

top =tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a1.json","model_weights1.h5")


EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image,1.3,5)
    try:
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            roi = cv2.resize(fc,(48,48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
        print("Predicted Emotion is" + pred)
        label1.configure(foreground="#011638",text = pred)
    except:
        label1.configure(foreground="#011638",text = "Unable to detect")


def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Emotion", command= lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except:
        pass

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top,text='Emotion Detector',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground="#364156")
heading.pack()
top.mainloop()

# Function to preprocess the input image
def preprocess_image(image_path, target_size=(48, 48)):
    image = load_img(image_path, color_mode='grayscale', target_size=target_size)
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to visualize activation maps
def visualize_activation_maps(model, image_path, layer_names, target_size=(48, 48)):
    image_array = preprocess_image(image_path, target_size)
    outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = Model(inputs=model.input, outputs=outputs)
    activations = activation_model.predict(image_array)

    for layer_name, activation in zip(layer_names, activations):
        num_filters = activation.shape[-1]
        size = activation.shape[1]

        activation_min = np.min(activation)
        activation_max = np.max(activation)
        activation = (activation - activation_min) / (activation_max - activation_min)

        fig, axes = plt.subplots(1, num_filters, figsize=(15, 15))
        fig.suptitle(f'Activations from layer: {layer_name}', fontsize=16)

        for i in range(num_filters):
            ax = axes[i] if num_filters > 1 else axes
            ax.imshow(activation[0, :, :, i], cmap='viridis')
            ax.axis('off')
        plt.show()

# Example Usage (Replace layer names as per your model)
# visualize_activation_maps(model, "path_to_image.jpg", ["conv2d", "conv2d_1"])
