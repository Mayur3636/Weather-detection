from tkinter import *
import os
from PIL import Image
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
from pathlib import Path
import keras.utils as image
# as image_utils
import PIL.ImageOps
import random
import tensorflow as tf
# from keras.utils import np_utils
from keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class Weather(Tk):
    def __init__(self):
        super().__init__()
        self.title("Welcome")
        self.geometry("900x500+20+20")
        self.name = ""
        self.mainframe = Frame(self)
        self.mainframe.place(x=0, y=0, height=500, width=900)
        Label(self.mainframe, text="Weather Detection", bg="#0008C1", fg="#FDF0E0",
              font=("Impact", 35, "bold")).place(x=60, y=20)
        Button(self.mainframe, text="Select Photo", font=("bold", 15), bg="white", command=self.openFile).place(x=100,
                                                                                                                y=100)
        self.l1 = Label(self.mainframe, text=f"",font=("Arial", 15,), fg="#0008C1")
        self.l1.place(x=520, y=240) 
        self.mainloop()

    def openFile(self):
        f = filedialog.askopenfilename(title="Select a File", filetypes=(("Image files", "*.jpg*"), ("all files",
                                                                                                     "*.*")))
        # Label(self.mainframe, text=f"{f}").place(x=60, y=150)
        def resize_image(event):
            new_width = 300
            new_height = 300
            image = copy_of_image.resize((new_width, new_height))
            photo = ImageTk.PhotoImage(image)
            label.config(image=photo)
            label.image = photo  # avoid garbage collection
                
        image = Image.open(f)
        copy_of_image = image.copy()
        photo = ImageTk.PhotoImage(image)
        label = Label(self.mainframe, image=photo)
        label.bind('<Configure>', resize_image)
        label.place(x=60, y=180)

        self.result(f)

    def result(self, image_path):

        img = Image.open(image_path)

        # Calculating edges in the image
        if img.size[0] >= img.size[1]:
            read_image = cv2.imread(image_path, 50)
            edges = cv2.Canny(read_image, 150, 300)
            shape = np.shape(edges)
            left = np.sum(edges[0:shape[0] // 2, 0:shape[1] // 2])
            right = np.sum(edges[0:shape[0] // 2, shape[1] // 2:])

            # More edges = Building
            # Less edges = Sky
            if right > left:
                sky_side = 0
            else:
                sky_side = 1

            # Resizing image to a particular size
            base_height = 400
            wpercent = (base_height / float(img.size[1]))
            wsize = int((float(img.size[0]) * float(wpercent)))
            img = img.resize((wsize, base_height), Image.ANTIALIAS)

            # Cropping sky area from the image
            if sky_side == 0:
                img = img.crop((0, 0, base_height, img.size[1]))
            else:
                img = img.crop((img.size[0] - base_height, 0, img.size[0], img.size[1]))

            # Saving the cropped image
            # destination = "D:/5th sem/AI/Project/Weather-Detection-Using-Images-master/Weather-Detection-Using-Images" \
            #               "-master"
            img.save("im.jpg")

        else:
            base_width = 400
            wpercent = (base_width / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))

            # Resizing the image
            img = img.resize((base_width, hsize), Image.ANTIALIAS)

            # Cropping the image
            img = img.crop((0, 0, img.size[0], 400))

            # Saving the cropped image
            # destination = "D:/5th sem/AI/Project/Weather-Detection-Using-Images-master/Weather-Detection-Using-Images" \
            #               "-master"
            img.save("im.jpg")

        # Converting images to matrix
        
        img = image.load_img("im.jpg", target_size = (100,100))
        img = PIL.ImageOps.invert(img)
        img = image.img_to_array(img)
        train_data = [img]
        # dest = "D:/5th sem/AI/Project/Weather-Detection-Using-Images-master/Weather-Detection-Using-Images-master/"
        np.save("img_data.npy", np.array(train_data))

        # Data is ready for testing

        model = load_model("trainedModelE10.h5")
        test_data = np.load("img_data.npy")
        y = (model.predict(test_data) > 0.5).astype("int32")
        count = 0
        m = 0
        for i in y:
            for ind in range(len(i)):
                if i[ind] == 1:
                    m = ind
            # count += 1
            print(m)
        weather_list = ["Cloudy", "Sunny", "Rainy", "Snowy", "Foggy"]

        self.l1.config(text=f"{weather_list[m]}") 


Weather()
