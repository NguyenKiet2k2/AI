import tkinter as tk
import tensorflow as tf
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np

root = Tk()
root.title('Plant Disease Detection')
frame = tk.Frame(root, bg='#45aaf2')

lbl_pic_path = tk.Label(frame, text='Image Path:', padx=25, pady=25,
                        font=('verdana', 16), bg='#45aaf2')
lbl_result = tk.Label(frame, text='Đây là:', padx=25, pady=25,
                      font=('verdana', 16), bg='#45aaf2')
lbl_result1 = tk.Label(frame, text='', padx=25, pady=25,
                      font=('verdana', 16), bg='#45aaf2')
lbl_show_pic = tk.Label(frame, bg='#45aaf2')
entry_pic_path = tk.Entry(frame, font=('verdana', 16))
btn_browse = tk.Button(frame, text='CHOOSE', bg='grey', fg='#ffffff',
                       font=('verdana', 16))
detect = tk.Button(frame, text='Check', bg='blue', fg='#ffffff',
                   font=('verdana', 16))


def selectPic():
    global img
    resulf =['APPLE_Blak_Rot',
        'APPLE_Cedar_Rust',
        'APPLE_Scrab',
        'BLUEBERRY_Botrytis_Blight',
        'BLUEBERRY_Mummy_Berry',
        'BLUEBRRY_Silver_Leaf',
        'CORN_Common_Rust',
        'CORN_Ear_And_Stalk_Rot',
        'ORANGE_Citrus_Canker',
        'ORANGE_Citrus_Melanose',
        'PEPER_Curling_Leaves',
        'PEPER_Yellow_Leaves',
        'POTATO_Black_Scurf _Rhizoctonia_Canker',
        'POTATO_Fusarium_Dry_Rot',
        'POTATO_Late_Blight',
        'POTATO_Virus_Y',
        'TOMATO_Bacterial_Spot',
        'TOMATO_Early_Blight',
        'TOMATO_Mosaic_Virus_Disease',
        'TOMATO_Spider_Mite',
        'TOMATO_Yellow_Leaf_Curl_Virus'
    ]

    filename = filedialog.askopenfilename(
        initialdir="/images", title="Select Image",
        filetypes=(("jpg images", "*.jpg"), ("png images", "*.png"))
    )
    img = Image.open(filename)
    img = img.resize((200, 200), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    lbl_show_pic['image'] = img
    entry_pic_path.insert(0, filename)
    imgs = tf.keras.utils.load_img(filename, target_size=(30, 40))
    imgs = tf.keras.utils.img_to_array(imgs)
    imgs = np.expand_dims(imgs, axis=0)  # Add an extra dimension for batch
    imgs = imgs.astype('float32')
    imgs = imgs /255.0

    model = tf.keras.models.load_model('C:\\Users\\KIET\\Downloads\\Plant_Disease_Detection.h5')
    index = np.argmax(model.predict(imgs))
    lbl_result.configure(text='Đây là: '+ resulf[index])


btn_browse['command'] = selectPic

frame.pack()


lbl_show_pic.grid(row=1, column=0, columnspan="2")
lbl_result.grid(row=2, column=0)
lbl_result1.grid(row=2, column=1)
btn_browse.grid(row=3, column=0, columnspan="2", padx=10, pady=10)
root.mainloop()