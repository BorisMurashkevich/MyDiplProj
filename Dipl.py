import sys
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import uuid
import random
from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import idx2numpy
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import RMSprop
import time
import datetime
from tkinter import Tk,Menu,Button,PhotoImage,Radiobutton,Label,IntVar,Frame,LabelFrame,Entry,W,Text,scrolledtext,END
from tkinter import Image as ImageTk
from tkinter import filedialog as fd
global notclicked
notclicked = True

def clicked():
    modelName = fd.askopenfilename()
    model = tf.keras.models.load_model(modelName)
    return modelName,model
def quit():
    window.destroy()
def openimg():
    global opnimg
    opnimg = fd.askopenfilename()
    opnimg1 = Image.open(opnimg)
    opnimg1 = opnimg1.resize((440,116))
    opnimg1.save('Temp\opened.png')
    window.photo1 = PhotoImage(file = 'Temp\opened.png')
    vlabel.configure(image=window.photo1)
    print ("updated")
    return opnimg
def trainbtn():
    model = mnist_make_model(image_w=28, image_h=28)
    mnist_train(model)
def recogn():
    lettersExtract()
    final(model)




letters = []
stroka = []
finallet = ''
let = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
for i in range(0,1000):
    stroka.append('D:\Dipl\Dipl\Temp\\'+str(uuid.uuid4())+'.png')
def lettersExtract():
    global openimg
    file_name = opnimg
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
    img1 = Image.open(file_name)
    width, height = img1.size
    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    output = img.copy()
  
    for idx, contour in enumerate(contours):
       
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            cropped = img1.crop((x-9, y-15, width-(width-(x+w+9)), height-(height-(y+h+15))))
            letters.append((x,w,cropped))
            letters.sort(key=lambda x: x[0], reverse=False)
            for im in range(0,len(letters)):
                letters[im][2].save(stroka[im])
    return letters
            









def mnist_make_model(image_w, image_h):
   global Sel,Selspis
   # Neural network model
   model = Sequential()
   model.add(Dense(784, activation=Selspis[Sel], input_shape=(image_w*image_h,)))
   model.add(Dense(62, activation=Selspis[Sel]))
   model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
   return model
def mnist_train(model):
   test_images = idx2numpy.convert_from_file('D:\\Emnist\\emnist-byclass-test-images-idx3-ubyte')
   test_labels = idx2numpy.convert_from_file('D:\\Emnist\\emnist-byclass-test-labels-idx1-ubyte')
   train_images = idx2numpy.convert_from_file('D:\\Emnist\\emnist-byclass-train-images-idx3-ubyte')
   train_labels = idx2numpy.convert_from_file('D:\\Emnist\\emnist-byclass-train-labels-idx1-ubyte')
   test_images_copy = np.copy(test_images)
   test_labels_copy = np.copy(test_labels)
   train_images_copy = np.copy(train_images)
   train_labels_copy = np.copy(train_labels)
   del(test_images,test_labels,train_images,train_labels)
   for test_im in range(len(test_images_copy)):
       test_images_copy[test_im] = np.rot90(test_images_copy[test_im],3)
       test_images_copy[test_im] = np.fliplr(test_images_copy[test_im])
   for test_im2 in range(len(train_images_copy)):
       train_images_copy[test_im2] = np.rot90(train_images_copy[test_im2],3)
       train_images_copy[test_im2] = np.fliplr(train_images_copy[test_im2])

   image_size = train_images_copy.shape[1]
   train_data = train_images_copy.reshape(train_images_copy.shape[0], image_size*image_size)
   test_data = test_images_copy.reshape(test_images_copy.shape[0], image_size*image_size)
   train_data = train_data.astype('float32')
   test_data = test_data.astype('float32')
   train_data /= 255.0
   test_data /= 255.0

   num_classes = 62
   train_labels_cat = keras.utils.to_categorical(train_labels_copy, num_classes)
   test_labels_cat = keras.utils.to_categorical(test_labels_copy, num_classes)
   print("Training the network...")
   t_start = time.time()
   model.fit(train_data, train_labels_cat, epochs=30, batch_size=64, verbose=1, validation_data=(test_data, test_labels_cat))
   model.save('letters_28x28.h5')
   return model



def letters_predict(model, image_file):
   global finallet
   image_size = 28
   img = keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size), color_mode='grayscale')
   img_arr = np.expand_dims(img, axis=0)
   img_arr = 1 - img_arr/255.0
   img_arr = img_arr.reshape((1, image_size*image_size))
   result = model.predict_classes([img_arr])
   finallet = finallet + let[result[0]]
   #print(finallet)
   return finallet


def final(model):
    global finallet
    global text_area
    for k in range(0,len(letters)):
        dn = letters[k+1][0] - letters[k][0] - letters[k][1] if k < len(letters) - 1 else 0
        letters_predict(model, stroka[k])
        if (dn > letters[k][1]/4):
            finallet += ' '
    for im1 in range(0,len(letters)):
       file_path = stroka[im1]
       os.remove(file_path)
    print(finallet)
    file = open("test.txt", "w")
    file.write(finallet)
    file.close()
    text_area.delete('1.0',END)
    text_area.insert('1.0',finallet)
    return finallet


window = Tk()  
window.title("Dipl")  
window.geometry('640x350')  
menu = Menu(window)
new_item = Menu(menu, tearoff=0)
new_item.add_command(label='Открыть', command=clicked) 
new_item.add_command(label='Закрыть программу', command=quit)
menu.add_cascade(label='Файл', menu=new_item)  
ph = PhotoImage(file = "open.png")
ph1 = PhotoImage(file = "train.png")
photo = Image.open('not loaded.png')
photo = photo.resize((430,116))
photo.save('Temp\q.png')
window.photo = PhotoImage(file = 'Temp\q.png')
vlabel=Label(window,image=window.photo)
vlabel.place(x=200,y=20)

selected = IntVar() 
label_frame = LabelFrame(window, text='Функции активации')
label_frame.place(x=20, y=100)
Radiobutton(label_frame,text='Сигмоидная', value=1,variable=selected).grid(row=0, column=0,sticky=W)
Radiobutton(label_frame,text='Линейная', value=2,variable=selected).grid(row=1, column=0,sticky=W)
Radiobutton(label_frame,text='ReLu', value=3,variable=selected).grid(row=2, column=0,sticky=W)
Radiobutton(label_frame,text='Гиперболический тангенс', value=4,variable=selected).grid(row=3, column=0,sticky=W)
Radiobutton(label_frame,text='softmax', value=5,variable=selected).grid(row=4, column=0,sticky=W)
Sel = selected.get()
Selspis = ['sigmoid','linear','relu','tahn','softmax']
btn = Button(height = 50, width = 50, image = ph, command=openimg)
btn1 = Button(height = 50, width = 50, image = ph1, command=trainbtn)
btn2 = Button(height = 3, width = 40, text="Распознать!", command=recogn)
btn.place(x=20,y=20)
btn1.place(x=90,y=20)
btn2.place(x=270,y=146)
text_area = scrolledtext.ScrolledText(window,width = 45, height = 5)
text_area.place(x=230, y=210)
text_area.insert('1.0','')
modelName = 'letters_28x28.h5'
model = tf.keras.models.load_model(modelName)
window.config(menu=menu)  
window.mainloop()

