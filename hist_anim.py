import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt
from matplotlib import animation
from tkinter import *
import time
import PIL.Image, PIL.ImageTk

def open():
    cap=cv2.VideoCapture("/home/pavel/cursovoy/cycle_img/video200ebanbIxSobak.mpg")
    while True:
        (ret,frame)=cap.read()
        cv2.imshow("video",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def get_image():
    cap=cv2.VideoCapture("/home/pavel/cursovoy/cycle_img/video200ebanbIxSobak.mpg")
    (ret,frame)=cap.read()
    height, width, no_channels = frame.shape
    canvas =Canvas(root,width=width,height=height)
    canvas.pack()
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
    canvas.create_image(0,0,image=photo,anchor=NW)
    root.mainloop()
def get_scale():
    x=scale1.get()
    
x=0

root=Tk()
button1=Button(root,text='start',command=open)
button2=Button(root,text='select Image',command=get_image)
scale1 = Scale(root,orient=HORIZONTAL,length=300,from_=0,to=500,
               resolution=1)
scale1.pack()               
button1.pack()
button2.pack()
root.mainloop()
