import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt
from matplotlib import animation
from tkinter import *
import time
import PIL.Image, PIL.ImageTk
import cur
x=0
x1=0
x2=0
y1=0
y2=0
h=0
w=0

def open():
   prog=cur.Core(True,x1,y1,w,h) 
   prog.run()
  
def get_image():
    
    root.mainloop()
def get_scale():
    x=scale1.get()
def move(event):
    x = event.x
    y = event.y
    s = "Движение мышью {}x{}".format(x, y)
    root.title(s)
def press(event):
    global x1
    global y1
    x1= event.x
    y1= event.y
    print('hi',h,'we',w)
    canvas.delete('rect')

def release(event):
    x2= event.x
    y2= event.y
    global h
    global w
    h=abs(y2-y1)
    w=abs(x2-x1)
    #cv2.rectangle(frame,(x1,y1),(h,w),(255,0,0),3)
    #height, width, no_channels = frame.shape
    #print(frame.shape)
    #photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
    #image_canvas=canvas.create_image(0,0,image=photo,anchor=NW) 
    canvas.create_rectangle(x1, y1, x2, y2,outline="blue", tag='rect',width=3)
   # print(x2,"*",y2)       
   # print(x1,"*",y1)     

def debug():
    print(h, w)

root=Tk()
button1=Button(root,text='start',command=open)
button2=Button(root,text='see',command=debug)
#button2=Button(root,text='select Image',command=get_image)
scale1 = Scale(root,orient=HORIZONTAL,length=300,from_=0,to=500,
               resolution=1)
cap=cv2.VideoCapture("/home/pavel/cursovoy/cycle_img/video.mpg")
(ret,frame)=cap.read()
height, width, no_channels = frame.shape
canvas =Canvas(root,width=width,height=height)
canvas.bind('<Button-1>',press)
canvas.bind('<ButtonRelease>',release)
photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
image_canvas=canvas.create_image(0,0,image=photo,anchor=NW)               
scale1.pack()
button2.pack()               
button1.pack()
canvas.pack()
#button2.pack()
root.mainloop()
