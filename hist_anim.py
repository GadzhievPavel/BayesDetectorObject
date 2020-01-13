import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt
from matplotlib import animation
from tkinter import *
import time

def open():
    cap=cv2.VideoCapture("/home/pavel/cursovoy/cycle_img/video200ebanbIxSobak.mpg")
    while True:
        (ret,frame)=cap.read()
        cv2.imshow("video",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()    

root=Tk()
button1=Button(root,text='start',command=open)
bitImage=BitmapImage()
label=Label(root)
label.pack()
button1.pack()
root.mainloop()
