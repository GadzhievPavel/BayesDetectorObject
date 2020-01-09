import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt
from matplotlib import animation

def segmentator(frame,hist):
    temp_frame=frame.copy()
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            if(hist[j,i]<0):
                temp_frame[np.where(frame == i)]=255
                frame=temp_frame    
    frame[np.where(frame!=255)]=0        
    return frame   

parser = argparse.ArgumentParser()
parser.add_argument('-x',"--coordX",type=int, default=0)
parser.add_argument('-y','--coordY',type=int, default=430)
parser.add_argument('-H','--height',type=int, default=100)
parser.add_argument('-W','--width',type=str, default=230)
args=vars(parser.parse_args())

x=args.get('coordX')
y=args.get('coordY')
h=args.get('height')
w=args.get('width')

cap = cv2.VideoCapture('/home/pavel/cursovoy/img/image_001.jpg')
(grabbed, frame)  = cap.read()
frame_init=frame[y:y+h, x:x+w]

ROI_x = x-50
ROI_y = y-50
ROI_width = w+100
ROI_hieght = h+100
if ROI_x < 0:
    ROI_x = 0
if ROI_y < 0:
    ROI_y = 0
if ROI_hieght > frame.shape[1]:
    ROI_hieght = frame.shape[1]
if ROI_width > frame.shape[0]:
    ROI_width = frame.shape[0]

frame_ROI = frame[ROI_y:ROI_y+ROI_hieght ,ROI_x:ROI_x+ROI_width]

gray_init=cv2.blur(frame_init,(5,5))
frame_ROI=cv2.blur(frame_ROI,(5,5))
numPixles_frame = np.prod(gray_init.shape[:2])
numPixles_ROI = np.prod(frame_ROI.shape[:2])
    
scar_x=cv2.Scharr(gray_init,-1,1,0)
scar_y=cv2.Scharr(gray_init,-1,0,1)
scar=scar_x+scar_y

scar_x=cv2.Scharr(frame_ROI,-1,1,0)
scar_y=cv2.Scharr(frame_ROI,-1,0,1)
scar_ROI=scar_x+scar_y

arr_scar=scar.flatten()
arr_gray=gray_init.flatten()
arr_ROI_scar=scar_ROI.flatten()
arr_ROI_frame=frame_ROI.flatten()

my_hist , x_edges, y_edges = np.histogram2d(arr_gray,arr_scar,bins=(50,50),normed=True)
my_ROI_hist,_,_ = np.histogram2d(arr_ROI_frame,arr_ROI_scar,bins=(50,50),normed=True)
plt.imshow(my_hist)
plt.show()
plt.imshow(my_ROI_hist)
plt.show()

hist=my_ROI_hist-my_hist
plt.imshow(hist)
plt.show()

cv2.imshow('img',frame_init)
cv2.waitKey(0)
cv2.imshow('ROI',frame_ROI)
cv2.waitKey(0)
binar=segmentator(frame=frame_ROI,hist=hist)
cv2.imshow("bin",binar)
cv2.waitKey(0)
cv2.destroyAllWindows()
