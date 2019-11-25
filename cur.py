import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

def range_segmentator(frame, hist):
    temp_frame=frame.copy()
    for i in range(len(hist)):
        if(hist[i]<0):
            temp_frame[np.where(frame == i)]=255
            frame=temp_frame    
    frame[np.where(frame!=255)]=0        
    return frame           
#const
bins=256
lw=1

cap = cv2.VideoCapture('video.mp4')#read video stream
cap_init=cv2.VideoCapture('etlon/etlon.jpg')#read first roi frame

#init argument
parser = argparse.ArgumentParser()
parser.add_argument('-x',"--coordX",type=int, default=420)
parser.add_argument('-y','--coordY',type=int, default=0)
parser.add_argument('-H','--height',type=int, default=230)
parser.add_argument('-W','--width',type=str, default=100)
args=vars(parser.parse_args())

x=args.get('coordX')
y=args.get('coordY')
h=args.get('height')
w=args.get('width')

(ret, frame_init)=cap.read()
frame_init=frame_init[x:x+w ,y:y+h] #first cadr

#(ret, frame_obj) = cap_init.read()  #create frame's first roi
#frame_obj=cv2.cvtColor(frame_obj,cv2.COLOR_BGR2GRAY)
#create histplot
fig_init, initx = plt.subplots()# create plot for img roi
fig, ax = plt.subplots()#create plot for frame video stream

#creat axis, title plot
initx.set_title('Histogram ROI')
ax.set_title('Histogram Video')

initx.set_xlabel('Bin')
initx.set_ylabel('%')

ax.set_xlabel('Bin')
ax.set_ylabel('%')

ax.set_xlim(0,bins-1)
initx.set_xlim(0,bins-1)

ax.set_ylim(0,1)
initx.set_ylim(0,1)

plt.ion()
plt.show()

lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw)
lineGray_Image, = initx.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw)


while True:
    (grabbed, frame)  = cap.read()# create frame stream

    #cvtColor
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_init = cv2.cvtColor(frame_init,cv2.COLOR_BGR2GRAY)
    ROI_x = x-30
    ROI_y = y-30
    ROI_width = w+30
    ROI_hieght = h+30
    if ROI_x < 0:
        ROI_x = 0
    if ROI_y < 0:
        ROI_y = 0
    if ROI_hieght > frame.shape[1]:
        ROI_hieght = frame.shape[1]
    if ROI_width > frame.shape[0]:
        ROI_width = frame.shape[0]            
    frame_ROI = frame[ROI_x:ROI_x+ROI_width ,ROI_y:ROI_y+ROI_hieght]
    print('X=',ROI_x,'y=',ROI_y,'width=',ROI_width,'hiegth=',ROI_hieght)

    cv2.imshow('VIDEO', frame)  
    cv2.imshow('Image',gray_init)

    numPixles_frame = np.prod(gray_init.shape[:2])
    numPixles_ROI = np.prod(frame_ROI.shape[:2])
    
    print(numPixles_frame)
    print(numPixles_ROI)

    histogramObj = cv2.calcHist([gray_init], [0], None, [bins], [0, 255])/numPixles_frame
    histogramROI = cv2.calcHist([frame_ROI], [0], None, [bins], [0, 255])/numPixles_ROI

    resultHist = histogramROI-histogramObj
    frame_ROI=range_segmentator(frame_ROI,resultHist)  
    cv2.imshow('ROI',frame_ROI)

    print(resultHist)
    lineGray_Image.set_ydata(resultHist)
    lineGray.set_ydata(histogramObj)

    fig.canvas.draw()
    fig_init.canvas.draw()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()