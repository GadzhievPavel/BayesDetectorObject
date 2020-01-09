import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import math
from util import segmentator_cross, no_hole


#fuction for binatyzation img by hist
def range_segmentator(frame, hist):
    temp_frame=frame.copy()
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            if(hist[j,i]<0):
                temp_frame[np.where(frame == i)]=255
                frame=temp_frame    
    frame[np.where(frame!=255)]=0        
    return frame           

def segmentator_one_hist(frame,hist):
    temp_frame=frame.copy()
    for i in range(len(hist)):
        if(hist[i]<0):
            temp_frame[np.where(frame==i)]=255
            frame=temp_frame
    frame[np.where(frame!=255)]=0
    return frame

def get_segments(frame):
    number=0
    for i in range(frame.shape[0]):
        temp_color=0
        for j in range(frame.shape[1]):
            color=frame[i][j]
            if(color!=0):
                if(temp_color!=color):
                    number=number+1
                    segments={number:({'x':i,'y':j})}
                    temp_color=color
                else:
                    segments={number:({'x':i,'y':j})}
                    temp_color=color

    return  segments  
#const
bins=256
lw=1

i=0

cap = cv2.VideoCapture('/home/pavel/cursovoy/cycle_img/video200ebanbIxSobak.mpg')#read video stream
cap_init=cv2.VideoCapture('etlon/etlon.jpg')#read first roi frame

#init argument
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

(ret, frame_init)=cap.read()
frame_init=frame_init[y:y+h, x:x+w] #first cadr
status=cv2.imwrite('/home/pavel/cursovoy/img_create/first_init.jpg',frame_init)
#create histplot

#fig_init, initx = plt.subplots()# create plot for img roi
#fig, ax = plt.subplots()#create plot for frame video stream

#creat axis, title plot
#initx.set_title('Histogram ROI')
#ax.set_title('Histogram Video')

#initx.set_xlabel('Bin')
#initx.set_ylabel('%')

#ax.set_xlabel('Bin')
#ax.set_ylabel('%')

#ax.set_xlim(0,bins-1)
#initx.set_xlim(0,bins-1)

#ax.set_ylim(0,1)
#initx.set_ylim(0,1)

#plt.ion()
#plt.show()

#lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw)
#lineGray_Image, = initx.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw)

gray_init = cv2.cvtColor(frame_init,cv2.COLOR_BGR2GRAY) #frame gray

while True:
    (grabbed, frame_сolor)  = cap.read()# create frame stream

    #cvtColor
    frame = cv2.cvtColor(frame_сolor,cv2.COLOR_BGR2GRAY) #frame on video

    #create ROI
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

    frame=cv2.GaussianBlur(frame,(5,5),sigmaX=0.9,sigmaY=0.5)
    
    #cut ROI in gray frame                
    frame_ROI = frame[ROI_y:ROI_y+ROI_hieght ,ROI_x:ROI_x+ROI_width]
    status=cv2.imwrite('/home/pavel/cursovoy/img_create/ROI'+str(i)+'.jpg',frame_ROI)
    gray_init=cv2.blur(gray_init,(5,5))
    frame_ROI=cv2.blur(frame_ROI,(5,5))

    cv2.imshow('Image',gray_init)

    #count pix in frame and ROI
    numPixles_frame = np.prod(gray_init.shape[:2])
    numPixles_ROI = np.prod(frame_ROI.shape[:2])
    
    scar_x=cv2.Scharr(gray_init,-1,1,0)
    scar_y=cv2.Scharr(gray_init,-1,0,1)
    scar=scar_x+scar_y
    status=cv2.imwrite('/home/pavel/cursovoy/img_create/scar_x'+str(i)+'.jpg',scar)
    #histogramScarInit= cv2.calcHist([scar],[0],None,[bins],[0,255])/numPixles_frame
    histogramObj = cv2.calcHist([gray_init], [0], None, [bins], [0, 255])/numPixles_frame #считать 2 канальное изображение для яркости и градиента 
 

    scar_x=cv2.Scharr(frame_ROI,-1,1,0)
    scar_y=cv2.Scharr(frame_ROI,-1,0,1)
    scar_ROI=scar_x+scar_y
    status=cv2.imwrite('/home/pavel/cursovoy/img_create/scar_ROI'+str(i)+'.jpg',scar)
    #histogramScarROI=cv2.calcHist([scar_ROI],[0],None,[bins],[0,255])/numPixles_ROI
    histogramROI = cv2.calcHist([frame_ROI], [0], None, [bins], [0, 255])/numPixles_ROI #погуглить про готовый алгоритм градиентного поля (его модуля)"gradient field" ""

    one_result_hist=histogramROI-histogramObj

    arr_scar=scar.flatten()
    arr_gray=gray_init.flatten()
    arr_ROI_scar=scar_ROI.flatten()
    arr_ROI_frame=frame_ROI.flatten()

    my_hist , x_edges, y_edges = np.histogram2d(arr_gray,arr_scar,bins=(150,150),normed=True)
    my_ROI_hist,_,_ = np.histogram2d(arr_ROI_frame,arr_ROI_scar,bins=(150,150),normed=True)
    plt.imshow(my_hist)
    plt.show()
    plt.imshow(my_ROI_hist)
    plt.show()

    # находим отличия на гистограммах
    resultHist=my_ROI_hist-my_hist
    plt.imshow(resultHist)
    plt.show()

    # бинаризируем на основе найденных отличий
    frame_ROI=range_segmentator(frame_ROI,resultHist)
    #frame_ROI=segmentator_one_hist(frame_ROI,one_result_hist)
    cv2.imshow('frame_ROI',frame_ROI)
    (hi,we)= frame_ROI[:2] 
    frame_ROI=cv2.resize(frame_ROI,(50,50))
    frame_ROI=cv2.erode(frame_ROI,(41,41))
    frame_ROI=cv2.erode(frame_ROI,(11,11))
    frame_ROI=cv2.blur(frame_ROI,(9,9))
    ret,frame_ROI=cv2.threshold(frame_ROI,150,255,cv2.THRESH_BINARY)
    frame_ROI=cv2.resize(frame_ROI,(ROI_width,ROI_hieght),interpolation=cv2.INTER_CUBIC)  
    status=cv2.imwrite('/home/pavel/cursovoy/img_create/frame_ROII'+str(i)+'.jpg',frame_ROI)
    _,conturus,hierarhi=cv2.findContours(frame_ROI,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   
    if(len(conturus)!=0):
        cv2.drawContours(frame_ROI,conturus,-1,(150,0,150),1,cv2.LINE_AA,hierarhi,1)
        c=max(conturus,key=cv2.contourArea)
        x,y,w,h=cv2.boundingRect(c)
        cv2.rectangle(frame_сolor,(ROI_x+x,ROI_y+y),(ROI_x+x+w,ROI_y+y+h),(150,255,100),2)
        gray_init=frame[ROI_y+y:ROI_y+y+h, ROI_x+x:ROI_x+x+w]
        status=cv2.imwrite('/home/pavel/cursovoy/img_create/gray_init_'+str(i)+'.jpg',gray_init)
        y=ROI_y+y
        x=ROI_x+x
        h=h
        w=w

    status=cv2.imwrite('/home/pavel/cursovoy/img_create/img_noseg_'+str(i)+'.jpg',frame_сolor)
    cv2.imshow('ROI',frame_ROI)
    cv2.imshow('VIDEO', frame)  
    i=i+1
 #   lineGray_Image.set_ydata(resultHist)
 #   lineGray.set_ydata(histogramObj)

 #   fig.canvas.draw()
 #   fig_init.canvas.draw()

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()