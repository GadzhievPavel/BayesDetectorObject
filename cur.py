import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import math

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

def findDiffHist(frame,frame_ROI):
        numPixles_frame = np.prod(frame.shape[:2])
        numPixles_ROI = np.prod(frame_ROI.shape[:2])
        histogramObj = cv2.calcHist([frame], [0], None, [bins], [0, 255])/numPixles_frame 
        histogramROI = cv2.calcHist([frame_ROI], [0], None, [bins], [0, 255])/numPixles_ROI
        one_result_hist=histogramROI-histogramObj
        return one_result_hist

def findDiff2Hist(frame,frame_ROI):
        scar_x=cv2.Scharr(frame,-1,1,0)
        scar_y=cv2.Scharr(frame,-1,0,1)
        scar=scar_x+scar_y

        scar_x=cv2.Scharr(frame_ROI,-1,1,0)
        scar_y=cv2.Scharr(frame_ROI,-1,0,1)
        scar_ROI=scar_x+scar_y

        arr_scar=scar.flatten()
        arr_gray=frame.flatten()
        arr_ROI_scar=scar_ROI.flatten()
        arr_ROI_frame=frame_ROI.flatten()

        my_hist , x_edges, y_edges = np.histogram2d(arr_gray,arr_scar,bins=(150,150),normed=True)
        my_ROI_hist,_,_ = np.histogram2d(arr_ROI_frame,arr_ROI_scar,bins=(150,150),normed=True)
        resultHist=my_ROI_hist-my_hist
        return resultHist

class Core(object):
    
    def __init__(self,diff,x,y,w,h):
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.diff=diff
        self.bins=255


#fuction for binatyzation img by hist


#init argument
#parser = argparse.ArgumentParser()
#parser.add_argument('-d',"--diff",type=bool, default=True)
#parser.add_argument('-x',"--coordX",type=int, default=0)
#parser.add_argument('-y','--coordY',type=int, default=430)
#parser.add_argument('-H','--height',type=int, default=100)
#parser.add_argument('-W','--width',type=str, default=230)
#args=vars(parser.parse_args())

#x=args.get('coordX') #начало координат объекта х
#y=args.get('coordY') # y
#h=args.get('height') # высота облвсти объекта
#w=args.get('width')  # ширина объекта
#diff=args.get('diff') #выбор алгоритма
    def run(self):
        i=0
        cap = cv2.VideoCapture('/home/pavel/cursovoy/cycle_img/video.mpg')#read video stream
        (ret, frame_init)=cap.read()
        frame_init=frame_init[self.y:self.y+self.h, self.x:self.x+self.w] #first cadr
        status=cv2.imwrite('/home/pavel/cursovoy/img_create/first_init.jpg',frame_init)
        gray_init = cv2.cvtColor(frame_init,cv2.COLOR_BGR2GRAY) #frame gray

        while True:
            (grabbed, frame_сolor)  = cap.read()# create frame stream

    #cvtColor
            frame = cv2.cvtColor(frame_сolor,cv2.COLOR_BGR2GRAY) #frame on video

    #create ROI
            ROI_x = self.x-50# cоздать переменные x1,y1,w1,h1 через GUI задать размеры ROI
            ROI_y = self.y-50
            ROI_width = self.w+100
            ROI_hieght = self.h+100
            if ROI_x < 0:
                ROI_x = 0
            if ROI_y < 0:
                ROI_y = 0
            if ROI_hieght > frame.shape[1]:
                ROI_hieght = frame.shape[1]
            if ROI_width > frame.shape[0]:
                ROI_width = frame.shape[0]

            frame=cv2.GaussianBlur(frame,(5,5),sigmaX=0.9,sigmaY=0.5) # размытие по Гауссу начального изображения
    
    #cut ROI in gray frame                
            frame_ROI = frame[ROI_y:ROI_y+ROI_hieght ,ROI_x:ROI_x+ROI_width]
            status=cv2.imwrite('/home/pavel/cursovoy/img_create/ROI'+str(i)+'.jpg',frame_ROI)
            gray_init=cv2.blur(gray_init,(5,5))# размытие матриц 
            frame_ROI=cv2.blur(frame_ROI,(5,5))#

    #cv2.imshow('Image',gray_init)

            if(self.diff==True):#выбор алгоритма
                hist=findDiff2Hist(frame=gray_init,frame_ROI=frame_ROI)
                frame_ROI=range_segmentator(frame_ROI,hist)
            else:
                hist=findDiffHist(frame=gray_init,frame_ROI=frame_ROI)    
                frame_ROI=segmentator_one_hist(frame_ROI,hist)


            cv2.imshow('frame_ROI',frame_ROI)
            status=cv2.imwrite('/home/pavel/cursovoy/img_create/ROI'+str(i)+'.jpg',frame_ROI)
    #(hi,we)= frame_ROI[:2] 
    #frame_ROI=cv2.resize(frame_ROI,(50,50))#сжатие ROI
            kernal=np.ones((5,5),np.uint8)
            frame_ROI=cv2.dilate(frame_ROI,kernal)
            cv2.imshow("1dil",frame_ROI)
            kernal=np.ones((20,20),np.uint8)
            frame_ROI=cv2.erode(frame_ROI,kernal)#эрозия
            cv2.imshow('erode1',frame_ROI)
            frame_ROI=cv2.dilate(frame_ROI,kernal)
            cv2.imshow('dil2',frame_ROI)
    #frame_ROI=cv2.blur(frame_ROI,(15,15))
    #frame_ROI=cv2.erode(frame_ROI,kernal)
    #frame_ROI=cv2.blur(frame_ROI,(9,9))#размытие 
            ret,frame_ROI=cv2.threshold(frame_ROI,150,255,cv2.THRESH_BINARY)#порог бинаризации сжатого изображение
    #frame_ROI=cv2.resize(frame_ROI,(ROI_width,ROI_hieght),interpolation=cv2.INTER_CUBIC)  
            status=cv2.imwrite('/home/pavel/cursovoy/img_create/frame_ROII'+str(i)+'.jpg',frame_ROI)
            _,conturus,hierarhi=cv2.findContours(frame_ROI,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   
            if(len(conturus)!=0):
                cv2.drawContours(frame_ROI,conturus,-1,(150,0,150),1,cv2.LINE_AA,hierarhi,1)
                c=max(conturus,key=cv2.contourArea)
                x,y,w,h=cv2.boundingRect(c)
                cv2.rectangle(frame_сolor,(ROI_x+x,ROI_y+y),(ROI_x+x+w,ROI_y+y+h),(150,255,100),2)
                gray_init=frame[ROI_y+y:ROI_y+y+h, ROI_x+x:ROI_x+x+w]
                status=cv2.imwrite('/home/pavel/cursovoy/img_create/gray_init_'+str(i)+'.jpg',gray_init)
                self.y=ROI_y+y
                self.x=ROI_x+x
                self.h=h
                self.w=w

            status=cv2.imwrite('/home/pavel/cursovoy/img_create/img_noseg_'+str(i)+'.jpg',frame_сolor)
            cv2.imshow('ROI',frame_сolor)
    #cv2.imshow('VIDEO', frame)  
            i=i+1
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()