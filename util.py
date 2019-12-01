import cv2
import numpy as np

def segmentator(frame):
    (hi,we)=frame.shape[:2]
    temp_frame=np.zeros(frame.shape[:2])
    number=0
    for w in range(we):
        temp=0
        for h in range(hi):
                if(temp!=frame[h][w] and frame[h][w]==255):
                    number=number+1
                    temp_frame[h][w]=number
                if(temp==frame[h][w] and frame[h][w]==255):
                    temp_frame[h][w]=number
                temp=frame[h][w]               
    return temp_frame                    

def segmentator_cross(frame):
    (hi,we)=frame.shape[:2]
    B=0
    C=0
    num=0
    for h in range(hi):
        for w in range(we):
            kh=h-1
            if(kh<=0):
                kh=1
                B=0
            else:
                B=frame[kh][w]
            kw=w-1
            if(kw<=0):
                kw=1
                C=0
            else:
                C=frame[h][kw]
            A=frame[h][w]
            if(A!=0):
                if( B==0 and C==0):
                    num=num+1
                    frame[h][w]=num
                elif(B!=0 and C==0):
                    frame[h][w]=B
                elif(B==0 and C!=0):
                    frame[h][w]=C
                elif(B!=0 and C!=0):
                    if(B==C):
                        frame[h][w]=B
                    else:
                        frame[h][w]=B                           
    return frame

def no_hole(frame):
    output=frame
    fill=0
    for y in range(frame.shape[0]):
        pre=0
        for x in range(frame.shape[1]):
            if((fill==0) and pre==0 and output[y][x]):
                fill=fill+1
            if((fill!=0)and pre !=0 and output[y][x]):
                fill=fill-1
            if (fill!=0):
                output[y][x]=255
            else:
                output[y][x]=0
            pre=output[y][x]            
    return frame