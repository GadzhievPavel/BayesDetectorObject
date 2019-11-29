import cv2
import numpy as np

def segmentator(frame):
    print(frame.shape[:2])
    (hi,we)=frame.shape[:2]
    temp_frame=np.zeros(frame.shape[:2])
    number=0
    for w in range(we):
        temp=0
        for h in range(hi):
                if(temp!=frame[h][w] and frame[h][w]!=0):
                    number=number+1
                    temp_frame[h][w]=number
                if(temp==frame[h][w] and frame[h][w]!=0):
                    temp_frame[h][w]=number
                temp=frame[h][w]    
                print(temp_frame[h][w])
    for h in range(hi):
        temp=0
        for w in range(we):
           if(temp_frame[h][w]!=0):
               if(temp!=0):
                   temp_frame[h][w]=temp
                     

    return temp_frame                    
                    
cap=cv2.VideoCapture('/home/pavel/cursovoy/img_create/img_0.jpg')
(ret,frame)=cap.read()
frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
frame=segmentator(frame)
print(frame)
while(True):
    cv2.imshow("0",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()