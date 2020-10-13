# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 01:57:46 2019

@author: Coco_wu
"""

import numpy as np 
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import time

plt.rcParams['savefig.dpi']=300
plt.rcParams['figure.dpi']=300

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
    
	return ear

def eye_distense(eye):
    H = dist.euclidean(eye[2], eye[4])
    return H



detector = dlib.get_frontal_face_detector()# 人脸检测器
predictor = dlib.shape_predictor( r'C:\Users\Nclab\anaconda3\Lib\site-packages\face_recognition_models-20200730T043942Z-001\face_recognition_models\models\shape_predictor_68_face_landmarks.dat')# 人脸特征点检测器

#EYE_AR_THRESH = 0.2# EAR阈值
#EYE_AR_CONSEC_FRAMES = 2# 当EAR小于阈值时，接连多少帧一定发生眨眼动作
EYE_AR_CONSEC_FRAMES1 = 3
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1


frame_counter = 0
blink_counter = 0

x=0
s=0
A=0
B=0
f1=0
f2=0
F=0
H_max=16.07
t1=0
t2=0
t3=0
t4=0
Attention=0
j=0
cap = cv2.VideoCapture(r'C:\Users\Nclab\Desktop\Coco\test\資料\7(high).mp4')
with open('e_7.txt','w') as f ,open('h_7.txt','w') as k:
    while(1):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  
        ret, img = cap.read()# 讀取一楨
        if ret != True:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰階化
        rects = detector(gray, 0)# 人脸檢測
        for i in rects:  
        #print('-'*20)
            shape = predictor(gray, i)# 檢測特徵點
            points = face_utils.shape_to_np(shape)# convert the facial landmark (x, y)-coordinates to a NumPy array
            leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]# 取出左眼特徵點
            rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]# 取出右眼特徵點
            leftEAR = eye_aspect_ratio(leftEye)# 计算左眼EAR
            rightEAR = eye_aspect_ratio(rightEye)# 计算右眼EAR
            ear = (leftEAR + rightEAR) / 2.0# 求左右眼EAR的均值
            B=ear-A      #B為斜率
            A=ear  
            x+=(1/30)
            s+=1
            y=np.array(ear)
            print(f"{y}", file=f)
            
            h_left=points[LEFT_EYE_START:LEFT_EYE_END + 1]
            h_right=points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
            H_left=eye_distense(h_left)
            H_right=eye_distense(h_right)
            H=(H_left+H_right)/2
            h=np.atleast_1d(H)
            print(f"{h}", file=k) 
"""     if (B<-0.04 or ear<0.2) and f1==0 and f2==0 :            
            f1=1;
        if f1==1: 
            if B<-0.04 and ear<0.2 and f2==0:
                f2=1;
                blink_counter += 1;
                plt.plot(x, ear ,'r-o',lw=5)
        if B>=-0.04 and ear>=0.2 and f1==1 and f2==1:
            f1=0;
            f2=0;
        if (s % 300==0):
            print('blink_counter = {0}'.format(blink_counter));
            #print(H);
            F=np.array(blink_counter);
            blink_counter=0;    
            
        if H < (H_max*0.8) and j==0 :
            t1=time.time()
            j=1
            print(1)
        if H < (H_max*0.2) and j==1 :
            t2=time.time()
            j=2
            print(2)
        if H > (H_max*0.2) and j==2:
            t3=time.time()
            j=3
            print(3)
        if H > (H_max*0.8)and j==3:   
            t4=time.time()   
            j=0
            print(4)
                        
        t3_2=(t3-t2)               
        t4_1=(t4-t1)   
        if t4_1 <0.15:
            C=0
        if t4_1 >=0.15 and t4_1 < 0.3 :
            C=1
        if t4_1 >= 0.3:
            C=2
        
        if t3_2 >0 and t4_1>0:
            f=t3_2*100/t4_1
            Attention=0.5*C+5*f  """
        
   
        
        #print('ear = {0}'.format(ear))
        
        
       #leftEyeHull = cv2.convexHull(leftEye)# 寻找左眼轮廓
        #rightEyeHull = cv2.convexHull(rightEye)# 寻找右眼轮廓
        #cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)# 绘制左眼轮廓
        #cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)# 绘制右眼轮廓
        
        #print('B = {0}'.format(B))        
        #cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        #cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
      
    #cv2.imshow("Frame", img)
    
    
"""plt.subplot(211)
    plt.plot(x, y ,'b-o',lw=2)
    plt.ylim(0.05,0.4)
    plt.ylabel('EAR')
    
    plt.subplot(212)
    plt.plot(x, h ,'r-o',lw=2)
    plt.ylim(3,17)
    plt.xlabel('Time(S)')
    plt.ylabel('ear opening')"""
    #@plt.ylabel('EAR')
    #plt.plot(x, h ,'a-o',lw=5)
    #if Attention >0:
        #print(Attention)
    #print(t1)    
    #print(h)

#print('blink_counter = {0}'.format(blink_counter))
cap.release()
cv2.destroyAllWindows()
