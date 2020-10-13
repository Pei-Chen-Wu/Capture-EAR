# Capture-EAR
使用python3.8擷取眼部特徵(EAR)
## EAR介紹
* 特性:
  
  EAR值稱為眼睛縱橫比，可透過此值來檢測眼睛是否眨眼
  
* 概念:

  ![Alt text](photo/圖片1.png)

  如上圖，將眼睛分為六點，再藉由下來公式，即可得當下眼睛EAR值
  
  ![Alt text](photo/圖片2.png)
  
  
## 程式碼: 

  分為影片與實時偵測

* 眼睛六點取法:
  
  使用 DLIB 68個人臉偵測點，可得下圖
  
  ![Alt text](photo/圖片4.png)
  
  經由對應點，可得右眼對應為37-42，左眼對應為43-48

* 成果: 
  將輸出的EAR值檔案輸入matlab，可得以下結果
  
   ![Alt text](photo/圖片3.png)
  
  EAR斜率也可輔助判斷是否眨眼
