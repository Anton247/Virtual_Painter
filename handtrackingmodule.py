import cv2
import mediapipe as mp
import time
import math
import numpy as np

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):#конструктор
        self.mode=mode 
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mpHands=mp.solutions.hands#инициализация модуля рук для данного экземпляра
        self.hands=self.mpHands.Hands(self.maxHands) #объект для рук для конкретного экземпляра
        self.mpDraw=mp.solutions.drawing_utils#объект для рисования
        self.tipIds = [4, 8, 12, 16, 20] #список всех ориентиров кончиков пальцев

    def findHands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#преобразование в RGB, поскольку распознавание рук работает только на RGB-изображении
        self.results=self.hands.process(imgRGB)#обработка RGB-изображения 
        if self.results.multi_hand_landmarks:# дает x,y,z каждого ориентира или если нет руки, то NONE
            for handLms in self.results.multi_hand_landmarks:#каждый ориентир руки в результатах
                if draw:
                     self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)#соединение точек на нашей руке
        
        return img

    def findPosition(self,img,handNo=0,draw=True):
        xList=[]
        yList=[]
        bbox=[]
        self.lmlist=[]
        if self.results.multi_hand_landmarks:# дает x,y,z каждого ориентира    
            myHand=self.results.multi_hand_landmarks[handNo]#дает результат для конкретной руки 
            for id,lm in enumerate(myHand.landmark):#дает id и lm(x,y,z)
                h,w,c=img.shape#получение h,w для преобразования десятичных значений x,y в пиксели 
                cx,cy=int(lm.x*w),int(lm.y*h)# координаты пикселей для ориентиров
                # print(id, cx, cy)
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
            print(self.lmlist)    
            xmin,xmax=min(xList),max(xList)
            ymin,ymax=min(yList),max(yList)
            bbox=xmin,ymin,xmax,ymax

            if draw:
                cv2.rectangle(img,(bbox[0]-20,bbox[1]-20),(bbox[2]+20,bbox[3]+20),(0,255,0),2)

        return self.lmlist,bbox

    def fingersUp(self):#проверка, какой палец открыт 
        fingers = []#сохранение конечного результата
        # Знак < только когда мы используем функцию flip, чтобы избежать зеркальной инверсии, иначе > знак
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:#проверка того, что x позиция 4 находится справа от x позиции 3
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):#проверка точки наконечника ниже точки наконечника-2 (только в направлении Y)
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15,t=3):# нахождение расстояния между двумя точками p1 и p2
        x1, y1 = self.lmlist[p1][1],self.lmlist[p1][2]#получение x,y из p1
        x2, y2 = self.lmlist[p2][1],self.lmlist[p2][2]#получение x,y из p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2#получение центральной точки

        if draw: #рисование прямой и окружностей по точкам
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
    

def main():

    PTime=0# previous time
    CTime=0# current time
    cap=cv2.VideoCapture(0)
    detector=handDetector()

    while True:
        success, img=cap.read()#T or F,frame
        img =detector.findHands(img)
        lmlist,bbox= detector.findPosition(img)
        if len(lmlist)!=0:
            print(lmlist[4])

        CTime=time.time()#current time
        fps=1/(CTime-PTime)#FPS
        PTime=CTime#previous time is replaced by current time

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)# showing Fps on screen


        cv2.imshow("Image",img)#showing img not imgRGB
        cv2.waitKey(1)


if __name__=="__main__":
    main()