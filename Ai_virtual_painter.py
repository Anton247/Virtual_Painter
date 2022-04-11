import cv2
import time
import handtrackingmodule as htm
import numpy as np
import os

overlayList=[] # список для хранения всех изображений

brushThickness = 25
eraserThickness = 100
drawColor=(255,0,255) #настройка фиолетового цвета

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8) # определение холста

#изображения в папке header
folderPath="Header"
myList=os.listdir(folderPath) #получение всех изображений, используемых в коде
#print(myList)
for imPath in myList: # чтение всех изображений из папки
    image=cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image) #вставка изображений по одному в список overlayList
header=overlayList[0] #хранение 1-го изображения 
cap=cv2.VideoCapture(0)
cap.set(3,1920)#width
cap.set(4,720)#height

detector = htm.handDetector(detectionCon=0.50,maxHands=1) #создание объекта

while True:

    # 1. Импортируйте изображение
    success, img = cap.read()
    img=cv2.flip(img,1) # зеркальной инверсией
    
    # 2. Найдите ориентиры для рук
    img = detector.findHands(img)#использование функций для соединения ориентиров
    lmList,bbox = detector.findPosition(img, draw=False)#использование функции для поиска определенного положения ориентира, рисование false означает отсутствие кругов на ориентирах
    
    if len(lmList)!=0:
        #print(lmList)
        x1, y1 = lmList[8][1],lmList[8][2]# кончик указательного пальца
        x2, y2 = lmList[12][1],lmList[12][2]# кончик среднего пальца
        
        # 3. Проверьте, какие пальцы подняты вверх
        fingers = detector.fingersUp()
        #print(fingers)

        # 4. Если режим выбора - два пальца подняты вверх
        if fingers[1] and fingers[2]:
            xp,yp=0,0
            #print("Selection Mode")
            #проверка нажатия
            if y1 < 125:
                if 250 < x1 < 450: # если я нажимаю на фиолетовую кисть
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:# если я нажимаю на синюю кисть
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:# если я нажимаю на зеленую кисть
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:#если я нажимаю на ластик
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)# режим выбора представлен в виде прямоугольника


        # 5. Если режим рисования - указательный палец поднят вверх
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)# режим рисования представлен в виде круга
            #print("Drawing Mode")
            if xp == 0 and yp == 0:#изначально xp и yp будут в точках 0,0, поэтому он проведет линию от 0,0 до той точки, в которой находится наш кончик.
                xp, yp = x1, y1 # поэтому, чтобы избежать этого, зададим xp=x1 и yp=y1
            #до сих пор мы создаем наш рисунок, но он удаляется, так как каждый раз наши кадры обновляются, поэтому мы должны определить наш холст, на котором мы можем рисовать и показывать.
            
            #ластик
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)#проводим линии от предыдущих координат к новым позициям 
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp,yp=x1,y1 # каждый раз присваивая значения xp, yp 
           
           # объединение двух окон в одно imgcanvas и img
    
    # 1 преобразование img в серый цвет
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    
    # 2 преобразование в двоичное изображение и затем инвертирование
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)# на холсте вся область, в которой мы рисовали, будет черной, а там, где она черная, она будет считаться белой, это создаст маску
    
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)#преобразование снова в серый цвет, потому что мы должны добавить RGB-изображение, т.е. img
    
    #добавляем исходный img с imgInv, делая это, мы получаем наш рисунок только в черном цвете
    img = cv2.bitwise_and(img,imgInv)
    
    #добавьте img и imgcanvas, сделав это, мы получим цвета на img
    img = cv2.bitwise_or(img,imgCanvas)


    #установка изображения заголовка
    img[0:125,0:1280]=header# на нашей рамке мы устанавливаем для изображений JPG значения H,W изображений jpg

    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imgCanvas)
    #cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)