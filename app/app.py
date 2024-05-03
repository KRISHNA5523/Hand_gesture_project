from flask import Flask, render_template, request
from handTracker import *
import cv2
import numpy as np
import threading
import mediapipe as mp
import random

app = Flask(__name__, static_folder='static')

#new
def run_calculator():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    cam = cv2.VideoCapture(0)
    x = []
    y = []
    text = ""
    k = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    idset = ["", "1", "12", "123", "1234", "01234", "0", "01", "012", "0123", "04", "4", "34", "014", "14", "234"]
    op = ["", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "+", "-", "*", "/"]

    while True:
        success, img = cam.read()
        imgg = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = imgg.shape
                    if id == 0:
                        x = []
                        y = []
                    x.append(int((lm.x) * w))
                    y.append(int((1 - lm.y) * h))

                    if len(y) > 20:
                        id = ""
                        big = [x[3], y[8], y[12], y[16], y[20]]
                        small = [x[4], y[6], y[10], y[14], y[18]]

                        for i in range(len(big)):
                            if big[i] > small[i]:
                                id += str(i)

                        if id in idset:
                            k[idset.index(id)] += 1

                            for i in range(len(k)):
                                if k[i] > 20:
                                    if i == 15:
                                        if text:
                                            try:
                                                ans = str(eval(text))
                                                text = "= " + ans
                                            except Exception as e:
                                                print(f"Error evaluating expression: {e}")
                                        for i in range(len(k)):
                                            k[i] = 0
                                    else:
                                        text += op[i]
                                        for i in range(len(k)):
                                            k[i] = 0

                cv2.putText(imgg, text, (100, 120), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 0), 5)
                mpDraw.draw_landmarks(imgg, handLms, mpHands.HAND_CONNECTIONS)

        else:
            text = " "

        cv2.imshow("WebCam", imgg)
        cv2.waitKey(1)



class ColorRect():
    def __init__(self, x, y, w, h, color, text='', alpha = 0.5):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text=text
        self.alpha = alpha
        
    
    def drawRect(self, img, text_color=(255,255,255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        
        alpha = self.alpha
        bg_rec = img[self.y : self.y + self.h, self.x : self.x + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = self.color
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1-alpha, 1.0)
        
       
        img[self.y : self.y + self.h, self.x : self.x + self.w] = res

        
        tetx_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x + self.w/2 - tetx_size[0][0]/2), int(self.y + self.h/2 + tetx_size[0][1]/2))
        cv2.putText(img, self.text,text_pos , fontFace, fontScale,text_color, thickness)


    def isOver(self,x,y):
        if (self.x + self.w > x > self.x) and (self.y + self.h> y >self.y):
            return True
        return False
    #new
    def isOverPoint(self, x, y):
       # x, y = point
        #return self.isOver(x, y) 
        if (self.x + self.w > x > self.x) and (self.y + self.h> y >self.y):
            return True
        return False 

detector = HandTracker(detectionCon=1)


def run_hand_gesture_project():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # creating canvas to draw on it
    canvas = np.zeros((720,1280,3), np.uint8)

    # define a previous point to be used with drawing a line
    px,py = 0,0
    #initial brush color
    color = (255,0,0)
    #
    brushSize = 5
    eraserSize = 20
    #

    ########### creating colors ########
    # Colors button
    colorsBtn = ColorRect(200, 0, 100, 100, (120,255,0), 'Colors')

    # Put the calc button
    calculatorBtn = ColorRect(1100, 500, 100, 100, (255, 0, 255), 'Calc')


    colors = []
    #random color
    b = int(random.random()*255)-1
    g = int(random.random()*255)
    r = int(random.random()*255)
    print(b,g,r)
    colors.append(ColorRect(300,0,100,100, (b,g,r)))
    #red
    colors.append(ColorRect(400,0,100,100, (0,0,255)))
    #blue
    colors.append(ColorRect(500,0,100,100, (255,0,0)))
    #green
    colors.append(ColorRect(600,0,100,100, (0,255,0)))
    #yellow
    colors.append(ColorRect(700,0,100,100, (0,255,255)))
    #erase (black)
    colors.append(ColorRect(800,0,100,100, (0,0,0), "Eraser"))


    #clear
    clear = ColorRect(900,0,100,100, (100,100,100), "Clear")

    ########## pen sizes #######
    pens = []
    for i, penSize in enumerate(range(5,25,5)):
        pens.append(ColorRect(1100,50+100*i,100,100, (50,50,50), str(penSize)))

    penBtn = ColorRect(1100, 0, 100, 50, color, 'Pen')

    # white board button
    boardBtn = ColorRect(50, 0, 100, 100, (255,255,0), 'Board')




    #define a white board to draw on
    whiteBoard = ColorRect(50, 120, 1020, 580, (255,255,255),alpha = 0.6)


    coolingCounter = 20
    hideBoard = True
    hideColors = True
    hidePenSizes = True

    while True:

        if coolingCounter:
            coolingCounter -=1
            #print(coolingCounter)

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)
        

        detector.findHands(frame)
        positions = detector.getPostion(frame, draw=False)
        upFingers = detector.getUpFingers(frame)

        


        if upFingers:
            x, y = positions[8][0], positions[8][1]
            if upFingers[1] and not whiteBoard.isOver(x, y):
                px, py = 0, 0

                ##### pen sizes ######
                if not hidePenSizes:
                    for pen in pens:
                        if pen.isOver(x, y):
                            brushSize = int(pen.text)
                            pen.alpha = 0
                        else:
                            pen.alpha = 0.5

                ####### chose a color for drawing #######
                if not hideColors:
                    for cb in colors:
                        if cb.isOver(x, y):
                            color = cb.color
                            cb.alpha = 0
                        else:
                            cb.alpha = 0.5

                    #Clear 
                    if clear.isOver(x, y):
                        clear.alpha = 0
                        canvas = np.zeros((720,1280,3), np.uint8)
                    else:
                        clear.alpha = 0.5
                
                # color button
                if colorsBtn.isOver(x, y) and not coolingCounter:
                    coolingCounter = 10
                    colorsBtn.alpha = 0
                    hideColors = False if hideColors else True
                    colorsBtn.text = 'Colors' if hideColors else 'Hide'
                else:
                    colorsBtn.alpha = 0.5
                
                # Pen size button
                if penBtn.isOver(x, y) and not coolingCounter:
                    coolingCounter = 10
                    penBtn.alpha = 0
                    hidePenSizes = False if hidePenSizes else True
                    penBtn.text = 'Pen' if hidePenSizes else 'Hide'
                else:
                    penBtn.alpha = 0.5

                
                #white board button
                if boardBtn.isOver(x, y) and not coolingCounter:
                    print("x:", x)
                    print("y:", y)

                    coolingCounter = 10
                    boardBtn.alpha = 0
                    hideBoard = False if hideBoard else True
                    boardBtn.text = 'Board' if hideBoard else 'Hide'

                else:
                    boardBtn.alpha = 0.5



    # Inside the main loop
                if calculatorBtn.isOverPoint(x, y) and not coolingCounter:
                    coolingCounter = 10
                    calculatorBtn.alpha = 0
                    run_calculator()
                else:
                    calculatorBtn.alpha = 0.5
        
                        

            elif upFingers[1] and not upFingers[2]:
                if whiteBoard.isOver(x, y) and not hideBoard:
                    #print('index finger is up')
                    cv2.circle(frame, positions[8], brushSize, color,-1)
                    #drawing on the canvas
                    if px == 0 and py == 0:
                        px, py = positions[8]
                    if color == (0,0,0):
                        cv2.line(canvas, (px,py), positions[8], color, eraserSize)
                    else:
                        cv2.line(canvas, (px,py), positions[8], color,brushSize)
                    px, py = positions[8]
            
            else:
                px, py = 0, 0
            
        calculatorBtn.drawRect(frame)
        cv2.rectangle(frame, (calculatorBtn.x, calculatorBtn.y), (calculatorBtn.x + calculatorBtn.w, calculatorBtn.y + calculatorBtn.h), (255, 255, 255), 2)    
        # put colors button
        colorsBtn.drawRect(frame)
        cv2.rectangle(frame, (colorsBtn.x, colorsBtn.y), (colorsBtn.x +colorsBtn.w, colorsBtn.y+colorsBtn.h), (255,255,255), 2)

        # put white board buttin
        boardBtn.drawRect(frame)
        cv2.rectangle(frame, (boardBtn.x, boardBtn.y), (boardBtn.x +boardBtn.w, boardBtn.y+boardBtn.h), (255,255,255), 2)

        #put the white board on the frame
        if not hideBoard:       
            whiteBoard.drawRect(frame)
            ########### moving the draw to the main image #########
            canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(canvasGray, 20, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, imgInv)
            frame = cv2.bitwise_or(frame, canvas)


        ########## pen colors' boxes #########
        if not hideColors:
            for c in colors:
                c.drawRect(frame)
                cv2.rectangle(frame, (c.x, c.y), (c.x +c.w, c.y+c.h), (255,255,255), 2)

            clear.drawRect(frame)
            cv2.rectangle(frame, (clear.x, clear.y), (clear.x +clear.w, clear.y+clear.h), (255,255,255), 2)


        ########## brush size boxes ######
        penBtn.color = color
        penBtn.drawRect(frame)
        cv2.rectangle(frame, (penBtn.x, penBtn.y), (penBtn.x +penBtn.w, penBtn.y+penBtn.h), (255,255,255), 2)
        if not hidePenSizes:
            for pen in pens:
                pen.drawRect(frame)
                cv2.rectangle(frame, (pen.x, pen.y), (pen.x +pen.w, pen.y+pen.h), (255,255,255), 2)


        cv2.imshow('video', frame)
        #cv2.imshow('canvas', canvas)
        k= cv2.waitKey(1)
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run_hand_gesture', methods=['POST'])
def run_hand_gesture():
    run_hand_gesture_project()
    return 'Thank You!!!...'

if __name__ == '__main__':
    app.run(debug=True)
