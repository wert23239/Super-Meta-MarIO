import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from directkeys import *
from getkeys import key_check
import os
from alexnet import alexnet
WIDTH=28
HEIGHT=28
LR= 1e-3
EPOCH=8
MODEL_NAME= 'MarioAI-{}-{}-{}-epochs.model'.format(LR,'alexnetv2',EPOCH)

Gather=False
def KeyPressButtons(key):
    
    if(key==0):
        ReleaseKey(X)
        PressKey(Z)
    elif(key==2):
       # print("here")
        PressKey(X)
        PressKey(Z)
    else:
        ReleaseKey(X)
        ReleaseKey(Z)

def KeyPressArrows(key):
    
    if(key==0):
        ReleaseKey(X)
        PressKey(Z)
    elif(key==2):
       # print("here")
        PressKey(X)
        PressKey(Z)
    else:
        ReleaseKey(X)
        ReleaseKey(Z)                     
# pyautogui.typewrite(['down','enter'])
# for i in list(range(4))[::-1]:
#     print(i+1)
#     time.sleep(1)
# pyautogui.typewrite(['down','enter'])

model =alexnet(WIDTH,HEIGHT,LR)
model.load(MODEL_NAME)

#model2=alexnet(WIDTH,HEIGHT,LR)
#model2.load(MODEL_NAME2)


def process_img(original_image):
    processed_img=cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img= cv2.resize(processed_img,(28,28))
    #rocess_img=cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img

def reset():
    pyautogui.moveTo(8,50,duration=.25)
    pyautogui.click(8,50)
    pyautogui.moveTo(8, 200, duration=0.25)
    pyautogui.click(8,200)
    pyautogui.moveTo(220, 200, duration=0.25)
    pyautogui.click(220,180)

def pause():
    pyautogui.click(70,50)
    pyautogui.moveTo(70,75, duration=0.25)
    pyautogui.click(70,75)    

def determine_key(time_step):
    if 0<=time_step<30:
        return RIGHT
    elif 30<=time_step:
        return X

def screen_record(): 
    last_time = time.time()
    time_step=0
    paused=False
    while(True):
        if not paused:
            # 800x600 windowed mode
            #PressKey(determine_key(time_step))
            printscreen =  np.array(ImageGrab.grab(bbox=(0,60,580,530)))
            new_screen=process_img(printscreen)
            #print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()

            prediction=model.predict([new_screen.reshape(WIDTH,HEIGHT,1)])[0]
            #prediction2=model.predict2([new_screen.reshape(WIDTH,HEIGHT,1)])[0]
            moves=list(np.around(prediction))
            #moves2=list(np.around(prediction2))
            print(moves,prediction)
            if moves==[1,0,0,0]:
                KeyPressButtons(0)
            elif moves==[0,0,1,0]:
                KeyPressButtons(2)    
            else:
                KeyPressButtons(3)   
            time.sleep(1.0)
            ReleaseKey(X)
            #cv2.imshow('window2',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
            #v2.imshow('window',new_screen)k     
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break         
            #ReleaseKey(determine_key(time_step))    
            time_step+=1  
        keys=key_check()
        #print(keys)
        if 'P' in keys:
            paused=(1-int(paused))
            time.sleep(1)
            KeyPressButtons(3)
            pause()

        if 'Q' in keys:
            cv2.destroyAllWindows()
            break

#screen_record()
reset()
pause()
screen_record()
pause()


