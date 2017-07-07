import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from directkeys import *



# pyautogui.typewrite(['down','enter'])
# for i in list(range(4))[::-1]:
#     print(i+1)
#     time.sleep(1)
# pyautogui.typewrite(['down','enter'])

def process_img(original_image):
    processed_img=cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    process_img=cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img

def reset():
    pyautogui.click(5,50)
    pyautogui.moveTo(5, 180, duration=0.25)
    pyautogui.click(5,180)
    pyautogui.moveTo(200, 180, duration=0.25)
    pyautogui.click(200,180)

def pause():
    pyautogui.click(50,50)
    pyautogui.moveTo(50,60, duration=0.25)
    pyautogui.click(50,60)    

def determine_key(time_step):
    if 0<=time_step<30:
        return RIGHT
    elif 30<=time_step:
        return X

def screen_record(): 
    last_time = time.time()
    time_step=0
    while(time_step<500):
        # 800x600 windowed mode



        printscreen =  np.array(ImageGrab.grab(bbox=(0,60,580,530)))
        new_screen=process_img(printscreen)
        #print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        #cv2.imshow('window2',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        cv2.imshow('window',new_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            ReleaseKey(Z) 
            ReleaseKey(X)
            time.sleep(3)   
            cv2.destroyAllWindows() 
            break      
        time_step+=1   
screen_record()
#reset()
#pause()
#screen_record()
#pause()


