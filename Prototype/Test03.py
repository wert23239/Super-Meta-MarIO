import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from directkeys import *
from getkeys import key_check
from grabscreen import grab_screen
import os

# pyautogui.typewrite(['down','enter'])
# for i in list(range(4))[::-1]:
#     print(i+1)
#     time.sleep(1)
# pyautogui.typewrite(['down','enter'])


def convert_to_one_hot(keys):
    #[Z,X,ZX,None]
    output=[0,0,0,0]
    if 'Z' in keys and 'X' in keys:
        output[2]=1
    elif 'Z' in keys:
        ouput[0]=1
    elif 'X' in keys:
        output[1]=1  
    else:
        output[3]=1     

    return output


file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists.Loading')
    training_data = list(np.load(file_name))
else:
    print('File does not exist')  
    training_data=[]  

def process_img(original_image):
    processed_img=cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img= cv2.resize(processed_img,(28,28))
    #rocess_img=cv2.Canny(processed_img, threshold1=200, threshold2=300)
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
    while(time_step<1000000):
        # 800x600 windowed mode
        #PressKey(determine_key(time_step))
        printscreen = grab_screen(region=(0,60,580,530))
        new_screen=process_img(printscreen)
        #print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        screen=process_img(printscreen)
        keys=key_check()
        output=convert_to_one_hot(keys)
        training_data.append([screen,output])
        #cv2.imshow('window2',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        cv2.imshow('window',new_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break     
        #ReleaseKey(determine_key(time_step))      
        time_step+=1     
#screen_record()


def screen_record2(): 
    last_time = time.time()
    time_step=0
    while(time_step<10000):
        # 800x600 windowed mode
        #PressKey(determine_key(time_step))
        printscreen =  np.array(ImageGrab.grab(bbox=(0,60,580,530)))
        screen=process_img(printscreen)
        last_time = time.time()
        #screen=process_img(printscreen)
        #keys=key_check()
        #output=convert_to_one_hot(keys)
        #training_data.append([screen,output])
        #print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window2',screen)
        # cv2.imshow('window',new_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break     
        #ReleaseKey(determine_key(time_step))      
        time_step+=1     
       # if len(training_data) % 500==0:
        #    print(len(training_data))
        #    np.save(file_name,training_data)
#screen_record()
reset()
pause()
screen_record()
#pause()


