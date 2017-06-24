import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from directkeys import *
from getkeys import key_check
import os


Gather=True
def convert_to_one_hot_buttons(keys):
    #[Z,X,ZX,None]
    output=[0,0,0,0]
    if 'Z' in keys and 'X' in keys:
        output[2]=1
    elif 'Z' in keys:
        output[0]=1
    elif 'X' in keys:
        output[1]=1  
    else:
        output[3]=1     

    return output

def convert_to_one_hot_arrows(keys):
    #[left,right,up,down,none]
    output=[0,0,0]
    if 'J'  in keys:
        output[0]=1
    elif 'L' in keys:
        output[1]=1 
    else:
        output[2]=1     

    return output    
# pyautogui.typewrite(['down','enter'])
# for i in list(range(4))[::-1]:
#     print(i+1)
#     time.sleep(1)
# pyautogui.typewrite(['down','enter'])

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
if Gather:
    file_name = 'training_data.npy'
    file_name2 = 'training_data2.npy'
    if os.path.isfile(file_name) and os.path.isfile(file_name2) and False:
        print('File exists.Loading')
        training_data = list(np.load(file_name))
        training_data2=list(np.load(file_name2))
    else:
        print('File does not exist')  
        training_data=[]  
        training_data2=[]

def screen_record(): 
    last_time = time.time()
    time_step=0
    while(True):
        # 800x600 windowed mode
        #PressKey(determine_key(time_step))
        printscreen =  np.array(ImageGrab.grab(bbox=(0,60,580,530)))
        new_screen=process_img(printscreen)
        keys=key_check()
        if Gather:
            output=convert_to_one_hot_buttons(keys)
            training_data.append([new_screen,output])
            output2=convert_to_one_hot_arrows(keys)
            training_data2.append([new_screen,output2])
        #print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        #cv2.imshow('window2',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        cv2.imshow('window',new_screen)
        if 'Q' in keys:
            cv2.destroyAllWindows()
            break     
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break         
        #ReleaseKey(determine_key(time_step)) 
        if Gather and len(training_data) % 500==0:
            print(len(training_data))
            np.save(file_name,training_data)  
            np.save(file_name2,training_data2)   
        time_step+=1  

#screen_record()
reset()
pause()
screen_record()
pause()


