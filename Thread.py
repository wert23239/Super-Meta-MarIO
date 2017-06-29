import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
try:
    xrange = xrange
except:
    xrange = range
from PIL import ImageGrab
import cv2
import time
import math
import random
# import pyautogui
#from directkeys import *
from lib.getkeys import key_check
from lib.reinforcement import agent,discount_rewards
from lib.create_table import SQLCalls
# import os
# from alexnet import alexnet
import sqlite3

SQL=SQLCalls()
WIDTH=28
HEIGHT=28
LR= 1e-3
EPOCH=8
MODEL_NAME= 'MarioAI-{}-{}-{}-epochs.model'.format(LR,'alexnetv2',EPOCH)
def process_img(original_image):
    processed_img=cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img= cv2.resize(processed_img,(28,28))
    #rocess_img=cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img



def gain_history():
    sql = '''Select image,action,score
    from rewards where done=1 and score is not NULL'''
    cur.execute(sql)
    x=cur.fetchall()
    return np.array(x)








#Pre Tenserflow Session Setup
tf.reset_default_graph()
myAgent = agent(lr=1e-2,s_size=28*28,a_size=4,h_size=8)
init = tf.global_variables_initializer()
update_frequency = 5


frame_count=0
ACTION,WAIT,DEATH=0,1,2


while SQL.check_table()==WAIT:
    pass
print("Ready!")   

def do_action(SQL,frame_count):
    print_screen = np.array(ImageGrab.grab(bbox=(0,60,580,530)))
    new_screen=np.array(np.reshape(process_img(print_screen),[-1]))
    #Gain Action from Tenserflow
    a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[new_screen]})
    a = np.random.choice(a_dist[0],p=a_dist[0])
    a = np.argmax(a_dist == a)
    rand_num=random.randint(0,10)
    if rand_num>1:
        a=5
    SQL.update_table(new_screen,int(a+1))   
    frame_count+=1
    return frame_count

with tf.Session() as sess: 

    #Vanilla Policy Setup
    sess.run(init)
    i = 0
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    #Infinite Loop For Actions
    while True:
        i+1
        keys=key_check()
        check=SQL.check_table()
        if 'Q' in keys:
            break   
        if check==ACTION: #Mario Needs an Action
            frame_count=do_action(SQL,frame_count)
        elif check==DEATH: #Mario has Died
            print("DEATH")
            ep_history =SQL.gain_history()
            ep_history[:,2] = discount_rewards(ep_history[:,2])
            feed_dict={myAgent.reward_holder:ep_history[:,2],
                    myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
            grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
            for idx,grad in enumerate(grads):
                gradBuffer[idx] += grad

            if i % update_frequency == 0 and i != 0:
                feed_dict= dict(zip(myAgent.gradient_holders, gradBuffer))
                _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
            SQL.clear_table()
            frame_count=0
            print("Round Complete")
            frame_count=do_action(SQL,frame_count)
            
             
SQL.exit()
#reset()
#pause()
#screen_record()
#pause()


