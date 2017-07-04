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
from lib.getkeys import key_check
from lib.reinforcement import agent,discount_rewards
from lib.SQL import SQLCalls
import sqlite3

SQL=SQLCalls()


def process_img(original_image):
    processed_img=cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img= cv2.resize(processed_img,(28,28))
    return processed_img

def do_action(SQL,frame_count):
    print_screen = np.array(ImageGrab.grab(bbox=(0,60,580,530)))
    new_screen=np.array(np.reshape(process_img(print_screen),[28,28,1]))
    #Gain Action from Tenserflow
    a_dist = sess.run(myAgent.final_output,feed_dict={myAgent.genomes:Genomes,myAgent.state_in:[new_screen]\
                                                             ,myAgent.used_genomes:UsedGenomes,myAgent.condition:0\
                                                                 ,myAgent.action_holder:[10]})
    a = np.random.choice(a_dist,p=a_dist)
    a = np.argmax([a_dist] == a)

    print("update "+ str(frame_count))
    SQL.update_table(new_screen,int(a+1)) 
    print("update completed")
    frame_count+=1
    return frame_count


#Pre Tenserflow Session Setup
tf.reset_default_graph()
myAgent = agent(lr=1e-2,s_size=28,a_size=4,h_size=64)
init = tf.global_variables_initializer()
update_frequency = 5


frame_count=0
ACTION,WAIT,DEATH=0,1,2


while SQL.check_table()==WAIT:
    pass
print("Ready!")   
Genomes=SQL.GatherGenomes()
UsedGenomes=np.ones(Genomes.shape[0])

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
            break
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


