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
from sys import stdout
import sqlite3
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
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
                                                             ,myAgent.used_genomes:UsedGenomes\
                                                                 ,myAgent.action_holder:[10]})
    a = np.random.choice(a_dist,p=a_dist)
    a = np.argmax([a_dist] == a)
    UsedGenomes[a]=-np.inf
    #print("update "+ str(frame_count))
    species,genome=SQL.convert_to_species_genome(a+1)
    SQL.update_table(new_screen,int(a),species,genome)
    #print("update completed")
    frame_count+=1
    return frame_count

#Pre Tenserflow Session Setup


epoch=0
frame_count=0
ACTION,WAIT,DEATH,GENERATION_OVER=0,1,2,3
img=ImageGrab.grab(bbox=(0,60,580,530))
img.save("Test.png")
while SQL.check_table()==WAIT:
    pass
print("Ready!")
Genomes=SQL.GatherGenomes()
POPULATION=len(Genomes)
print(POPULATION)
UsedGenomes=np.ones(Genomes.shape[0])

tf.reset_default_graph()
myAgent = agent(lr=1e-2,s_size=28,a_size=4,h_size=64,pop_size=POPULATION)
init = tf.global_variables_initializer()
update_frequency = 5


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
        elif check==DEATH or check==GENERATION_OVER: # Mario has Died
            #print("DEATH")
            ep_history =SQL.gain_history()
            #ep_history[:,2] = discount_rewards(ep_history[:,2])
            states=np.vstack(ep_history[:,0])
            states=np.reshape(states,[POPULATION,28,28,1])
            UsedGenomes=np.ones(Genomes.shape[0])
            loss_total=0
            for k in range(POPULATION):
                feed_dict={myAgent.reward_holder:[ep_history[:,2][k]],myAgent.action_holder:[ep_history[:,1][k]],
                    myAgent.state_in:[states[k]],myAgent.used_genomes:UsedGenomes,myAgent.genomes:Genomes}
                grads,loss,ro = sess.run([myAgent.gradients,myAgent.loss,myAgent.responsible_output], feed_dict=feed_dict)
                loss_total+=loss
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad
            loss_total=loss_total/POPULATION
            print("Epoch " + str(epoch) + " Complete")
            epoch+=1
            print("Loss "+str(loss_total))
            print()
            if i % update_frequency == 0 and i != 0:
                feed_dict= dict(zip(myAgent.gradient_holders, gradBuffer))
                _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0      
            SQL.clear_table()
            frame_count=0
            if check==GENERATION_OVER:
                Genomes=SQL.GatherGenomes()
            frame_count=do_action(SQL,frame_count)


SQL.exit()
#reset()
#pause()
#screen_record()
#pause()
