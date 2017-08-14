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
from lib.reinforcement import Qnetwork,updateTarget,updateTargetGraph
from lib.SQL import SQLCalls
from sys import stdout
import sqlite3
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
SQL=SQLCalls()
def process_img(original_image):
    processed_img= cv2.resize(original_image,(84,84))
    return np.array(np.reshape(processed_img,[84,84,3]))

def do_action(SQL,frame_count):
    print_screen = np.array(ImageGrab.grab(bbox=(0,60,580,530)))
    new_screen=np.array(np.reshape(process_img(print_screen),[84,84,3]))
    f_dict={mainQN.used_genomes:UsedGenomes,mainQN.genomes:Genomes,mainQN.imageIn:[new_screen]}
    #Gain Action from Tenserflow
    a,before = sess.run([mainQN.predict,mainQN.Smooth],feed_dict=f_dict)
    #a = np.random.choice(a_dist,p=a_dist)
    #a = np.argmax([a_dist] == a)
    print(a)
    UsedGenomes[a]=1000
    #print("update "+ str(frame_count))
    species,genome=SQL.convert_to_species_genome(a+1)
    SQL.update_image(new_screen)
    SQL.update_table(new_screen,int(a),species,genome)
    #print("update completed")
    #Does one extra
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
UsedGenomes=np.zeros(Genomes.shape[0])

tf.reset_default_graph()
#Hyper Params
batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 10000. #How many steps of training to reduce startE to endE.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 1024 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network
img_size=84 #Size of the image.


tf.reset_default_graph()
mainQN = Qnetwork(h_size,img_size,POPULATION,"Main")
targetQN = Qnetwork(h_size,img_size,POPULATION,"Target")

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)


e = startE
stepDrop = (startE - endE)/anneling_steps
total_steps = 0


if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:

    #Vanilla Policy Setup
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)

    #Infinite Loop For Actions
    while True:
        keys=key_check()
        check=SQL.check_table()
        if 'Q' in keys:
            break
        if check==ACTION: #Mario Needs an Action
            frame_count=do_action(SQL,frame_count)
        elif check==DEATH or check==GENERATION_OVER: # Mario has Died
            #Update final one
            print("here")
            #print_screen = np.array(ImageGrab.grab(bbox=(0,60,580,530)))
            #new_screen=np.array(np.reshape(process_img(print_screen),[84,84,3]))
            #SQL.update_image(new_screen)
            ep_history =numpy.shuffle(SQL.gain_history())
            break
            #ep_history[:,2] = discount_rewards(ep_history[:,2])
            states=np.vstack(ep_history[:,0])
            states=np.reshape(states,[POPULATION,84,84,3])
            states_after=np.vstack(ep_history[:,3])
            states_after=np.reshape(states,[POPULATION,84,84,3])
            UsedGenomes=np.zeros(Genomes.shape[0])
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
            break
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
