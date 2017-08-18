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


# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
#https://stackoverflow.com/a/15860757
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = print("\rPercent: [{0}] {1}% {2}").format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def do_action(SQL,frame_count):
    print_screen = np.array(ImageGrab.grab(bbox=(0,60,580,530)))
    new_screen=np.array(np.reshape(process_img(print_screen),[84,84,3]))
    f_dict={mainQN.used_genomes:UsedGenomes,mainQN.genomes:Genomes,\
    mainQN.imageIn:[new_screen],mainQN.condition:0,mainQN.correct_action:[10],mainQN.correct_mean:[10]}
    #Gain Action from Tenserflow
    a,before = sess.run([mainQN.predict,mainQN.Smooth],feed_dict=f_dict)
    #a = np.random.choice(a_dist,p=a_dist)
    #a = np.argmax([a_dist] == a)
    #print(a)
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
ACTION,WAIT,DEATH,GENERATION_OVER,RESTORE=0,1,2,3,4
img=ImageGrab.grab(bbox=(0,60,580,530))
img.save("Test.png")

#Hyper Params
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



while SQL.check_table()==WAIT:
    pass
if SQL.check_table()==RESTORE:
    load_model=True
Genomes=SQL.GatherGenomes()
POPULATION=len(Genomes)
print(POPULATION)
UsedGenomes=np.zeros(Genomes.shape[0])
print("Load Model is " + str(load_model) )
tf.reset_default_graph()
batch_size = POPULATION//4 #How many experiences to use for each training step.

mainQN = Qnetwork(h_size,img_size,POPULATION,batch_size,"Main")
targetQN = Qnetwork(h_size,img_size,POPULATION,batch_size,"Target")

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)


e = startE
stepDrop = (startE - endE)/anneling_steps
total_steps = 0

print("Ready!")


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
        check=SQL.check_table()
        if check==ACTION or check==RESTORE: #Mario Needs an Action
            frame_count=do_action(SQL,frame_count)
        elif check==DEATH or check==GENERATION_OVER: # Mario has Died
            #Update final one
            #print_screen = np.array(ImageGrab.grab(bbox=(0,60,580,530)))
            #new_screen=np.array(np.reshape(process_img(print_screen),[84,84,3]))
            #SQL.update_image(new_screen)
            trainBatch=SQL.gain_history()
            np.random.shuffle(trainBatch)
            trainBatch=trainBatch[0:batch_size]
            states=np.vstack(trainBatch[:,0])
            states=np.reshape(states,[batch_size,84,84,3])
            states_after=np.vstack(trainBatch[:,3])
            states_after=np.reshape(states,[batch_size,84,84,3])
            UsedGenomes=np.zeros(Genomes.shape[0])
            #ep_history[:,2] = discount_rewards(ep_history[:,2])
            action_list=[]
            answer_list=[]
            for k in range(batch_size):
                m_dict={mainQN.used_genomes:UsedGenomes,mainQN.genomes:Genomes,
                        mainQN.imageIn:[states_after[k]],mainQN.condition:0,
                        mainQN.correct_action:[10],mainQN.correct_mean:[10]}
                t_dict={targetQN.used_genomes:UsedGenomes,targetQN.genomes:Genomes,
                        targetQN.imageIn:[states_after[k]],targetQN.condition:0,
                        targetQN.correct_action:[10],targetQN.correct_mean:[10]}
                Q1 = sess.run(mainQN.predict,feed_dict=m_dict)
                action_list.append(Q1)
                mean,Q2,Value = sess.run([targetQN.Mean,targetQN.Qout,targetQN.Value],feed_dict=t_dict)
                answer_list.append(Q2[0][Q1])
            UsedGenomesBatch=np.zeros(batch_size)
            end_multiplier = -(trainBatch[:,4] - 1)
            doubleQ = np.array(answer_list)
            targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
            mean_list=[]
            #Update the network with our target values.
            for k in range(batch_size):
                m_dict={mainQN.used_genomes:UsedGenomes,mainQN.genomes:Genomes,
                        mainQN.imageIn:[states[k]],mainQN.condition:0,
                        mainQN.correct_action:[10],mainQN.correct_mean:[10]}
                mean = sess.run([mainQN.Mean],feed_dict=m_dict)
                mean_list.append(mean[0])
            final_dict={mainQN.used_genomes:UsedGenomes,mainQN.genomes:Genomes,
                        mainQN.imageIn:states,mainQN.condition:1,
                        mainQN.correct_action:trainBatch[:,1],mainQN.correct_mean:np.hstack(mean_list),
                       mainQN.targetQ:targetQ}
            _,loss =sess.run([mainQN.updateModel,mainQN.loss], \
                feed_dict=final_dict)
            updateTarget(targetOps,sess)
            print("Epoch " + str(epoch) + " Complete")
            epoch+=1
            print("Loss "+str(loss))
            print()
            if epoch%3==0: 
                saver.save(sess,path+'/model-'+str(epoch)+'.ckpt')
                print("Saved Model")
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
