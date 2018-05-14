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
import datetime
from lib.getkeys import key_check
from lib.reinforcement import Qnetwork,updateTarget,updateTargetGraph
from lib.SQL import SQLCalls
from sys import stdout
from keras import backend as K
from keras.models import load_model
from keras.models import Model,Sequential
import sqlite3
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
SQL=SQLCalls()
SQL.clear_permanent_tables()

def process_img(original_image):
    processed_img= cv2.resize(original_image,(580,580))
    return np.array(processed_img)


np.set_printoptions(precision=1)
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
#https://stackoverflow.com/a/15860757

def setup_genomes():
    BoxRadius=6
    BoxLength=BoxRadius*2+1
    BoxArea=(BoxLength)*(BoxLength)
    gene_image=np.empty([len(Genomes),BoxLength,BoxLength,12])
    gene_image.fill(0)
    BUTTON_AMOUNT=6
    for Genome_Num,Genome in enumerate(Genomes):
        for gene in Genome:
            genome_type=0
            #print(gene[0],Genome_Num,BoxArea*2)
            if gene[0]<BoxArea:
                pass
                #print("Normal Input")
            elif gene[0]>BoxArea*2:    
                #print("bias")
                continue
            else:
                pass
               # print("Inverse Input")
                genome_type+=BUTTON_AMOUNT
            genome_type+=int(gene[1]-1000001) 
            if genome_type>=0:
                # print ,Y,Type(Type of Input,Button Pressed)
                gene_image[Genome_Num][int(gene[0]%(BoxArea)//BoxLength)][int(gene[0]%(BoxArea)%13)][genome_type]=gene[2] 
    return gene_image

def update_progress(progress):
    barLength = 15 # Modify this to change the length of the progress bar
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
    text = str("\rPercent: [{0}] {1:02.0f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status))
    stdout.write(text)
    stdout.flush()
def do_action(SQL,frame_count):
    print_screen = np.array(ImageGrab.grab(bbox=(0,60,580,530)))
    new_screen=process_img(print_screen)
    image_duplicated=np.tile(new_screen, (POPULATION,1,1,1))
    #print(image_duplicated[0])
    #print(gene_images)
    history=model.predict([image_duplicated,gene_images],batch_size=16)
    #print(len(history))
    if epoch%2==0:
        max_prev=max(history[:,1])
        results=history[:,1]*UsedGenomes
    else:
        results=history[:,0]*UsedGenomes 

    frame_count+=1
    if frame_count<=POPULATION:
        a=np.argmax(results)
        #a=FakeGenomes.pop(0)
        UsedGenomes[a]=0
        print(a)
        update_progress(frame_count/POPULATION)
    else:
        a=1
    species,genome=SQL.convert_to_species_genome(a+1)
    SQL.update_image(new_screen)
    SQL.update_table(new_screen,int(a)+1,species,genome)
    
    return frame_count

#Pre Tenserflow Session Setup


epoch=0
frame_count=0
ACTION,WAIT,DEATH,GENERATION_OVER,RESTORE=0,1,2,3,4
print("Taking picture of the top-left of the screen.")
print("Please check image to ensure it only displays the emulator.")
img=ImageGrab.grab(bbox=(0,60,580,530))
img.save("../Test.png")

#Hyper Params
update_freq = 4 #How often to perform a training step.
y = .1 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 10000. #How many steps of training to reduce startE to endE.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.
#load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 1024 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network
img_size=84 #Size of the image.



while SQL.check_table()==WAIT:
    pass
# if SQL.check_table()==RESTORE:
#     load_model=True
Genomes=SQL.GatherGenomes()
gene_images=setup_genomes()
timeStamp=datetime.datetime.now().time()
POPULATION=len(Genomes)
print(POPULATION)

UsedGenomes=np.ones(Genomes.shape[0])
FakeGenomes=list(range(0,(Genomes.shape[0])))
random.shuffle(FakeGenomes)
#print("Load Model is " + str(load_model) )
print()
tf.reset_default_graph()
batch_size = POPULATION//4 #How many experiences to use for each training step.
#mainQN = Qnetwork(h_size,img_size,POPULATION,batch_size,"Main")
#targetQN = Qnetwork(h_size,img_size,POPULATION,batch_size,"Target")
#mainQN=FrozenValueNetwork()
#mainQN_model=mainQN.make_model()
model = load_model('dqn_frozen_modelv4.h5')
#init = tf.global_variables_initializer()

#saver = tf.train.Saver()

#trainables = tf.trainable_variables()

#targetOps = updateTargetGraph(trainables,tau)


#e = startE
#stepDrop = (startE - endE)/anneling_steps
#total_steps = 0

print("Ready!")
print()

if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    pass
    #Vanilla Policy Setup
    # sess.run(init)
    # if load_model == True:
    #     print('Loading Model...')
    #     ckpt = tf.train.get_checkpoint_state(path)
    #     saver.restore(sess,ckpt.model_checkpoint_path)
    #     epoch=int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
print()
#Infinite Loop For Actions
while True:
    check=SQL.check_table()
    if check==ACTION or check==RESTORE: #Mario Needs an Action
        frame_count=do_action(SQL,frame_count)
    elif check==DEATH or check==GENERATION_OVER: # Mario has Died
        print("Epoch " + str(epoch) + " Complete")
        #timeStamp=datetime.datetime.now().time()
        trainBatch=SQL.gain_history()
        FakeGenomes=list(range(0,(Genomes.shape[0])))
        random.shuffle(FakeGenomes)
        frame_count=0
        epoch+=1
        if check==GENERATION_OVER:
            SQL.insert_into_permanent_tables(gene_images,trainBatch,timeStamp,True)
            Genomes=SQL.GatherGenomes()
            gene_images=setup_genomes()
        else:
            SQL.insert_into_permanent_tables(gene_images,trainBatch,timeStamp,False)
            timeStamp=datetime.datetime.now().time()
        SQL.clear_table()
        SQL.clear_extra_genomes()
        frame_count=do_action(SQL,frame_count)
        UsedGenomes=np.ones(Genomes.shape[0])
            # epoch+=1
            # #Update final one
            # #print_screen = np.array(ImageGrab.grab(bbox=(0,60,580,530)))
            # #new_screen=np.array(np.reshape(process_img(print_screen),[84,84,3]))
            # #SQL.update_image(new_screen)
            # trainBatch=SQL.gain_history()
            # np.random.shuffle(trainBatch)
            # trainBatch=trainBatch[0:batch_size]
            # states=np.vstack(trainBatch[:,0])
            # states=np.reshape(states,[batch_size,84,84,3])
            # states_after=np.vstack(trainBatch[:,3])
            # states_after=np.reshape(states,[batch_size,84,84,3])
            # UsedGenomes=np.zeros(Genomes.shape[0])
            # FakeGenomes=list(range(0,(Genomes.shape[0])))
            # random.shuffle(FakeGenomes)
            # action_list=[]
            # answer_list=[]
            # for k in range(batch_size):
            #     m_dict={mainQN.used_genomes:UsedGenomes,mainQN.genomes:Genomes,
            #             mainQN.imageIn:[states_after[k]],mainQN.condition:0,
            #             mainQN.correct_action:[10],mainQN.correct_mean:[10]}
            #     t_dict={targetQN.used_genomes:UsedGenomes,targetQN.genomes:Genomes,
            #             targetQN.imageIn:[states_after[k]],targetQN.condition:0,
            #             targetQN.correct_action:[10],targetQN.correct_mean:[10]}
            #     Q1 = sess.run(mainQN.predict,feed_dict=m_dict)
            #     action_list.append(Q1)
            #     mean,Q2,Value = sess.run([targetQN.Mean,targetQN.Qout,targetQN.Value],feed_dict=t_dict)
            #     answer_list.append(Q2[0][Q1])
            # #print(answer_list)
            # print(trainBatch[:,1])    
            # UsedGenomesBatch=np.zeros(batch_size)
            # end_multiplier = -(trainBatch[:,4] - 1)
            # doubleQ = np.array(answer_list)
            # targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
            # mean_list=[]
            # #print(Genomes[0])
            # #Update the network with our target values.
            # for k in range(batch_size):
            #     m_dict={mainQN.used_genomes:UsedGenomes,mainQN.genomes:Genomes,
            #             mainQN.imageIn:[states[k]],mainQN.condition:0,
            #             mainQN.correct_action:[10],mainQN.correct_mean:[10]}
            #     mean = sess.run([mainQN.Mean],feed_dict=m_dict)
            #     mean_list.append(mean[0])
            # final_dict={mainQN.used_genomes:UsedGenomes,mainQN.genomes:Genomes,
            #             mainQN.imageIn:states,mainQN.condition:1,
            #             mainQN.correct_action:trainBatch[:,1],mainQN.correct_mean:np.hstack(mean_list),
            #            mainQN.targetQ:targetQ}
            # _,loss,Qs =sess.run([mainQN.updateModel,mainQN.loss,mainQN.Qout], \
            #     feed_dict=final_dict)
            # updateTarget(targetOps,sess)
            # print("Epoch " + str(epoch) + " Complete")
            # epoch+=1
            # print("Loss "+str(loss))
            # print()
            # if epoch%3==0: 
            #     saver.save(sess,path+'/model-'+str(epoch)+'.ckpt',global_step=epoch)
            #     print("Saved Model")
            #     print()
            # break
            # SQL.clear_table()
            # frame_count=0
            # if check==GENERATION_OVER:
            #     Genomes=SQL.GatherGenomes()
            # frame_count=do_action(SQL,frame_count)
SQL.exit()
#reset()
#pause()
#screen_record()
#pause()
