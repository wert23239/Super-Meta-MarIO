#train_model.py
import numpy as np
from alexnet import alexnet

WIDTH=28
HEIGHT=28
LR= 1e-3
EPOCH=8
MODEL_NAME= 'MarioAI-{}-{}-{}-epochs.model'.format(LR,'alexnetv2',EPOCH)

model= alexnet(WIDTH,HEIGHT,LR)

train_data=np.load('training_data_v2.npy')

train= train_data[:800*3]
test=train_data[800*3:]
print(len(train),len(test))
X= np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y= [i[1] for i in train]
X_test= np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
Y_test= [i[1] for i in test]

model.fit({'input':X},{'targets': Y},n_epoch=EPOCH,
validation_set=({'input':X_test},{'targets': Y_test}),snapshot_step=500,
show_metric=True, run_id=MODEL_NAME)

# tensorboard --logdir=foo:C:/Users/lambe/OneDrive - Michigan State University/Source/MarIO2ProtoType/log

model.save(MODEL_NAME)