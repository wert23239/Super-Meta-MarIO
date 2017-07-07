import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
train_data = np.load('training_data2.npy')

 
df=pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

# Z=[]
# ZX=[]
# Non=[]

# shuffle(train_data)

# for data in train_data:
#     img= data[0]
#     choice =data[1]

#     if choice==[1,0,0,0]:
#         Z.append([img,choice])
#     elif choice==[0,0,1,0]:
#         ZX.append([img,choice])
#     elif choice==[0,0,0,1]:
#         Non.append([img,choice])
# Z=Z[:960] 
# ZX=ZX[:960]
# Non=Non[:960]       

# final_data=Z+ZX+Non

# shuffle(final_data)

# print(len(final_data))

# np.save('training_data_2_v2.npy',final_data)