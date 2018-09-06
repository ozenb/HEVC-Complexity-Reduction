
# coding: utf-8

# # Keras CNN v1

# ## Import Packages

# In[14]:


import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input
from keras.utils import plot_model
import h5py


# ## Preprocess and Variables

# In[15]:


input_shape = (64,64,1)
image16 = Input(shape = (16,16,1))
image32 = Input(shape = (32,32,1))
image64 = Input(shape = (64,64,1))
qp = Input(shape = (1,),dtype='float32')


# ## Generate Model

# ### Feature Selection and Branches

# In[16]:


#Downsampling and mean removal is ignored

#Branch 1
branch_1 = Conv2D(16,(4,4),strides=(4,4))(image16)
branch_1_2 = Conv2D(24,(2,2),strides=(2,2))(branch_1)
branch_1_3 = Conv2D(32,(2,2),strides=(2,2))(branch_1_2)

#Branch 2
branch_2 = Conv2D(16,(4,4),strides=(4,4))(image32)
branch_2_2 = Conv2D(24,(2,2),strides=(2,2))(branch_2)
branch_2_3 = Conv2D(32,(2,2),strides=(2,2))(branch_2_2)

#Branch 3
branch_3 = Conv2D(16,(4,4),strides=(4,4))(image64)
branch_3_2 = Conv2D(24,(2,2),strides=(2,2))(branch_3)
branch_3_3 = Conv2D(32,(2,2),strides=(2,2))(branch_3_2)


# ### Flatten Layer

# In[17]:


#Branch 1
branch_1_2_f = Flatten()(branch_1_2)
branch_1_3_f = Flatten()(branch_1_3)
# Branch 2
branch_2_2_f = Flatten()(branch_2_2)
branch_2_3_f = Flatten()(branch_2_3)

#Branch 3
branch_3_2_f = Flatten()(branch_3_2)
branch_3_3_f = Flatten()(branch_3_3)


flatten = keras.layers.Concatenate()([branch_1_2_f, branch_1_3_f, branch_2_2_f, branch_2_3_f, branch_3_2_f, branch_3_3_f])


# ### 3 Level FC Layers
#

# In[18]:


#Level 1
level_1_1 = Dense(64,activation = 'relu')(flatten)
#Add QP
level_1_1 = keras.layers.Concatenate()([level_1_1, qp])
level_1_2 = Dense(48,activation = 'relu')(level_1_1)
level_1_2 = keras.layers.Concatenate()([level_1_2, qp])

#Level 2
level_2_1 = Dense(128,activation='relu')(flatten)
#Add QP
level_2_1 = keras.layers.Concatenate()([level_2_1, qp])
level_2_2 = Dense(96,activation='relu')(level_2_1)
level_2_2 = keras.layers.Concatenate()([level_2_2, qp])

#Level 3
level_3_1 = Dense(256,activation='relu')(flatten)
#Add QP
level_3_1 = keras.layers.Concatenate()([level_3_1, qp])
level_3_2 = Dense(192,activation='relu')(level_3_1)
level_3_2 = keras.layers.Concatenate()([level_3_2, qp])


# ### Output Layers

# In[19]:


y_1 = Dense(1,activation='sigmoid')(level_1_2)

y_2 = Dense(4,activation='sigmoid')(level_2_2)

y_3 = Dense(8,activation='sigmoid')(level_3_2)


# ### Initialize the Model

# In[20]:


MODEL = Model(inputs=[image16, image32, image64, qp], outputs = [y_1, y_2, y_3])


# ### Summary and Visualization

# In[21]:


#model.summary()


# In[22]:


#plot_model(model, to_file='model.png')


# In[10]:


#model.compile(loss=keras.losses.mean_squared_error,
#            optimizer = 'adam',
#            metrics=['accuracy'])


# In[12]:


#model.save('cnnv1.h5')




def __getitem__(self,index):
    self.f = open(path,'rb')

    #Condition for CU 64x64
    if self.cu == 64:
        #1 sample is enough for 1 input
        offset = (1+ self.cu*self.cu)*index
        self.f.seek(offset,0)

        #Start Reading Bytes

        for self.i in range(self.batch_size):
            #Label
            self.label1[self.i,0] = self.f.read(1)[0]
            #Input
            self.tmp = self.f.read(self.cu * self.cu)

            for self.k in range(self.cu * self.cu):
                #Assign input and decode
                self.input[self.i, self.k] = self.tmp[self.k]
            #Preprocessing can be done here!

            #Mean Removal
            self.mean = np.mean(self.input[self.i,:])
            self.input[self.i,:] = self.input[self.i,:] - self.mean

            #Downsampling 64x64 -> 16x16
            self.down16[self.i,:] = self.input[self.i,::4,::4]
        #Batch size end

        self.down16 = self.down16.reshape((self.batch_size,16,16))
        self.f.close()

        return np.array[self.down16, None, None], self.label1

    elif self.cu == 32:
        # 4 sample is needed for 1 input -> 2x2 32x32 sample
        self.input = self.input.reshape((self.batch_size,64,64))
        offset = (1+ 32*32)*index
        self.f.seek(offset,0)

        #Start Reading Bytes
        self.counter = 0

        for self.i in range(self.batch_size):
            for self.counter in range(4):
                #Label
                self.label2[self.i,self.counter] = self.f.read(1)[0]
                #input
                self.tmp = self.f.read(32*32)
                for self.k in range(32*32):
                    #Assign input and decode
                    self.store32[self.i,self.counter,self.k]  = self.tmp[self.k]
                #Preprocessing can be done here!

                #Mean removal

                self.mean = np.mean(self.store32[self.i,self.counter,:])
                self.store32[self.i,self.counter,:] = self.store32[self.i,self.counter,:] - self.mean
            #4 samples are obtained
            #Concat 4 samples
            self.input[self.i,:] =np.vstack(( np.hstack((self.store32[self.i,0,:].reshape((self.batch_size,32,32)),self.store32[self.i,1,:].reshape((self.batch_size,32,32)))), np.hstack((self.store32[self.i,2,:].reshape((self.batch_size,32,32)), self.store32[self.i,3,:].reshape((self.batch_size,32,32)))) ))
        #Batching is done!


        #Downsampling
        self.down32 = self.input[:,::2,::2]

        return np.array[None, self.down32, None], self.label2
