#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np

class ODEsol(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss") #loss_tracker pude ser cambiada por mivalor_de_costo o como queramos
        
        
    @property
    def metrics(self):
        return [self.loss_tracker] #igual cambia el loss_tracker
    
    
    def train_step(self, data):
        batch_size = tf.shape(data)[0]
     
        x = tf.random.uniform((batch_size,1), minval=-5, maxval=5)
        x_o = tf.zeros((batch_size,1))
        
        
        with tf.GradientTape() as tape:
            
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x)
                
                with tf.GradientTape(persistent=True) as tape3:
                    tape3.watch(x)
                    tape3.watch(x_o)
                    y_pred = self(x, training=True)
                    y_o = self(x_o, training=True)
                        
                dy_dx = tape3.gradient(y_pred, x)
                dy_dxo = tape3.gradient(y_o, x_o)
                
            dy2_dx2 = tape2.gradient(dy_dx, x)
            
            eq = dy2_dx2 + y_pred
            ic = y_o -1. 
            ic2 = dy_dxo + 0.5
            loss = keras.losses.mean_squared_error(0.,eq) + keras.losses.mean_squared_error(0., ic)+ keras.losses.mean_squared_error(0.,ic2)
                
            
            
                
        #aplica los gradientes        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        
        #actualiza metricas
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}
    
    


# In[2]:


model = ODEsol()

model.add(Dense(10,activation ='tanh', input_shape=(1,)))
model.add(Dense(50, activation ='tanh'))
model.add(Dense(100, activation ='tanh'))
#model.add(Dense(150, activation ='tanh'))
#model.add(Dropout(0.2))
#model.add(Dense(60, activation ='tanh'))
model.add(Dense(50, activation ='tanh'))
model.add(Dense(10, activation ='tanh'))
model.add(Dense(1, activation ='tanh'))
model.add(Dense(1, activation ='linear'))


model.summary()

model.compile(optimizer=RMSprop(), metrics=['loss'])

x=tf.linspace(-5,5,100)
history = model.fit(x,epochs=1000,verbose=1)

x_testv = tf.linspace(-5,5,100)
a=model.predict(x_testv)





# In[3]:


plt.figure(figsize=(10,6))
plt.plot(x_testv,a)
plt.plot( x_testv,np.cos(x)-(0.5)*np.sin(x))
legend = ['Sol. Numeric','Sol. Analitycal']
plt.legend(loc='upper left', labels= legend)
plt.show()
model.save("T3_RNA_EDO2.h5") 


# In[ ]:




