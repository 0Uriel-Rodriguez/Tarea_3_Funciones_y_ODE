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

class ODE(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss") 
        
    @property
    def metrics(self):
        return [self.loss_tracker] 
    
    
    def train_step(self, data):
        batch_size = 100 
        
        x = tf.random.uniform((batch_size,1), minval=-5, maxval=5)
        
        
        with tf.GradientTape() as tape:
            
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y_pred = self(x, training=True)
                dy =tape2.gradient(y_pred, x)
                x_o = tf.zeros((batch_size,1))
                y_o = self(x_o, training=True)
                eq = x*dy + y_pred - (x**2)*(tf.math.cos(x))
                ic = y_o + 0.
                loss = keras.losses.mean_squared_error(0.,eq) + keras.losses.mean_squared_error(0., ic)
                
        #aplica los gradientes        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        
        #actualiza metricas
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}
    
    


# In[2]:


model = ODE()

model.add(Dense(10,activation ='tanh', input_shape=(1,)))
model.add(Dense(50, activation ='tanh'))
model.add(Dense(100, activation ='tanh'))
model.add(Dense(50, activation ='tanh'))
model.add(Dense(10, activation ='tanh'))
model.add(Dense(1, activation ='tanh'))
model.add(Dense(1, activation ='linear'))



model.summary()

model.compile(optimizer=RMSprop(), metrics=['loss'])

x=tf.linspace(-5,5,100)
history = model.fit(x,epochs=3000,verbose=1)

x_testv = tf.linspace(-5,5,100)
a=model.predict(x_testv)




# In[3]:


plt.figure(figsize=(10,6))
plt.plot(x_testv,a)
plt.plot( x_testv,x*np.sin(x)+2*np.cos(x)-2*np.sin(x)/x)
legend = ['Sol. Numeric','Sol. Analitycal']
plt.legend(loc='upper left', labels= legend)
plt.show()
model.save("T3_RNA_EDO1.h5") 

