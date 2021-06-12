# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:19:25 2021

@author: Hubert Kyeremateng-Boateng
"""
from distribution import fit_distribution
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder, Normalizer

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, InputLayer,  BatchNormalization
from keras.models import Sequential

# read data
train_data=pd.read_csv('nsl-kdd/KDDTrain+.txt', sep = ',', error_bad_lines=False, header=None)
#test_data=pd.read_csv('nsl-kdd/KDDTest+.txt', sep = ',', error_bad_lines=False, header=None)
encoder = LabelEncoder()
train_data[1]= encoder.fit_transform(train_data.iloc[:,1])
train_data[2]= encoder.fit_transform(train_data.iloc[:,2])
train_data[3]= encoder.fit_transform(train_data.iloc[:,3])
train_ds = train_data.iloc[:,:-2] #Remove the last two columns from train ds

train_labels = train_data.iloc[:,41]
encoded_labels = encoder.fit_transform(train_labels)

#Scale and Normalizer
normalizer = Normalizer()
train_X = normalizer.fit_transform(train_ds)

class DNNIntrusion(Model):
    
    def __init__(self, data, name):
        super(DNNIntrusion,self).__init__(name=name)
        self.data = data
        #self.inputLayer = InputLayer(input_shape=(125973,))
        self.dense1 = Dense(1024, name="dense1")
        self.dense2 = Dense(1024,  activation="relu", name="dense2")
        self.dense3 = Dense(512, activation="relu", name="dense3")
        self.dense4 = Dense(256, activation="relu", name="dense4")
        self.dense5 = Dense(23, activation="softmax", name="dense5")
        self.relu = tf.keras.layers.ReLU()
        self.batch = BatchNormalization()
        
    def call(self, x):
        #x = self.inputLayer(x)
        x = self.dense1(x)
        
        x = self.relu(x)
        x = self.batch(x)
        x = self.dense2(x)
        x = self.batch(x)
        x = self.dense2(x)
        x = self.batch(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)

        return self.dense5(x)

epochs = 10
#Split data

x_train = tf.convert_to_tensor(train_X)
y_train = np.reshape(encoded_labels,(-1,1))
y_train = tf.convert_to_tensor(y_train)

cnn_model = DNNIntrusion(x_train,name="intrusion")

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

## Metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

layer = []
def print_layer_variables(model):
    tes = model.trainable_variables[6]
    #print(model.trainable_variables)
    try:
        print(tes,"test")
        results = fit_distribution(tes[0].numpy(),0.99,0.01)

        print(results.iloc[0]['chi_square'], results.iloc[0]['Distribution'])
        results = fit_distribution(tes[1].numpy(),0.99,0.01)
        print(results.iloc[0]['chi_square'], results.iloc[0]['Distribution'])
    except NotImplementedError:
        pass
    except ValueError:
        pass
    except AttributeError:
        pass



@tf.function
def train_step(trainDS, labels,training=True):
  
  with tf.GradientTape() as tape:

    predictions = cnn_model(trainDS)
    loss = loss_object(labels, predictions)
  
  gradients = tape.gradient(loss, cnn_model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, cnn_model.trainable_variables))
  
  train_loss(loss)
  train_accuracy(labels, predictions)
  print_layer_variables(cnn_model)
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)
  
  
EPOCHS = 1


for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  
  #fit_distribution(X_train,0.99,0.01)
  train_step(x_train, y_train)
  # with train_summary_writer.as_default():
  #   tf.summary.scalar('loss', train_loss.result(), step=epoch)
  #   tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
  ''' To be Implemented'''
  #for test_images, test_labels in test_ds:
  #  test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}: '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
  )
  
  print_layer_variables(cnn_model)
  
class RNNIntrusion(Model):
    def __init__(self, data):
        super(RNNIntrusion, self).__init__(name="RNN Instrusion")
        #self.dense = 
        pass
    
    def call(self, d):
        pass#d = 