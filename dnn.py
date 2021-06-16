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
from sklearn.model_selection import train_test_split

# read data
train_data=pd.read_csv('nsl-kdd/KDDTrain+.txt', sep = ',', error_bad_lines=False, header=None)
test_data=pd.read_csv('nsl-kdd/KDDTest+.txt', sep = ',', error_bad_lines=False, header=None)
train_labels = train_data.iloc[:,41]
train_data = train_data.iloc[:,:-2] #Remove the last two columns from train ds
test_ds = test_data.iloc[:,:-2] #Remove the last two columns from train ds


encoder = LabelEncoder()



train_data[1]= encoder.fit_transform(train_data.iloc[:,1])
train_data[2]= encoder.fit_transform(train_data.iloc[:,2])
train_data[3]= encoder.fit_transform(train_data.iloc[:,3])
# test_labels = test_data.iloc[:,41]
# test_encoded_labels = encoder.fit_transform(test_labels)

# test_data[1]= encoder.fit_transform(test_data.iloc[:,1])
# test_data[2]= encoder.fit_transform(test_data.iloc[:,2])
# test_data[3]= encoder.fit_transform(test_data.iloc[:,3])

X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.33, random_state=42)


encoded_labels = encoder.fit_transform(y_train)


test_encoded_labels = encoder.fit_transform(y_test)



class DNNIntrusion(Model):
    
    def __init__(self, data, name):
        super(DNNIntrusion,self).__init__(name=name)
        self.data = data
        self.dense1 = Dense(1024, name="dense1")
        self.dense2 = Dense(1024,  activation="relu", name="dense2")
        self.dense3 = Dense(512, activation="relu", name="dense3")
        self.dense4 = Dense(256, activation="relu", name="dense4")
        self.dense5 = Dense(23, activation="softmax", name="dense5")
        self.relu = tf.keras.layers.ReLU()
        self.batch = BatchNormalization()
        
    def call(self, x):
        x = self.dense1(x)
        x = self.batch(x)
        x = self.dense2(x)
        #x = self.batch(x)
        # x = self.dense2(x)
        # x = self.batch(x)
        # x = self.dense2(x)
        # x = self.batch(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return x

#Scale and Normalizer
normalizer = Normalizer()
train_X = normalizer.fit_transform(X_train)
test_X = normalizer.fit_transform(X_test)
epochs = 10
#Split data

x_train = tf.convert_to_tensor(train_X)
y_train = np.reshape(encoded_labels,(-1,1))
y_train = tf.convert_to_tensor(y_train)

x_test = tf.convert_to_tensor(test_X)
y_test = np.reshape(test_encoded_labels,(-1,1))
y_test = tf.convert_to_tensor(y_test)
cnn_model = DNNIntrusion(x_train,name="intrusion")

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(0.001)

## Metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

def print_layer_variables(model):
    nodes = model.trainable_variables[8]
    #print(model.trainable_variables)
    try:
        for i in range(256):
          results = fit_distribution(nodes[i].numpy(),0.99,0.01)
          print('{}'.format(i+1),results.iloc[0]['chi_square'], results.iloc[0]['Distribution'])
        # results = fit_distribution(tes[1].numpy(),0.99,0.01)
        # print(results.iloc[0]['chi_square'], results.iloc[0]['Distribution'])
        # results = fit_distribution(tes[2].numpy(),0.99,0.01)
        # print(results.iloc[0]['chi_square'], results.iloc[0]['Distribution'])
        # results = fit_distribution(tes[3].numpy(),0.99,0.01)
        # print(results.iloc[0]['chi_square'], results.iloc[0]['Distribution'])        
        # print(tes[255].numpy())
    except NotImplementedError:
        pass
    except ValueError:
        pass
    except AttributeError:
        pass



@tf.function
def train_step(trainDS, labels):
  
  with tf.GradientTape() as tape:

    predictions = cnn_model(trainDS,training=True)
    loss = loss_object(labels, predictions)
  
  gradients = tape.gradient(loss, cnn_model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, cnn_model.trainable_variables))
  
  train_loss(loss)
  train_accuracy(labels, predictions)
  
@tf.function
def test_step(test_data, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = cnn_model(test_data, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

  print_layer_variables(cnn_model)
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)
  
  
EPOCHS = 2


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
  test_step(x_test, y_test)

  print(
    f'Epoch {epoch + 1}: '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'

  )
  print_layer_variables(cnn_model)
  

  
class RNNIntrusion(Model):
    def __init__(self, data):
        super(RNNIntrusion, self).__init__(name="RNN Instrusion")
        #self.dense = 
        pass
    
    def call(self, d):
        pass#d = 