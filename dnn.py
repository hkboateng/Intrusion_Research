# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:19:25 2021

@author: Hubert Kyeremateng-Boateng
"""
from distribution import fit_distribution, calculate_dis_props
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

X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3, random_state=42)


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
        #self.relu = tf.keras.layers.ReLU()
        #self.softmax = tf.keras.layers.Activation('softmax')
        self.batch = BatchNormalization()
        
    def call(self, x, training=False):
        x = self.dense1(x)
        if training:
            x = self.batch(x)
        x = self.dense2(x)
        if training:
            x = self.batch(x)
        x = self.dense2(x)
        if training:
            x = self.batch(x)
        x = self.dense2(x)
        x = self.dense2(x)
        if training:
            x = self.batch(x)
        x = self.dense3(x)
        x = self.dense4(x)
        #x = self.relu(x)
        x = self.dense5(x)
        #x = self.softmax(x)
        return x

#Scale and Normalizer
normalizer = Normalizer()
train_X = normalizer.fit_transform(X_train)
test_X = normalizer.fit_transform(X_test)
epochs = 1
#Split data

x_train = tf.convert_to_tensor(train_X)
y_train = np.reshape(encoded_labels,(-1,1))
y_train = tf.convert_to_tensor(y_train)

x_test = tf.convert_to_tensor(test_X)
y_test = np.reshape(test_encoded_labels,(-1,1))
y_test = tf.convert_to_tensor(y_test)

dnn_model = DNNIntrusion(x_train,name="intrusion")

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(0.001)

## Metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

predict= []




@tf.function
def train_step(trainDS, labels):
  
  with tf.GradientTape() as tape:

    predictions = dnn_model(trainDS,training=True)
    print(predictions)
    loss = loss_object(labels, predictions)
    
    
  
  gradients = tape.gradient(loss, dnn_model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, dnn_model.trainable_variables))
  
  train_loss(loss)
  train_accuracy(labels, predictions)
  
@tf.function
def test_step(test_data, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = dnn_model(test_data)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

  
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)
def print_layer_variables(model):
    print(model.layers)
    # distribution_list = []
    # nodes = model.trainable_variables[8]
    # try:
    #     for i in range(1):
    #       distribution, results = calculate_dis_props(nodes[i].numpy())
    #       distribution_list.append(distribution)
    # except NotImplementedError:
    #     pass
    # except ValueError:
    #     pass
    # except AttributeError:
    #     pass
    # return distribution_list

anomaly = []
count = 0;

EPOCHS = 1
dist_list = []
for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  
  #fit_distribution(X_train,0.99,0.01)
  #train_step(x_train, y_train)
  
  with tf.GradientTape() as tape:

     predictions = dnn_model(x_train,training=True)
     print(predictions)
     print(dnn_model.get_layer("dense5"))
     loss = loss_object(y_train, predictions)

    
  
  gradients = tape.gradient(loss,dnn_model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, dnn_model.trainable_variables))
  
  train_loss(loss)
  train_accuracy(y_train, predictions)
  
  #test_step(x_test, y_test)
  # predictions = cnn_model(x_test, training=False)
  # output = tf.argmax(predictions, axis=1, output_type=tf.int32)
  # 1000
  # results = fit_distribution(output.numpy(),0.99,0.01)
  # result_list.append(results)
  # print(results.iloc[0]['chi_square'], results.iloc[0]['Distribution'])
  
  # distribution, results = calculate_dis_props(predictions[0].numpy())
  # dist_list.append(distribution)
  
  # t_loss = loss_object(y_test, predictions)

  # test_loss(t_loss)
  # test_accuracy(y_test, predictions)
  print(dnn_model.summary())
  print(
    f'Epoch {epoch + 1}: '
    f'Train Loss: {train_loss.result()}, '
    f'Train Accuracy: {train_accuracy.result() * 100}, '\
  )
      

# input_lay = tf.keras.Input(shape=(41,))
# inputs = Dense(1024,name="d1")(input_lay)
# inputs = Dense(256)(inputs)
# inputs = tf.keras.layers.ReLU(name="relu")(inputs)
# inputs = Dense(23)(inputs)
# outputs = tf.keras.layers.Softmax(name="softmax")(inputs)
# model = tf.keras.Model(inputs=input_lay, outputs=outputs, name="test_model")
# model.summary()
# model.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=tf.keras.optimizers.Adam(),
#     metrics=["accuracy"],
# )
# history = model.fit(x_train,y_train,epochs=1)
# features_layer1 = tf.keras.models.Model(
#     inputs=model.inputs,
#     outputs=model.get_layer(name="softmax").output,
# )
# print(features_layer1(x_test))
# def checkAnomaly(trainDistNodes, testDist):
#     print("------- Checking Anomaly----------")
#     tNode = trainDistNodes
#     tsNode = testDist
#     for a,b in enumerate(tNode):
#         test = (tsNode['Type of Distribution'] == b['Type of Distribution'])
#         anomaly.append(test.iloc[0])

# anomaly = []
def calculateAnomaly(trNode, tsNode):
    for a, b in enumerate(trNode):
        test = (tsNode[a]['Type of Distribution'] == b['Type of Distribution'])
        print(test)
        anomaly.append(test.iloc[0])
    print(f'Percent Anomaly: {(1-sum(anomaly)/1) * 100}%')
        
    
# train_dist_list = print_layer_variables(dnn_model)
# predictions = dnn_model(x_test, training=False)


# t_loss = loss_object(y_test, predictions)

# test_loss(t_loss)
# test_accuracy(y_test, predictions)
# # #output = tf.argmax(predictions, axis=1, output_type=tf.int32)

# # # r,s = calculate_dis_props(predictions[4].numpy())

# print(
#     f'Test Loss: {test_loss.result()}, '
#     f'Test Accuracy: {test_accuracy.result() * 100}, '\
#   )
# # test_model = DNNIntrusion(x_train,name="intrusion")    
# # checkAnomaly(dist_list, r)
# # s = output.numpy() == test_encoded_labels
# test_dist_list = print_layer_variables(dnn_model)

# calculateAnomaly(train_dist_list, test_dist_list)


#print(f'Accuracy: {(sum(output.numpy() == test_encoded_labels)/test_encoded_labels.shape[0]) * 100}')
