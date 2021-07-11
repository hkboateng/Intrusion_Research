# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:19:25 2021

@author: Hubert Kyeremateng-Boateng
"""
from distribution import method_stats
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder, Normalizer
from fitter import Fitter
import scipy.stats as st
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

X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)


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


input_data = tf.keras.Input(shape=(41,))
inputs = Dense(1024, activation="relu", name="dense1")(input_data)
inputs = Dense(1024, activation="relu", name="dense2")(inputs)
inputs = BatchNormalization()(inputs)
inputs = Dense(1024, activation="relu", name="dense3")(inputs)
inputs = BatchNormalization()(inputs)
#inputs = Dense(1024, activation="relu", name="dense3")(inputs)
#inputs = Dense(1024, activation="relu", name="dense3")(inputs)
inputs = Dense(512, activation="relu", name="dense4")(inputs)
inputs = BatchNormalization()(inputs)
inputs = Dense(256, activation="relu", name="dense5")(inputs) #256x23
inputs = BatchNormalization()(inputs)
#inputs = Dense(23, activation=tf.nn.softmax)(inputs)
outputs = Dense(23, activation=tf.nn.softmax, name="softmax")(inputs)
model = tf.keras.Model(inputs=input_data, outputs=outputs, name="test_model")
model.summary()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)
history = model.fit(x_train,y_train,epochs=1, validation_split=0.2)

softmax_feature_layer = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name="softmax").output,
)
train_outputs = softmax_feature_layer(x_train)

import scipy.stats
print("--------------------------------------------------------------------------------------------")
#https://github.com/cokelaer/fitter/blob/58a41cf4b4a119e0d86e1adc7462cc14eefc93e7/src/fitter/fitter.py#L42
# def get_all_distributions():
#     distributions = []
#     for this in dir(scipy.stats):
#         if "fit" in eval("dir(scipy.stats." + this + ")"):
#             distributions.append(this)
#     return distributions
# all_distributions = get_all_distributions()
# data = train_outputs.numpy()
# all_distributions = [st.laplace, st.norm]
# mles = []
# for distribution in all_distributions:
#     m,v,s,k = distribution.stats(data, moments='mvsk')
#     pars = distribution.fit(data)
#     mle = distribution.nnlf(pars, data)
#     mles.append(mle)
    
# results = [(distribution.name, mle) for distribution, mle in zip(all_distributions, mles)]
# best_fit = sorted(zip(all_distributions, mles), key=lambda d: d[1])[0]
# print('Best fit reached using {}, MLE value: {}'.format(best_fit[0].name, best_fit[1]))
data_test = train_outputs.numpy()
train_distributions = method_stats(train_outputs.numpy()) #100778x23
print("-"*50)
test_outputs = softmax_feature_layer(x_test)

test_distributions = method_stats(test_outputs.numpy())
# anomaly = []
# def calculateAnomaly(train_distributions, test_distributions):
#     compared = train_distributions.compare(test_distributions, keep_shape=True, keep_equal=True)
#     #print(f'Percent Anomaly: {(1-sum(anomaly)/1) * 100}%')
#     return compared
# result = calculateAnomaly(train_distributions, test_distributions)
# anomaly = []
# count = 0;

# EPOCHS = 1
# dist_list = []
# for epoch in range(EPOCHS):
#   # Reset the metrics at the start of the next epoch
#   train_loss.reset_states()
#   train_accuracy.reset_states()
#   test_loss.reset_states()
#   test_accuracy.reset_states()
  
#   #fit_distribution(X_train,0.99,0.01)
#   #train_step(x_train, y_train)
  
#   with tf.GradientTape() as tape:

#       predictions = dnn_model(x_train,training=True)

#       loss = loss_object(y_train, predictions)

    
  
#   gradients = tape.gradient(loss,dnn_model.trainable_variables)

#   optimizer.apply_gradients(zip(gradients, dnn_model.trainable_variables))
  
#   train_loss(loss)
#   train_accuracy(y_train, predictions)
  

#   predictions = dnn_model(x_train, training=False)
#   # output = tf.argmax(predictions, axis=1, output_type=tf.int32)
#   print(predictions)

  
#   # distribution, results = calculate_dis_props(predictions[0].numpy())
#   # dist_list.append(distribution)
  
#   # t_loss = loss_object(y_test, predictions)

#   # test_loss(t_loss)
#   # test_accuracy(y_test, predictions)

#   print(
#      f'Epoch {epoch + 1}: '
#      f'Train Loss: {train_loss.result()}, '
#      f'Train Accuracy: {train_accuracy.result() * 100}, '\
#   )
      


# def checkAnomaly(trainDistNodes, testDist):
#     print("------- Checking Anomaly----------")
#     tNode = trainDistNodes
#     tsNode = testDist
#     for a,b in enumerate(tNode):
#         test = (tsNode['Type of Distribution'] == b['Type of Distribution'])
#         anomaly.append(test.iloc[0])


        
    
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
