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
import statistics
# read data
train_data=pd.read_csv('nsl-kdd/KDDTrain+.txt', sep = ',', error_bad_lines=False, header=None)
test_data=pd.read_csv('nsl-kdd/KDDTest+.txt', sep = ',', error_bad_lines=False, header=None)
train_labels = train_data.iloc[:,41]
train_data = train_data.iloc[:,:-2] #Remove the last two columns from train ds
test_ds = test_data.iloc[:,:-2] #Remove the last two columns from train ds
num_classes = len(list(set(train_labels)))

encoder = LabelEncoder()



train_data[1]= encoder.fit_transform(train_data.iloc[:,1])
train_data[2]= encoder.fit_transform(train_data.iloc[:,2])
train_data[3]= encoder.fit_transform(train_data.iloc[:,3])


X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)


encoded_labels = encoder.fit_transform(y_train)


test_encoded_labels = encoder.fit_transform(y_test)



# class DNNIntrusion(Model):
    
#     def __init__(self, data, name):
#         super(DNNIntrusion,self).__init__(name=name)
#         self.data = data
#         self.dense1 = Dense(1024, name="dense1")
#         self.dense2 = Dense(1024,  activation="relu", name="dense2")
#         self.dense3 = Dense(512, activation="relu", name="dense3")
#         self.dense4 = Dense(256, activation="relu", name="dense4")
#         self.dense5 = Dense(23, activation="softmax", name="dense5")
#         #self.relu = tf.keras.layers.ReLU()
#         #self.softmax = tf.keras.layers.Activation('softmax')
#         self.batch = BatchNormalization()
        
#     def call(self, x, training=False):
#         x = self.dense1(x)
#         if training:
#             x = self.batch(x)
#         x = self.dense2(x)
#         if training:
#             x = self.batch(x)
#         x = self.dense2(x)
#         if training:
#             x = self.batch(x)
#         x = self.dense2(x)
#         x = self.dense2(x)
#         if training:
#             x = self.batch(x)
#         x = self.dense3(x)
#         x = self.dense4(x)
#         #x = self.relu(x)
#         x = self.dense5(x)
#         #x = self.softmax(x)
#         return x

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

outputs = Dense(23, activation=tf.nn.softmax, name="softmax")(inputs)
model = tf.keras.Model(inputs=input_data, outputs=outputs, name="test_model")
model.summary()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)
history = model.fit(x_train,y_train,epochs=2, validation_split=0.2)

softmax_feature_layer = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name="dense5").output,
)
train_outputs = softmax_feature_layer(x_train)

print("----------------------------------------------------------------------------------------")

train_distributions = method_stats(train_outputs.numpy()) #100778x23
print("----------------------------Test Distribution-------------------------------------------")
test_outputs = softmax_feature_layer(x_test)
test_results_output = test_outputs.numpy()
test_output_transpose = test_results_output.transpose()
train_outputs_transpose = train_outputs.numpy().transpose()

anomaly_threshold = 0.50
    #print(tr)
def calculateAnomaly(train_distributions, test_data):
    degree_of_freedom = 2
    valid_count = 0
    anomaly_count = 0;
    for index,data in train_distributions.iterrows():
        test = test_results_output[index]
        moni_width = len(test)
        train_mean = data['Mean']
        tr_std = data['Standard Deviation']
        val_1 = (tr_std - (degree_of_freedom * train_mean)) < test
        val_2 = test < (tr_std + (degree_of_freedom * train_mean))
        if sum(val_1)/moni_width > anomaly_threshold or sum(val_2)/moni_width > anomaly_threshold:
            valid_count +=1
        else:
            anomaly_count +=1
    return valid_count, anomaly_count


valid_count, anomaly_count = calculateAnomaly(train_distributions, test_results_output)
print('Anamoly Detected is: {:.2f}%'.format((anomaly_count/test_data.shape[1])*100))

# test_distributions = method_stats(test_outputs.numpy())

# def calculateAnomaly(train_distributions, test_distributions):
#     return np.sum(train_distributions['Type of Distribution']==test_distributions['Type of Distribution'])

# result = calculateAnomaly(train_distributions, test_distributions)

# print(f'Percent Anomaly: {((result/num_classes)) * 100}%')

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
