# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:19:25 2021

@author: Hubert Kyeremateng-Boateng
"""
from distribution import method_stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder

import scipy.stats as st
import tensorflow as tf

from tensorflow.keras.layers import Dense,  BatchNormalization

from sklearn.model_selection import train_test_split

# read data
train_data=pd.read_csv('nsl-kdd/KDDTrain+.txt', sep = ',', error_bad_lines=False, header=None)
test_data=pd.read_csv('nsl-kdd/KDDTest+.txt', sep = ',', error_bad_lines=False, header=None)
train_labels = train_data.iloc[:,41]
test_labels = test_data.iloc[:,41]
train_data = train_data.iloc[:,:-2] #Remove the last two columns from train ds
test_ds = test_data.iloc[:,:-2] #Remove the last two columns from train ds
num_classes = len(list(set(train_labels)))

encoder = LabelEncoder()
# hotEncoder = OneHotEncoder()

# ht_train = hotEncoder.fit_transform(train_data)

train_data[1]= encoder.fit_transform(train_data.iloc[:,1])
test_ds[1]= encoder.transform(test_ds.iloc[:,1])

train_data[2]= encoder.fit_transform(train_data.iloc[:,2])
test_ds[2]= encoder.transform(test_ds.iloc[:,2])

train_data[3]= encoder.fit_transform(train_data.iloc[:,3])
test_ds[3]= encoder.transform(test_ds.iloc[:,3])


encoded_labels = encoder.fit_transform(train_labels)
test_encoded_labels = encoder.transform(test_labels)

#Extract the desire function. Use np.where


#Scale and Normalizer
normalizer = Normalizer()
train_X = normalizer.fit_transform(train_data)
test_X = normalizer.fit_transform(test_ds)
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
history = model.fit(x_train,y_train,epochs=1, validation_split=0.2)


softmax_feature_layer = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name="dense5").output,
)
train_outputs = softmax_feature_layer(x_train)

print("----------------------------------------------------------------------------------------")
train_outputs_numpy = train_outputs.numpy()
train_distributions = method_stats(train_outputs_numpy) #100778x23
print("----------------------------Test Distribution-------------------------------------------")
test_outputs = softmax_feature_layer(x_test)
test_results_output = test_outputs.numpy()
# test_output_transpose = test_results_output.transpose()
# train_outputs_transpose = train_outputs.numpy().transpose()


anomaly_threshold = 0.50
    #print(tr)
def saveDNNModel(model):
    model.save("saved_model/"+model.name)
    
def calculateAnomaly(train_distributions, test_data):
    degree_of_freedom = 1
    valid_count = 0
    anomaly_count = 0;
    for index,data in train_distributions.iterrows():
        test = test_results_output[index]
        moni_width = len(test)
        train_mean = data['Mean']
        tr_std = data['Standard Deviation']
        val_1 = sum((tr_std - (degree_of_freedom * train_mean)) < test)
        val_2 = sum(test < (tr_std + (degree_of_freedom * train_mean)))

        if (val_1/moni_width) > anomaly_threshold and (val_2/moni_width) > anomaly_threshold: ##Re-check
            valid_count +=1
        else:
            anomaly_count +=1
        
    return valid_count, anomaly_count


valid_count, anomaly_count = calculateAnomaly(train_distributions, test_results_output)
print(test_results_output.shape,anomaly_count)
print('Anamoly Detected is: {:.2f}%'.format((anomaly_count/test_results_output.shape[1])*100))
