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
train_df = train_data
test_df = test_data
train_labels = train_data.iloc[:,41]
test_labels = test_data.iloc[:,41]

def generate_dataset_labels(dataset):
    labels = dataset.iloc[:,41]
    labels = labels.reset_index()
    labels = labels.drop(['index'], axis=1)
    
    data = dataset.drop([41], axis=1)
    data = data.reset_index()
    data = data.drop(['index'], axis=1)
    return data, labels

num_classes = len(list(set(train_labels)))

train_data = train_data.drop([41], axis=1) #Drop labels from train dataset
test_data  = test_data.drop([41], axis=1) 

unique_train_labels = np.unique(train_labels)
unique_test_labels = np.unique(test_labels)
unique_btw_test_train = np.intersect1d(unique_test_labels,unique_train_labels)
unique_vals_test_train = list(set(unique_test_labels)-set(unique_train_labels))
#labels = list(lambda x:x not in train_labels)
test_data_filter = test_df.loc[test_df.iloc[:,41].isin(unique_vals_test_train)]

#num_classes = len(list(set(train_labels)))

encoder = LabelEncoder()
hotEncoder = OneHotEncoder(handle_unknown='ignore')

htTrain = hotEncoder.fit_transform(train_data)
htTest = hotEncoder.transform(test_data)

encoded_labels = encoder.fit_transform(train_labels)

#Scale and Normalizer
normalizer = Normalizer()
train_X = normalizer.fit_transform(htTrain)
test_X = normalizer.transform(htTest)
epochs = 1

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))


sparse_tensor_train = convert_sparse_matrix_to_sparse_tensor(train_X)

x_train = sparse_tensor_train#tf.convert_to_tensor(train_X)
y_train = np.reshape(encoded_labels,(-1,1))
y_train = tf.convert_to_tensor(y_train)

x_test = convert_sparse_matrix_to_sparse_tensor(test_X)
# y_test = np.reshape(test_encoded_labels,(-1,1))
# y_test = tf.convert_to_tensor(y_test)

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# optimizer = tf.keras.optimizers.Adam(0.001)

# ## Metrics
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
def saveDNNModel(model, filename):
    model.save("saved_model/"+filename)

def anomaly_model(train_data_shape):
    input_data = tf.keras.Input(shape=(train_data_shape,))
    inputs = Dense(1024, activation="relu", name="dense1")(input_data)
    inputs = Dense(1024, activation="relu", name="dense2")(inputs)
    inputs = BatchNormalization()(inputs)
    inputs = Dense(1024, activation="relu", name="dense3")(inputs)
    inputs = BatchNormalization()(inputs)
    inputs = Dense(512, activation="relu", name="dense4")(inputs)
    inputs = BatchNormalization()(inputs)
    inputs = Dense(256, activation="relu", name="dense5")(inputs) #256x23
    inputs = BatchNormalization()(inputs)
    inputs = Dense(128, activation="relu", name="dense6")(inputs) #256x23
    inputs = BatchNormalization()(inputs)
    outputs = Dense(23, activation=tf.nn.softmax, name="softmax")(inputs)
    model = tf.keras.Model(inputs=input_data, outputs=outputs, name="test_model")

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model


# history = model.fit(x_train,y_train,epochs=500, validation_split=0.2)

def process_dataset(train_dataSet, test_dataSet, class_labels):
    encoder = LabelEncoder()
    hotEncoder = OneHotEncoder(handle_unknown='ignore')

    htTrain = hotEncoder.fit_transform(train_dataSet)
    htTest = hotEncoder.transform(test_dataSet)

    encoded_labels = encoder.fit_transform(class_labels)

    #Scale and Normalizer
    normalizer = Normalizer()
    train_X = normalizer.fit_transform(htTrain)
    test_X = normalizer.fit_transform(htTest)

    #Split data
    
    y_train = np.reshape(encoded_labels,(-1,1))
    y_train = tf.convert_to_tensor(y_train)

    x_train = convert_sparse_matrix_to_sparse_tensor(train_X)
    x_test = convert_sparse_matrix_to_sparse_tensor(test_X)
    return x_train, y_train, x_test

def process_individual_dataset(train_dataSet, class_labels):
    encoder = LabelEncoder()
    hotEncoder = OneHotEncoder(handle_unknown='ignore')

    htTrain = hotEncoder.fit_transform(train_dataSet)
    #htTest = hotEncoder.transform(test_dataSet)

    encoded_labels = encoder.fit_transform(class_labels)

    #Scale and Normalizer
    normalizer = Normalizer()
    train_X = normalizer.fit_transform(htTrain)
    #test_X = normalizer.fit_transform(htTest)

    #Split data
    
    y_train = np.reshape(encoded_labels,(-1,1))
    y_train = tf.convert_to_tensor(y_train)

    x_train = convert_sparse_matrix_to_sparse_tensor(train_X)
    #x_test = convert_sparse_matrix_to_sparse_tensor(test_X)
    return x_train, y_train
def generate_individual_class_model(class_train_data,epochs):
    class_labels = class_train_data.iloc[:,41]
    unique_train_labels = np.unique(class_labels)
    model_df = pd.DataFrame(columns=('Class Label','Model'))
    print(unique_train_labels)
    for classname in unique_train_labels:
        classData = class_train_data.loc[class_train_data.iloc[:,41] == classname]
        train_data, train_labels = generate_dataset_labels(classData)
        print(classname)
        x_train, y_train = process_individual_dataset(train_data, train_labels)
        model = anomaly_model(x_train.shape[1])
        history = model.fit(x_train,y_train,epochs=epochs)
        model_df = model_df.append({'Class Label': classname, 'Model': model },ignore_index=True)

    return model_df
        
def generate_class_model(dataset_df, test_df,individual_model=False,  epochs=1):
    train_data, train_labels = generate_dataset_labels(dataset_df)
    test_data, test_labels = generate_dataset_labels(test_df)

    # processed_dataset,processed_labels, processed_test_data = process_dataset(train_data, test_data,train_labels)
    if individual_model:
        trained_model = generate_individual_class_model(dataset_df,epochs)
        return trained_model
    else:
        train_data, train_labels = generate_dataset_labels(dataset_df)
        test_data, test_labels = generate_dataset_labels(test_df)
        processed_dataset,processed_labels, processed_test_data = process_dataset(train_data, test_data,train_labels)
        print(processed_dataset.shape)
        model = anomaly_model(processed_dataset.shape[1])
        history = model.fit(processed_dataset,processed_labels,epochs=epochs)
        saveDNNModel(model, "trained_model")
    
    
    
individual_model = True
trained_model = generate_class_model(train_df, test_df, individual_model, epochs=20)

# #train_data = train_data.drop([41], axis=1) #Drop labels from train dataset
# #train_X, y_train = process_train_data(train_data,train_labels)

# # model = anomaly_model(x_train.shape[1])
# # history = model.fit(x_train,y_train,epochs=50)

def loadDNNModel(fileName):
    return tf.keras.models.load_model("saved_model/"+fileName)
model = loadDNNModel("trained_model")
softmax_feature_layer = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name="dense6").output,
)

train_outputs = softmax_feature_layer(x_train)

print("----------------------------------------------------------------------------------------")
train_outputs_numpy = train_outputs.numpy()
train_distributions = method_stats(train_outputs_numpy) #100778x23
print("----------------------------Test Distribution-------------------------------------------")
test_outputs = softmax_feature_layer(x_test)
# print(x_train.shape)
test_results_output = test_outputs.numpy()
# # test_output_transpose = test_results_output.transpose()
# # train_outputs_transpose = train_outputs.numpy().transpose()


anomaly_threshold = np.float64(0.9)
#     #print(tr)

# # saveDNNModel(model, "filtered_dataset")    



def calculateAnomaly(train_distributions, test_data):
    degree_of_freedom = 1
    valid_count = 0
    anomaly_count = 0;
    defect_count = 0;
    for index,data in train_distributions.iterrows():
        test = test_results_output[index]

        moni_width = len(test)
        train_mean = data['Mean']
        tr_std = data['Standard Deviation']
        val_1 = sum((tr_std - (degree_of_freedom * train_mean)) < test)
        val_2 = sum(test < (tr_std + (degree_of_freedom * train_mean)))
        #print(val_1/moni_width, val_2/moni_width)
        total_sum = ((tr_std - (degree_of_freedom * train_mean)) < test) & (test < (tr_std + (degree_of_freedom * train_mean)))
        print(sum(total_sum),sum(total_sum)/moni_width , np.float64(sum(total_sum)/moni_width) < anomaly_threshold)
        if np.float64(sum(total_sum)/moni_width) < anomaly_threshold:
            defect_count  += 1

    return defect_count


defect_count = calculateAnomaly(train_distributions, test_results_output)

print('Anamoly Detected is: {:.2f}%'.format((defect_count/test_results_output.shape[1])*100))
