# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:19:25 2021

@author: Hubert Kyeremateng-Boateng
"""
from distribution import method_stats
import numpy as np

from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder

import joblib as jb


from tensorflow import keras
import tensorflow as tf

# read data
#from tensorflow.keras import layers
#from tensorflow.keras.layers import Dense,  BatchNormalization

from sklearn.model_selection import train_test_split
import pandas as pd
def calculateAnomaly(train_distributions, test_data):
    anomaly_threshold = np.float64(0.50)
    degree_of_freedom = 1
    defect_count = 0;
    for index,data in train_distributions.iterrows():
        test = test_data[index]
        moni_width = len(test)
        train_mean = data['Mean']
        tr_std = data['Standard Deviation']
        total_sum = ((tr_std - (degree_of_freedom * train_mean)) < test) & (test < (tr_std + (degree_of_freedom * train_mean)))
        print(sum(total_sum),sum(total_sum)/moni_width , np.float64(sum(total_sum)/moni_width) < anomaly_threshold)
        if np.float64(sum(total_sum)/moni_width) < anomaly_threshold:
            defect_count  += 1

    print('Anamoly Detected is: {:.2f}%'.format((defect_count/test_data.shape[1])*100))


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))

def saveDNNModel(model, filename):
    model.save("saved_model/"+filename)
    

def generate_dataset_labels(dataset):
    labels = dataset.iloc[:,41]
    labels = labels.reset_index()
    labels = labels.drop(['index'], axis=1)
    
    data = dataset.drop([41], axis=1)
    data = data.reset_index()
    data = data.drop(['index'], axis=1)
    return data, labels

def load_train_preprocessing_paramters(filename_df):
    return jb.load(filename_df)

def anomaly_model(train_data_shape):
    input_data = tf.keras.Input(shape=(train_data_shape,))
    inputs = keras.layers.Dense(1024, activation="relu", name="dense1")(input_data)
    inputs = keras.layers.Dense(1024, activation="relu", name="dense2")(inputs)
    inputs = keras.layers.BatchNormalization()(inputs)
    inputs = keras.layers.Dense(1024, activation="relu", name="dense3")(inputs)
    inputs = keras.layers.BatchNormalization()(inputs)
    inputs = keras.layers.Dense(512, activation="relu", name="dense4")(inputs)
    inputs = keras.layers.BatchNormalization()(inputs)
    inputs = keras.layers.Dense(256, activation="relu", name="dense5")(inputs) #256x23
    inputs = keras.layers.BatchNormalization()(inputs)
    inputs = keras.layers.Dense(128, activation="relu", name="dense6")(inputs) #256x23
    inputs = keras.layers.BatchNormalization()(inputs)
    outputs = keras.layers.Dense(23, activation=tf.nn.softmax, name="softmax")(inputs)
    model = tf.keras.Model(inputs=input_data, outputs=outputs, name="test_model")

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model

# history = model.fit(x_train,y_train,epochs=500, validation_split=0.2)

def process_dataset(train_dataSet,  class_labels, hotEncoder=None, normalizer=None):
    
    encoder = LabelEncoder()
    if hotEncoder == None:
        hotEncoder = OneHotEncoder(handle_unknown='ignore')

    if normalizer == None:
        normalizer = Normalizer()
    htTrain = hotEncoder.fit_transform(train_dataSet)
    # htTest = hotEncoder.transform(test_dataSet)

    encoded_labels = encoder.fit_transform(class_labels)

    #Scale and Normalizer
    
    train_X = normalizer.fit_transform(htTrain)
    #test_X = normalizer.fit_transform(htTest)

    #Split data
    
    y_train = np.reshape(encoded_labels,(-1,1))
    y_train = tf.convert_to_tensor(y_train)

    x_train = convert_sparse_matrix_to_sparse_tensor(train_X)
    #x_test = convert_sparse_matrix_to_sparse_tensor(test_X)

    return x_train, y_train, hotEncoder, normalizer#, x_test

def process_individual_dataset(train_dataSet, class_labels):
    encoder = LabelEncoder()
    hotEncoder = OneHotEncoder(handle_unknown='ignore')

    htTrain = hotEncoder.fit_transform(train_dataSet)

    encoded_labels = encoder.fit_transform(class_labels)

    #Scale and Normalizer
    normalizer = Normalizer()
    train_X = normalizer.fit_transform(htTrain)

    y_train = np.reshape(encoded_labels,(-1,1))
    y_train = tf.convert_to_tensor(y_train)

    x_train = tf.convert_to_tensor(train_X)
    return x_train, y_train,hotEncoder,normalizer

def save_train_preprocessing_paramters(oneHotEncoder, normalizer, filename_df):
    
    preprocessor_df = pd.DataFrame(columns=('Class Label','HotEncoder','Normalizer'))
    preprocessor_df = preprocessor_df.append({'Class Label': filename_df,  'HotEncoder': oneHotEncoder, 'Normalizer': normalizer},ignore_index=True)
    jb.dump(preprocessor_df, filename_df, compress=True)
    
def save_data(filename, date_to_save):
    jb.dump(date_to_save, filename)
    
def load_data(filename):
    return jb.load(filename)
def generate_individual_class_model(class_train_data,epochs):
    class_labels = class_train_data.iloc[:,41]  #Get All Labels from Train data
    unique_train_labels = np.unique(class_labels)

    model_df = pd.DataFrame(columns=('Class Label','Model'))
    #preprocess_df = pd.DataFrame(columns=('Class Label','HotEncoder','Normalizer'))

    for classname in unique_train_labels:
        classData = class_train_data.loc[class_train_data.iloc[:,41] == classname]
        train_data, train_labels = generate_dataset_labels(classData)

        x_train, y_train,hotEncoder,normalizer = process_individual_dataset(train_data, train_labels)
        save_train_preprocessing_paramters( hotEncoder, normalizer,classname)
        model = anomaly_model(x_train.shape[1])
        model.fit(x_train,y_train,epochs=epochs)
        # model_df = model_df.append({'Class Label': classname, 'Model': model},ignore_index=True)
        # preprocess_df = preprocess_df.append({'Class Label': classname,  'HotEncoder': hotEncoder, 'Normalizer': normalizer},ignore_index=True)
    return model_df
        
def loadDNNModel(fileName):
    return tf.keras.models.load_model("saved_model/"+fileName)

def generate_test_dataset(dataset, filename):
    test_data, test_labels = generate_dataset_labels(dataset)
    preprocessor_df = load_data(filename)
    oneHotEncoder = preprocessor_df['HotEncoder'][0]
    normalizer= preprocessor_df['Normalizer'][0]
    htTest = oneHotEncoder.transform(test_data)
    normTest = normalizer.transform(htTest)
    test_data = convert_sparse_matrix_to_sparse_tensor(normTest)
    return test_data

def get_monitoring_node(model, node):
    monitoring_node = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(name=node).output,
    )
    return monitoring_node

def filter_test_train_dataset(test_df, train_df, test_labels, train_labels):
    unique_train_labels = np.unique(train_labels)
    unique_test_labels = np.unique(test_labels)
    unique_vals_test_train = list(set(unique_test_labels)-set(unique_train_labels))
    #labels = list(lambda x:x not in train_labels)
    return test_df.loc[test_df.iloc[:,41].isin(unique_vals_test_train)]

def generate_class_model(dataset_df,individual_model=False, training_mode= True, epochs=1):
    # processed_dataset,processed_labels, processed_test_data = process_dataset(train_data, test_data,train_labels)
    if individual_model:
        #train_data, train_labels = generate_dataset_labels(dataset_df)
        if training_mode:
            preprocess_df = generate_individual_class_model(dataset_df,epochs)
            return preprocess_df
        else:
            test_data, test_labels = generate_dataset_labels(dataset_df)
            
            processed_dataset,processed_labels, hotEncoder,normalizer  = process_dataset(test_data,test_labels)
            save_train_preprocessing_paramters('all_filtered.joblib', hotEncoder, normalizer )

            model = anomaly_model(processed_dataset.shape[1])
            model.fit(processed_dataset,processed_labels,epochs=epochs)
            saveDNNModel(model, "trained_model")

    else:
        if training_mode:

            train_data, train_labels = generate_dataset_labels(dataset_df)

            processed_dataset,processed_labels, hotEncoder,normalizer  = process_dataset(train_data,train_labels)
            print("-"*20,">  Saving Normalizer and OneHotEncoder")
            save_train_preprocessing_paramters(hotEncoder, normalizer ,'all_filtered.joblib')
            print("Saving Normalizer and OneHotEncoder complete")
            # preprocess_df = pd.DataFrame(columns=('Class Label','HotEncoder','Normalizer'))
            # preprocess_df = preprocess_df.append({'Class Label': 'all_filtered',  'HotEncoder': hotEncoder, 'Normalizer': normalizer},ignore_index=True)
            print(type(processed_dataset))
            model = anomaly_model(processed_dataset.shape[1])
            model.fit(processed_dataset,processed_labels,epochs=epochs)
            print("------ Saving Trained Model--------------------------")
            saveDNNModel(model, "model_all_filtered")
            print("------ Trained model saved-----------------------")
            print("------ Generating Monitoring node distributions -----------")
            #model = loadDNNModel("model_all_filtered")
            
            monitoring_node = get_monitoring_node(model, "softmax")
            train_node = monitoring_node(processed_dataset)
            train_distributions = method_stats(train_node.numpy())
            save_data('train_distributions.joblib', train_distributions)
            print("------ Completed Generating Monitoring node distributions ----------------")
            # x_test = generate_test_dataset(test_df,'all_filtered.joblib')
            # model = loadDNNModel("model_all_filtered")
            # monitoring_node = get_monitoring_node(model, "dense6")
            # test_monitoring_node = monitoring_node(x_test)
            # train_distributions = load_data("train_distribution.joblib")
            # calculateAnomaly(train_distributions, test_monitoring_node.numpy())
        else:
            x_test = generate_test_dataset(dataset_df,'all_filtered.joblib')
            model = loadDNNModel("model_all_filtered")
            monitoring_node = get_monitoring_node(model, "softmax")
            test_monitoring_node = monitoring_node(x_test)
            train_distributions = load_data("train_distributions.joblib")
            calculateAnomaly(train_distributions, test_monitoring_node.numpy())
    


train_data=pd.read_csv('nsl-kdd/KDDTrain+.txt', sep = ',', error_bad_lines=False, header=None)
test_data=pd.read_csv('nsl-kdd/KDDTest+.txt', sep = ',', error_bad_lines=False, header=None)
train_df = train_data
test_df = test_data
train_labels = train_data.iloc[:,41]
test_labels = test_data.iloc[:,41]



num_classes = len(list(set(train_labels)))

train_data = train_data.drop([41], axis=1) #Drop labels from train dataset
test_data  = test_data.drop([41], axis=1) 

unique_train_labels = np.unique(train_labels)
unique_test_labels = np.unique(test_labels)
unique_btw_test_train = np.intersect1d(unique_test_labels,unique_train_labels)
unique_vals_test_train = list(set(unique_test_labels)-set(unique_train_labels))
#labels = list(lambda x:x not in train_labels)
test_data_filter = test_df.loc[test_df.iloc[:,41].isin(unique_vals_test_train)]


#filtered_dataset  = filter_test_train_dataset(test_data, train_data, test_labels, train_labels);
individual_model = True
training_mode = False
generate_class_model(train_df, individual_model, epochs=1)


# params = load_train_preprocessing_paramters('all_filtered.joblib')
# test_dataset,_ = generate_dataset_labels(test_data_filter)
# ht = params['HotEncoder'][0]
# ht_test = ht.transform(test_dataset)
generate_class_model(test_data_filter, individual_model,training_mode)

# print('Anamoly Detected is: {:.2f}%'.format((defect_count/test_results_output.shape[1])*100))
# x_test = generate_test_dataset(test_df)
# #train_data = train_data.drop([41], axis=1) #Drop labels from train dataset
# #train_X, y_train = process_train_data(train_data,train_labels)

# # model = anomaly_model(x_train.shape[1])
# # history = model.fit(x_train,y_train,epochs=50)


# model = loadDNNModel("trained_model")


# x_train = None
# train_outputs = monitoring_node(x_train)

# print("----------------------------------------------------------------------------------------")
# train_outputs_numpy = train_outputs.numpy()
# train_distributions = method_stats(train_outputs_numpy) #100778x23
# print("----------------------------Test Distribution-------------------------------------------")
# test_outputs = monitoring_node(x_test)
# print(x_train.shape)
#test_results_output = test_outputs.numpy()
# # test_output_transpose = test_results_output.transpose()
# # train_outputs_transpose = train_outputs.numpy().transpose()

# # saveDNNModel(model, "filtered_dataset")    

