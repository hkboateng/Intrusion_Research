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
import os
import pandas as pd

import seaborn as sns
from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

def calculateFITSAnomaly(train_distributions, test_activations):
    anomaly_threshold = np.float64(0.5)
    
    lognormal = []
    lognormal_count = 0
    degree_of_freedom = 1
    defect_count = 0;
    #print('Size of Test Activations : {}'.format(len(test_activations)))
    for index,data in train_distributions.iterrows():
        test = test_activations[index] # [2,3,4,5,6,4]
        '''
            Take the log of the activations, take the 
        '''
        moni_width = len(test)

        train_mean = data['Mean']
        tr_std = data['Standard Deviation']
        dist_name = data['Type of Distribution']

        if dist_name == "lognorm":
            epsilon = data['Epsilon']

            norm_data = np.float64(test + epsilon)
            normalized_data_log  = np.log(norm_data)
            test_data_abs  = np.abs(test)
            norm_data = np.float64(test_data_abs + epsilon)
            normalized_data_log  = np.log(norm_data)
            norm_data_z = (normalized_data_log- train_mean)/tr_std
            
            #print('Min val: {0} epsilon {1} Count Nan {2}'.format(np.min(test), norm_data, sum(np.isinf(normalized_data_log))))

            total = np.sum(((train_mean - (degree_of_freedom * tr_std)) < normalized_data_log) & (normalized_data_log < (train_mean + (degree_of_freedom * tr_std))))

            if np.float64(total/moni_width) < anomaly_threshold:
                defect_count +=1
                print("Anomaly.. distribution: {} : {}".format(dist_name,total/moni_width))
            else:
                print("Not Anomaly.. distribution: {0} : {1}".format(dist_name,total/moni_width))       
        elif dist_name == "uniform":
            '''
            For Uniform distribution
            - Store the train a and b. What every the X value is we substract the mu and check the if falls between -sigma*sqrt(3) <=  X- mu <= sigma*sqrt(3)

            '''
            uniform_data = test - train_mean
            total_sum = ((-(tr_std * np.sqrt(3))) <= uniform_data) & (uniform_data <= (tr_std * np.sqrt(3)))
            if np.float64(sum(total_sum)/moni_width) < anomaly_threshold:
                defect_count  += 1
                print("Anomaly.. distribution: {}".format(dist_name))
            else:
                print("Not Anomaly.. distribution: {}".format(dist_name))
        elif dist_name == 'triang':
            peak_val = np.mean(test)
            max_val = np.amax(test)
            min_val = np.amin(test)
            f_fxn = (peak_val - min_val)/(max_val - min_val)
        else:
            total_sum = ((train_mean - (degree_of_freedom * tr_std)) < test) & (test < (train_mean + (degree_of_freedom * tr_std)))
            if np.float64(sum(total_sum)/moni_width) < anomaly_threshold:
                defect_count  += 1
                print("Anomaly.. distribution: {}".format(dist_name))
            else:
                print("Not Anomaly.. distribution: {}".format(dist_name))
        print("<","*"*25,">")
    return defect_count



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
    inputs = keras.layers.BatchNormalization(name="batch1")(inputs)
    inputs = keras.layers.Dense(512, activation='relu', name="dense2")(inputs)
    inputs = keras.layers.BatchNormalization(name="batch2")(inputs)
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

    encoded_labels = encoder.fit_transform(class_labels)
    
    train_X = normalizer.fit_transform(htTrain)
    
    y_train = np.reshape(encoded_labels,(-1,1))
    y_train = tf.convert_to_tensor(y_train)

    x_train = convert_sparse_matrix_to_sparse_tensor(train_X)


    return x_train, y_train, hotEncoder, normalizer,encoder

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

    x_train = convert_sparse_matrix_to_sparse_tensor(train_X)
    return x_train, y_train,hotEncoder,normalizer,encoder

def save_train_preprocessing_paramters(filename_df, oneHotEncoder, normalizer,labelEncoder):
    preprocessor_df = pd.DataFrame(columns=('Class Label','HotEncoder','Normalizer', 'Label Encoder'))
    preprocessor_df = preprocessor_df.append({'Class Label': filename_df,  'HotEncoder': oneHotEncoder, 'Normalizer': normalizer, 'Label Encoder': labelEncoder},ignore_index=True)


    save_data(pathname("optimizers"),filename_df,preprocessor_df)
    

def save_data(foldername,filename, date_to_save):
    filepath = os.path.join(foldername, os.getcwd()+'/distributions/'+filename)

    jb.dump(date_to_save, filepath)
    
def load_data(foldername,filename):
    filepath = os.path.join(foldername, os.getcwd()+'/distributions/'+filename)
    return jb.load(filepath)
def generate_individual_class_model(class_train_data,epochs,monitoring_node="softmax"):

    class_labels = class_train_data.iloc[:,41]  #Get All Labels from Train data
    unique_train_labels = np.unique(class_labels)

    #preprocess_df = pd.DataFrame(columns=('Class Label','HotEncoder','Normalizer'))

    for classname in unique_train_labels:
        classData = class_train_data.loc[class_train_data.iloc[:,41] == classname]
        train_data, train_labels = generate_dataset_labels(classData)

        x_train, y_train,hotEncoder,normalizer,encoder = process_individual_dataset(train_data, train_labels)
        save_train_preprocessing_paramters(classname+".joblib", hotEncoder, normalizer,encoder)
        model = anomaly_model(x_train.shape[1])
        model.fit(x_train,y_train,epochs=epochs)
        saveDNNModel(model, "models/"+classname+"_model")
        print("------ Generating Monitoring node distributions -----------")
        train_node = getMonitoringNode(model, node=monitoring_node, dataset=x_train)
        train_distributions = method_stats(train_node.numpy())
        save_data("distributions",classname+'_train_distributions.joblib', train_distributions)
        print("------ Completed Generating Monitoring node distributions ----------------")


def loadDNNModel(fileName):
    print(fileName)
    return tf.keras.models.load_model("saved_model/"+fileName)

def generate_test_dataset(dataset, filename):
    test_data, test_labels = generate_dataset_labels(dataset)

    preprocessor_df = load_data("distributions",filename)

    oneHotEncoder = preprocessor_df['HotEncoder'][0]
    normalizer= preprocessor_df['Normalizer'][0]
    htTest = oneHotEncoder.transform(test_data)
    normTest = normalizer.transform(htTest)
    test_data = convert_sparse_matrix_to_sparse_tensor(normTest)
    return test_data,test_labels

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

    return test_df.loc[test_df.iloc[:,41].isin(unique_vals_test_train)]


def test_individual_model_classes(model, train_classes):
    for classname in train_classes:
        pass
        

def generate_class_model(dataset_df,individual_model=False, training_mode= True, epochs=1,train_classes=None, monitoring_node="softmax"):

    if individual_model:
        if training_mode:
            generate_individual_class_model(dataset_df,epochs,monitoring_node)
        else:

            for classname in train_classes:
                test_data, _ = generate_test_dataset(dataset_df,classname+".joblib")
                train_distributions = load_data("distributions",classname+"_train_distributions.joblib")
                model = loadDNNModel("models/"+classname+"_model")
                monitored_node = getMonitoringNode(model,node=monitoring_node,dataset=test_data)
                defect_count = calculateFITSAnomaly(train_distributions, monitored_node.numpy())
                print('Anamoly Detected for class {} is: {:.2f}%'.format(classname, (defect_count/monitored_node.shape[1])*100))

    else:
        if training_mode:

            train_data, train_labels = generate_dataset_labels(dataset_df)

            processed_dataset,processed_labels, hotEncoder,normalizer, labelEncoder  = process_dataset(train_data,train_labels)
            print("-"*10,">  Saving Normalizer, Label Encoder and OneHotEncoder")
            save_train_preprocessing_paramters('all_filtered.joblib', hotEncoder, normalizer, labelEncoder )
            print("-"*10,">Saving Normalizer and OneHotEncoder complete")
            # model = loadDNNModel("models/model_all_filtered")
            model = anomaly_model(processed_dataset.shape[1])
            model.fit(processed_dataset,processed_labels,epochs=epochs)
            print("-"*10,">Saving Trained Model")
            saveDNNModel(model, "models/model_all_filtered")
            print("-"*10,">Trained model saved")
            print("-"*10,">Generating Monitoring node distributions","-"*10)

            monitoring_node = get_monitoring_node(model, node=monitoring_node)
            train_node = monitoring_node(processed_dataset)

            train_distributions = method_stats(train_node.numpy())
            save_data("distributions",'train_distributions.joblib', train_distributions)
            return train_distributions
        else:
            x_test,_ = generate_test_dataset(dataset_df,'all_filtered.joblib')

            test_monitoring_node = getMonitoringNode(modelName="models/model_all_filtered",node=monitoring_node,dataset=x_test)
            train_distributions = load_data("distributions","train_distributions.joblib")

            defect_count = calculateFITSAnomaly(train_distributions, test_monitoring_node.numpy())

            print('Anamoly Detected for class is: {:.2f}%'.format((defect_count/test_monitoring_node.shape[1])*100))
            return test_monitoring_node,train_distributions

def getMonitoringNode(model=None, modelName=None,node=None, dataset=None):
    if model == None:
        model = loadDNNModel(modelName)
        monitored_node = get_monitoring_node(model, node)
        test_monitoring_node = monitored_node(dataset)
        return test_monitoring_node
    else:
        monitored_node = get_monitoring_node(model, node)
        test_monitoring_node = monitored_node(dataset)
        return test_monitoring_node

def plot_confusion_matrix(correct_labels, predict_labels):
    cm = confusion_matrix(correct_labels, predict_labels)
    sns.heatmap(cm)

def gennerate_confusion_matrix(dataset,parameters_filename, model_name):
    x_test,y_label =  generate_test_dataset(dataset,parameters_filename)
    encoder = LabelEncoder()
    y_labels = encoder.fit_transform(y_label)

    model = loadDNNModel(model_name)

    y_pred = model.predict(x_test)
    output = tf.argmax(y_pred, axis=1, output_type=tf.int32)
    plot_confusion_matrix(y_labels,output.numpy())
    return y_labels, y_label

def pathname(pathname):
    return os.path.dirname(os.path.join(os.getcwd(),pathname,pathname))
    
train_data=pd.read_csv('nsl-kdd/KDDTrain+.txt', sep = ',', error_bad_lines=False, header=None)
test_data=pd.read_csv('nsl-kdd/KDDTest+.txt', sep = ',', error_bad_lines=False, header=None)
train_df = train_data
test_df = test_data
train_labels = train_data.iloc[:,41]
test_labels = test_data.iloc[:,41]
X_train, y_train, X_test, y_test = train_test_split(train_df,train_labels, train_size=0.35, random_state=42)
monitoring_node = "softmax"

num_classes = len(list(set(train_labels)))

train_data = train_data.drop([41], axis=1) #Drop labels from train dataset
test_data  = test_data.drop([41], axis=1) 

unique_train_labels = np.unique(train_labels)
unique_test_labels = np.unique(test_labels)
unique_btw_test_train = np.intersect1d(unique_test_labels,unique_train_labels)
unique_vals_test_train = list(set(unique_test_labels)-set(unique_train_labels))

test_data_filter = test_df.loc[test_df.iloc[:,41].isin(unique_vals_test_train)]
individual_model = False
training_mode = False

train_distributions = generate_class_model(X_train, individual_model, epochs=20,monitoring_node=monitoring_node)


train_df_protocol_types = list(set(train_df.iloc[:,1]))
train_df_services = list(set(train_df.iloc[:,2]))
train_df_flag = list(set(train_df.iloc[:,3]))

one_hot_protocol = CountVectorizer(vocabulary=train_df_protocol_types, binary=True)
one_hot_services = CountVectorizer(vocabulary=train_df_services, binary=True)
one_hot_flag = CountVectorizer(vocabulary=train_df_flag, binary=True)

train_df_protocol_types_onehot = one_hot_protocol.fit_transform(train_df.iloc[:,1].values)
train_df_services_onehot = one_hot_services.fit_transform(train_df.iloc[:,2].values)
train_df_flag_onehot = one_hot_flag.fit_transform(train_df.iloc[:,3].values)

train_distributions = generate_class_model(train_df, individual_model, epochs=1,monitoring_node=monitoring_node)


test_node, train_distributions_test = generate_class_model(test_data_filter, individual_model,training_mode,train_classes= unique_train_labels,monitoring_node=monitoring_node)
# test_node_data = test_node.numpy()


'''
Meeting Notes - 09/03/2021
For training node activiation, for each feature F(1) -F(N),
 find  the min value, (add the min val plus very small value) - Epsilon.

For the test, add the same episilon (min trianing value and small value).

Take the log of test and then check the range of the std and mean. 

For Uniform distribution
- Store the train a and b. What every the X value is we substract the mu and check the if falls between -sigma*sqrt(3) <=  X- mu <= sigma*sqrt(3)

- AAAI Symposium
Inter Joint Conf on AI(IJCAI)
CVPI
ICML

- Survey review on IEEE Anamoly Detection

DBLP
Google Scholar
'''
