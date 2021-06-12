# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:05:33 2021

@author: Hubert Kyeremateng-Boateng
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.layers import Dense,LSTM, Dropout, Input,Conv2D
from keras.models import Sequential

class Intrusion:
    
    def __init__(self,kdd):
        self.data = kdd

    def model(self):
        f = 'test'
        model = keras.Sequential()
        model.add(keras.Input(shape=(250, 250, 3)))  # 250x250 RGB images
        model.add(keras.layers.Conv2D(32, 5, strides=2, activation="relu"))
        model.add(keras.layers.Conv2D(32, 3, activation="relu"))
        model.add(keras.layers.MaxPooling2D(3))
        return model

    def train(self):
        model = self.model()
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        model.compile(loss=loss_fn, optimizer='adam')
        model.fit(self.data)
# read data
data=pd.read_csv('nsl-kdd/KDDTrain+.txt', sep = ',', error_bad_lines=False)

intrusion = Intrusion(data)
intrusion.train()

