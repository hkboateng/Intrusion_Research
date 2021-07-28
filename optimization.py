# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 13:39:32 2021

@author: Hubert Kyeremateng-Boateng

Optimize Tensorflow training and determining distribution using Dask

Reference: https://github.com/dask/dask-tutorial; https://tutorial.dask.org/
"""

import dask.array as da
import dask.dataframe as dd

train_df = dd.read_csv('nsl-kdd/KDDTrain+.txt', sep = ',', error_bad_lines=False, header=None)