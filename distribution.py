# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:38:35 2021

@author: Hubert Kyeremateng-Boateng
"""

import re
import pandas as pd
import numpy as np
# from mlutils import dataset, connector
import scipy.stats
from scipy.stats import *
from scipy.stats import lognorm, gamma,beta,exponnorm
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm 
import seaborn as sns
import pylab as py 
from scipy import stats
# vehicle_df = pd.read_csv('vehicles.csv')
dist_names = ['weibull_min','norm','weibull_max','beta','invgauss','uniform','gamma','expon','lognorm','pearson3','triang']
# columns = ['id','price','year','manufacturer','condition','odometer']
# vehicle_data = vehicle_df[columns]


# vehicle_data.isna().sum()

def standarise(column,pct,pct_lower):
    sc = StandardScaler() 
   # x_y = vehicle_data[column][vehicle_data[column].notnull()]
    y = column #vehicle_data[column][vehicle_data[column].notnull()].to_list()
    y.sort()
    len_y = len(y)
    y = y[int(pct_lower * len_y):int(len_y * pct)]
    len_y = len(y)
    yy=([[x] for x in y])
    sc.fit(yy)
    y_std =sc.transform(yy)
    y_std = y_std.flatten()
    return y_std,len_y,y


def fit_distribution(column,pct,pct_lower):
    # Set up list of candidate distributions to use
    # See https://docs.scipy.org/doc/scipy/reference/stats.html for more
    y_std,size,y_org = standarise(column,pct,pct_lower)
    chi_square_statistics = []
    # 11 equi-distant bins of observed Data 
    percentile_bins = np.linspace(0,100,11)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        #print("{}\n{}\n".format(dist, param))


        # Get expected counts in percentile bins
        # cdf of fitted sistrinution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)

        ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square_statistics.append(ss)


    #Sort by minimum ch-square statistics
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square_statistics
    results.sort_values(['chi_square'], inplace=True)

    #print ('\nDistributions listed by Betterment of fit:')
    #print ('............................................')
    #print (results)
    #print(results[0])
    return results

def calculate_dis_props(dist_data):
    result = fit_distribution(dist_data, 0.99, 0.01)
    dist_name = result.iloc[0]['Distribution']

    distribution = pd.DataFrame()
    distribution['Type of Distribution'] = [dist_name]
    if dist_name == "lognorm":
        mean, var, skew, kurt = lognorm.stats(dist_data, moments='mvsk')
        distribution['Mean'] = [mean]
        distribution['Standard Deviation'] = [np.sqrt(var)]
        distribution['Skewness'] = [skew]
        distribution['Kurtosis'] = [kurt]
    elif dist_name == "gamma":
        mean, var, skew, kurt = gamma.stats(dist_data, moments='mvsk')
        distribution['Mean'] = [mean]
        distribution['Standard Deviation'] = [np.sqrt(var)]
        distribution['Skewness'] = [skew]
        distribution['Kurtosis'] = [kurt]
    elif dist_name == 'weibull_min':
        mean, var, skew, kurt = weibull_min.stats(dist_data, moments='mvsk')
        distribution['Mean'] = [mean]
        distribution['Standard Deviation'] = [np.sqrt(var)]
        distribution['Skewness'] = [skew]
        distribution['Kurtosis'] = [kurt]
    elif dist_name == 'norm':
        mean, var, skew, kurt = norm.stats(dist_data, moments='mvsk')
        distribution['Mean'] = [mean]
        distribution['Standard Deviation'] = [var]
        distribution['Skewness'] = [skew]
        distribution['Kurtosis'] = [kurt]
    elif dist_name == 'weibull_max':
        mean, var, skew, kurt = weibull_max.stats(dist_data, moments='mvsk')
        distribution['Mean'] = [mean]
        distribution['Standard Deviation'] = [var]
        distribution['Skewness'] = [skew]
        distribution['Kurtosis'] = [kurt]
    elif dist_name == "beta":
        mean, var, skew, kurt = beta.stats(dist_data,1, moments='mvsk')
        distribution['Mean'] = [mean]
        distribution['Standard Deviation'] = [var]
        distribution['Skewness'] = [skew]
        distribution['Kurtosis'] = [kurt]
    elif dist_name == 'invgauss':
        mean, var, skew, kurt = invgauss.stats(dist_data, moments='mvsk')
        distribution['Mean'] = [mean]
        distribution['Standard Deviation'] = [var]
        distribution['Skewness'] = [skew]
        distribution['Kurtosis'] = [kurt]
    elif dist_name == 'expon':
        mean, var, skew, kurt = expon.stats(dist_data, moments='mvsk')
        distribution['Mean'] = [mean]
        distribution['Standard Deviation'] = [var]
        distribution['Skewness'] = [skew]
        distribution['Kurtosis'] = [kurt]
    elif dist_name == 'uniform':
        mean, var, skew, kurt = uniform.stats(dist_data, moments='mvsk')
        distribution['Mean'] = [mean]
        distribution['Standard Deviation'] = [var]
        distribution['Skewness'] = [skew]
        distribution['Kurtosis'] = [kurt]
    elif dist_name == 'pearson3':
        mean, var, skew, kurt = pearson3.stats(dist_data, moments='mvsk')
        distribution['Mean'] = [mean]
        distribution['Standard Deviation'] = [np.sqrt(var)]
        distribution['Skewness'] = [skew]
        distribution['Kurtosis'] = [kurt]
    elif dist_name == 'triang':
        mean, var, skew, kurt = triang.stats(dist_data, moments='mvsk')
        distribution['Mean'] = [mean]
        distribution['Standard Deviation'] = [np.sqrt(var)]
        distribution['Skewness'] = [skew]
        distribution['Kurtosis'] = [kurt]  
    return distribution,result
# x_y,results = fit_distribution('year',0.99,0.01)
dist_names = ['weibull_min','norm','weibull_max','beta','invgauss','uniform','gamma','expon','lognorm','pearson3','triang']

# results_sort = results['chi_square']
# results_arr = np.array(results_sort)
# sort_results = sorted(results_arr)

# results.iloc[0]['chi_square']