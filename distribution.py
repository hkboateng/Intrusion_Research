# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:38:35 2021

@author: Hubert Kyeremateng-Boateng
"""


import pandas as pd
import numpy as np
# from mlutils import dataset, connector
import scipy.stats
from scipy.stats import triang
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statistics
from fitter import Fitter
def standardise(column,pct,pct_lower):
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


def fit_distribution(dataset,pct,pct_lower):

    # Set up list of candidate distributions to use
    # See https://docs.scipy.org/doc/scipy/reference/stats.html for more
    y_std,size,y_org = standardise(dataset,pct,pct_lower)
    chi_square_statistics = []
    # 11 equi-distant bins of observed Data 
    percentile_bins = np.linspace(0,100,11)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    percentile_cutoffs.sort()
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
    return results

def calculate_dis_props(dist_data, distribution):
    f = Fitter(dist_data, distributions=dist_names)
    f.fit()
    results = f.summary()
    dist_name = results.iloc[0].name

    if dist_name == "lognorm":
        '''
        https://www.coursera.org/lecture/compstatsintro/lognormal-distribution-DSdi9
        https://brilliant.org/wiki/log-normal-distribution/
        The term "log-normal" comes from the result of taking the logarithm of both sides:

            \log X = \mu +\sigma Z.
            logX=μ+σZ.

        As Z is normal, \mu+\sigma Zμ+σZ is also normal (the transformations just scale the distribution, 
        and do not affect normality), meaning that the logarithm of XX is normally distributed 
        (hence the term log-normal).
        '''
        min_val = np.min(dist_data)
        small_val = 1e-5
        epsilon = np.abs(min_val) + small_val
        dist_data = dist_data + epsilon
        mean_x = np.mean(np.log(dist_data))
        std_x = np.std(np.log(dist_data))
        m = np.log(mean_x**2/(np.sqrt(mean_x**2+std_x**2)))
        s = np.log(1+(std_x**2 / mean_x**2))
        distribution = distribution.append({'Type of Distribution':dist_name,'Mean': m,'Standard Deviation': s,'Epsilon' : epsilon,'Kurtosis' :min_val},ignore_index=True)
    
    elif dist_name == 'norm':
        distribution = distribution.append({'Type of Distribution':dist_name,'Mean': np.mean(dist_data),'Standard Deviation':np.std(dist_data),'Skewness' : 0,'Kurtosis' :0},ignore_index=True)
    
    elif dist_name == 'expon':
        '''
        Exponential Distribution:
            Reference:
                https://courses.lumenlearning.com/introstats1/chapter/the-exponential-distribution/
            The exponential distribution is often concerned with the amount of time until some specific event occurs.
            The exponential distribution is widely used in the field of reliability. Reliability deals with the amount of time a product lasts.
            The standard deviation, σ, is the same as the mean. μ = σ
        '''
        lamda = len(dist_data)
        mean = 1/lamda
        sigma = np.sqrt(1/lamda**2)
        distribution = distribution.append({'Type of Distribution':dist_name,'Mean': mean,'Standard Deviation':sigma, 'Lambda':lamda},ignore_index=True)
    elif dist_name == 'uniform':
        max_val = np.max(dist_data)
        min_val = np.min(dist_data)
        mean = (max_val + min_val)/2
        sigma = np.sqrt(((min_val - max_val)**2)*(1/12))
        distribution = distribution.append({'Type of Distribution':dist_name,'Mean': mean,'Standard Deviation': sigma,'Min' : min_val,'Max' :max_val},ignore_index=True)
    elif dist_name == 'triang':
        '''
        Refence:
        https://www.statisticshowto.com/triangular-distribution/
        '''
        peak_val = statistics.mode(dist_data) #c: Redo  use Python function for mode
        max_val = np.amax(dist_data) #b
        min_val = np.amin(dist_data) #a
        mean = (max_val + peak_val + min_val)/3
        sigma = (1/np.sqrt(6))*max_val

        #print('Comparing Triangular dist standard deviation: {0} : {1}'.format(sig, sigma))
        distribution = distribution.append({'Type of Distribution':dist_name,'Mean': mean,'Standard Deviation':sigma,'Min' : min_val,'Max' : max_val, 'Peak': peak_val},ignore_index=True)
    return distribution

dist_names = ['expon','uniform','triang','norm','lognorm']

def  method_stats(dist_data):
    distribution = pd.DataFrame(columns=('Type of Distribution','Mean','Standard Deviation','Epsilon','Max','Min','Peak','Lambda'))
    for i in dist_data:
        distribution = calculate_dis_props(i, distribution)
    return distribution


'''
To test distributions
'''
def test_distribution_function(dist, distribution_names):
    
    dist_data_txt = dist + "_data.txt";
    dist_data = np.loadtxt(dist_data_txt)
    
    f = Fitter(dist_data, distributions=distribution_names)
    f.fit()
    results = f.summary()

    assert results.iloc[0].name == dist, "Distribution name {0}  is not correct".format(results.iloc[0].name)


def dist_function(dist_list):
    
    for dist in dist_list:
        test_distribution_function(dist, dist_list)

# distribution_names = ['gamma','norm','lognorm','triang']
# dist_function(distribution_names)

def distribution_test():
    num_samples = 1
    for dist in dist_names:
        sample_list = []
        print('--'*40)
        dist_obj = getattr(scipy.stats, dist)
        for i in range(num_samples):
            if dist == 'norm':
                data = dist_obj.rvs(size=100000)
            elif dist == 'lognorm':
                data = dist_obj.rvs(0.158, size=100000)
            elif dist == 'triang':
                data = dist_obj.rvs(0.158, size=100000)
            elif dist == 'expon':
                data = dist_obj.rvs(size=100000)
            elif dist == 'uniform':
                data = dist_obj.rvs(size=100000)
            test_generated_distribution(dist,data)
            print('-*-'*20)
            sample_list.append(data)


def test_generated_distribution(distribution_name, stats_sample ):
    results = fit_distribution(stats_sample, 0.99, 0.01)
    print('---------------------- {0} ---------------'.format(distribution_name))
    print('Distribution for dataset is {} and fit_distribution function returned {}'.format(distribution_name,results.iloc[0]['Distribution']))
    print("-"*40)
    f = None
    f = Fitter(stats_sample, distributions=dist_names)
    f.fit()
    result = f.summary()
    print('Distribution for dataset is {} and Fitter module returned {}'.format(distribution_name,result.iloc[0].name))

#sample_list = distribution_test()

