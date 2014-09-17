# -*- coding: utf-8 -*-
"""
Created on Mon Sep 08 10:35:11 2014

@author: c_cook
"""


import pandas as pd
import numpy as np
import csv
import string
import os
from tabulate import tabulate
import patsy
import statsmodels as smf
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from pandas import isnull
#------------------------------------------------
# STEP 1: CLEANING THE DATA
#------------------------------------------------
# read in main datasets
os.chdir('S:\General\Training\Ongoing Professional Development\Kaggle\Predicting car being lemon\EPC')
training = pd.read_csv('Raw_data/training.csv')
test = pd.read_csv('Raw_data/test.csv')
reliability = pd.read_csv('Raw_data/reliability_external.csv')

 # Exploratory   
training['PRIMEUNIT'].value_counts() # 3357 NO, 62 YES
training['AUCGUART'].value_counts() # 3340 GREEN, 79 RED
training['Make'].value_counts() # top five are Chevy, Dodge, Ford, Chrysler, Pontiac. Sharp drop off 
training['IsOnlineSale'].value_counts() # 71138 0s, 1845 1s \
training['VNZIP1'].value_counts() #lots, although some very concentrated ones. Top 10 are >1500
training['VNST'].value_counts() # most states. No VT or NH (or likely many others). Highest numbers in TX, FL, CA, NC and AZ
training['Auction'].value_counts() # 41043 Manheim, 14439 Adesa, 17501 other  -- this is the auction provider for the location where the car was purchased
training['BYRNO'].value_counts() # this var identifies unique buyer id. Some buyers bought a ton. 
training['BYRNO'].describe() # mean of 2414 with max of 3943 

# Function to clear. Will be applied to training and test
def clean(data): 
    data = pd.merge(data, reliability, how='left', on='Make')
    
    # Handling Missings & Coding Classifiers 
    # prime units -- presumable higher quality
    data['PRIMEUNIT'] = data['PRIMEUNIT'].replace("YES", 1)
    data['PRIMEUNIT'] = data['PRIMEUNIT'].replace("NO", 0)
    
    # warranty classification
    data.rename(columns={'AUCGUART': 'is_green_wrnty'}, inplace=True)
    data['is_green_wrnty'] = data['is_green_wrnty'].replace("GREEN", 1)
    data['is_green_wrnty'] = data['is_green_wrnty'].replace("RED", 0)
    
    # some of the transmissions were alternate cases
    data['Transmission'] = data['Transmission'].replace("Manual", "MANUAL")
    
    # add dummies for make, state, auction provider, transmission, color, size, vehicle year, wheel type and manufacturer nationality
    need_dums = ['Make', 'VNST', 'Auction', 'Transmission', 'Color', 'WheelType', 'Nationality', 'Size', 'VehYear']
    for d in need_dums: 
        dummies = pd.core.reshape.get_dummies(data[d])
        data = pd.concat([data,dummies], axis=1)
    
    # create variable for the number of cars that a buyer has purchased in this dataset
    data = data.merge(pd.DataFrame({'num_bought':data.groupby(['BYRNO']).size()}), left_on=['BYRNO'], right_index=True)

    # filling missing values with means, create dummy for missing
    missing_vars = ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'reliability_score', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice']

   
    for m in missing_vars: 
        data[m] = data.groupby('Make').transform(lambda x: x.fillna(x.median()))
        

    return data

# clean each of the datasets
training_clean = clean(training)
test_clean = clean(test)



#------------------------------------------------
# STEP 2: ANALYSIS 
#------------------------------------------------

# Variables to use (potentially) -- for dummies, one has already been taken out to avoid dummy var trap
continuous_vars = ['VehicleAge','VehOdo', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 'VehBCost', 'IsOnlineSale', 'WarrantyCost', 'reliability_score', 'num_bought']
make_cats = ['BUICK', 'CADILLAC', 'CHEVROLET', 'CHRYSLER', 'DODGE', 'FORD', 'GMC', 'HONDA', 'HUMMER', 'HYUNDAI', 'INFINITI', 'ISUZU', 'JEEP', 'KIA', 'LEXUS', 'LINCOLN', 'MAZDA', 'MERCURY', 'MINI', 'MITSUBISHI', 'NISSAN', 'OLDSMOBILE', 'PLYMOUTH', 'PONTIAC', 'SATURN', 'SCION', 'SUBARU', 'SUZUKI', 'TOYOTA', 'TOYOTA SCION', 'VOLKSWAGEN', 'VOLVO']
state_cats = ['AR', 'AZ', 'CA', 'CO', 'FL', 'GA', 'IA', 'ID', 'IL', 'IN', 'KY', 'LA', 'MA', 'MD', 'MI', 'MN', 'MO', 'MS', 'NC', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 'UT', 'VA', 'WA', 'WV']
auction_cats = ['ADESA', 'MANHEIM', 'OTHER']
trans_cats = ['AUTO']
color_cats = ['BEIGE', 'BLACK', 'BLUE', 'BROWN', 'GOLD', 'GREEN', 'GREY', 'MAROON', 'NOT AVAIL', 'ORANGE', 'OTHER', 'PURPLE', 'RED', 'SILVER', 'WHITE', 'YELLOW']
wheel_cats = ['Alloy', 'Covers', 'Special']
nat_cats = ['AMERICAN', 'OTHER', 'OTHER ASIAN', 'TOP LINE ASIAN']
size_cats =['COMPACT', 'CROSSOVER', 'LARGE', 'LARGE SUV', 'LARGE TRUCK', 'MEDIUM', 'MEDIUM SUV', 'SMALL SUV', 'SMALL TRUCK', 'SPECIALTY', 'SPORTS', 'VAN']
year_cats = ['2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010_x']

all_vars = continuous_vars + make_cats + state_cats + auction_cats + trans_cats + color_cats + wheel_cats + nat_cats + size_cats + year_cats
hashable = all_vars
## Ridge Regression 
ridge_reg = Ridge(alpha=1)

ridge_reg.fit(training_clean[all_vars], training['IsBadBuy'])
























