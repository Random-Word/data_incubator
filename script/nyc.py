#!/usr/bin/env python3

import pandas as pd
import numpy as np
import dateutil as du

DATA_PATH = '../data/rows.csv'
R = 6371 #Radius of the Earth in km

def lat_lon_dist(lat1, lon1, lat2, lon2):
    """Use Haverton's equation to compute distance between
    Lat/Long points in km."""
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    dlat = lat1-lat2
    dlon = lon1-lon2
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin(dlon)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R*c
    return d
"""
Q1_data = pd.read_csv(DATA_PATH, usecols=['Agency'])
counts = Q1_data['Agency'].value_counts()
print("Q1:")
print(counts[1]/counts.sum())

Q2_data = pd.read_csv(DATA_PATH, usecols=['Latitude', 'Longitude'])
print("Q2:")
print(Q2_data.quantile(0.9,axis=0)-Q2_data.quantile(0.1,axis=0))

print("Q3:")
mean_lat = Q2_data['Latitude'].mean()
lat_std = Q2_data['Latitude'].std()
mean_lon = Q2_data['Longitude'].mean()
lon_std = Q2_data['Longitude'].std()

lat_dist = lat_lon_dist(mean_lat+lat_std, mean_lon, mean_lat-lat_std,
        mean_lon)/2.0
lon_dist = lat_lon_dist(mean_lat, mean_lon+lon_std, mean_lat,
        mean_lon-lon_std)/2.0

print(np.pi*lon_dist*lat_dist)
"""

Q3_data = pd.read_csv(DATA_PATH,  usecols=['Created Date'])
print(Q3_data.describe())
test = Q3_data['Created Date'][0:1000]
test = pd.DatetimeIndex(test)
print(test.hour)
print(test.hour.value_counts())
print(test.describe())

