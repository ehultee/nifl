#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:39:25 2020
Test load-in of Felikson flowlines

@author: lizz
"""

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import iceutils as ice

## Pull single flowline from Denis Felikson
fp1 = '/Users/lizz/Documents/GitHub/Data_unsynced/Felikson-flowlines/netcdfs/glaciera199.nc'
ncfile = Dataset(fp1, 'r')
xh = ncfile['flowline05'].variables['x'][:]
yh = ncfile['flowline05'].variables['y'][:]
s = ncfile['flowline05']['geometry']['surface']['GIMP']['nominal'].variables['h'][:] # GIMP DEM
b = ncfile['flowline05']['geometry']['bed']['BedMachine']['nominal'].variables['h'][:] # BedMachine v3
dh = ncfile['flowline05']['dh']['GIMP-Arctic']['nominal']['dh'][:]
# ncfile.close()

fp2 = '/Users/lizz/Documents/GitHub/Data_unsynced/Felikson-flowlines/netcdfs/glacierb199.nc'
ncfile2 = Dataset(fp2, 'r')
xh2 = ncfile2['flowline05'].variables['x'][:]
yh2 = ncfile2['flowline05'].variables['y'][:]
s2 = ncfile2['flowline05']['geometry']['surface']['GIMP']['nominal'].variables['h'][:] # GIMP DEM
b2 = ncfile2['flowline05']['geometry']['bed']['BedMachine']['nominal'].variables['h'][:] # BedMachine v3
dh2 = ncfile2['flowline05']['dh']['GIMP-Arctic']['nominal']['dh'][:]
# ncfile2.close()

## Set up combined hdf5 stack
fpath='/Users/lizz/Documents/GitHub/Data_unsynced/Gld-Stack/'
hel_stack = ice.MagStack(files=[fpath+'vx.h5', fpath+'vy.h5'])
data_key = 'igram' # B. Riel convention for access to datasets in hdf5 stack

## Extract time series at selected points
upstream_max = 500 # index of last xh,yh within given distance of terminus--pts roughly 50m apart
xys = [(xh[i], yh[i]) for i in range(0, upstream_max, 100)]
labels=[str(i) for i in range(len(xys))]
series = [hel_stack.timeseries(xy=xyi, key=data_key) for xyi in xys]
