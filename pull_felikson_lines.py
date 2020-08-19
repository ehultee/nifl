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