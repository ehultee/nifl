#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:18:30 2020
Compare surface snowmelt with velocity changes

@author: lizz
"""
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import pyproj as pyproj
from scipy import interpolate


##Load in HIRHAM monthly melt
gl_melt_path = '/Users/lizz/Documents/GitHub/Data_unsynced/HIRHAM5-SMB/DMI-HIRHAM5_GL2_ERAI_1980_2016_SNMELT_MM.nc'
fh3 = Dataset(gl_melt_path, mode='r')
x_lon = fh3.variables['lon'][:].copy() #x-coord (latlon)
y_lat = fh3.variables['lat'][:].copy() #y-coord (latlon)
#zs = fh3.variables['height'][:].copy() #height in m - is this surface elevation or SMB?
ts = fh3.variables['time'][:].copy()
melt_raw = fh3.variables['snmelt'][:].copy()
fh3.close()

## Select Helheim
xl1, xr1 = 190, 260
yt1, yb1 = 345, 405
x_lon_h = x_lon[yt1:yb1, xl1:xr1]
y_lat_h = y_lat[yt1:yb1, xl1:xr1]

wgs84 = pyproj.Proj("+init=EPSG:4326") # LatLon with WGS84 datum used by GPS units and Google Earth
psn_gl = pyproj.Proj("+init=epsg:3413") # Polar Stereographic North used by BedMachine (as stated in NetCDF header)
xs, ys = pyproj.transform(wgs84, psn_gl, x_lon_h, y_lat_h)
Xmat, Ymat = np.meshgrid(x_hel, y_hel) # BedMachine coords from helheim-profiles

## Timeslice-specific SMB functions 2006-2014
melt_dict = {} #set up a dictionary of surface mass balance fields indexed by year
time_indices = range(311, 444) # go from Jan 2006 to Dec 2016 in monthly series
# time_indices = range(25, 34) # go from 2006-2014 in yearly series
melt_dates = pd.date_range(start='2006-01-01', end='2016-12-31', periods=len(time_indices))
for t,d in zip(time_indices, melt_dates):
    melt_t = melt_raw[t][0][::-1, ::][yt1:yb1, xl1:xr1]
    regridded_melt_t = interpolate.griddata((xs.ravel(), ys.ravel()), melt_t.ravel(), (Xmat, Ymat), method='nearest')
    melt_dict[d] = interpolate.interp2d(x_hel, y_hel, regridded_melt_t, kind='linear')   
    
## Pull SMmeltB time series at point coincident with velocity pull
xy_1 = (308103., -2577200.) #polar stereo coordinates of a point near Helheim 2009 terminus, in m
xy_2 = (302026., -2566770.) # point up on North branch
xy_3 = (297341., -2571490.) # point upstream on main branch
xy_4 = (294809., -2577580.) # point on southern tributary

melt_series = [float(melt_dict[d](xy_4[0],xy_4[1])) for d in melt_dates]
fig, ax = plt.subplots(1)
ax.plot(melt_dates, melt_series)
plt.show()

## Now interpolate time series and pull values coincident with satellite shots
smb_d = [d.utctimetuple() for d in smb_dates]
dates_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in smb_d]
smb_series_func = interpolate.interp1d(dates_interp, smb_series, bounds_error=False)
coincident_dates = t_grid[t_grid<=max(dates_interp)]
coincident_smb = smb_series_func(coincident_dates) # sample at same dates as helheim-tseries_decomp
# fig, ax = plt.subplots(1)
# ax.plot(dates_grid[t_grid<=max(dates_interp)], coincident_smb)
# plt.show()

## Compute cross-correlation with smooth preds[i] from helheim-tseries
fig, ax1 = plt.subplots(1)
lags, c, _, _ = ax1.xcorr(coincident_smb, preds[0]['full'][t_grid<=max(dates_interp)], usevlines=False, maxlags=None, normed=True, lw=2)
ci = [2/np.sqrt(len(coincident_smb)-abs(k)) for k in lags]
ax1.plot(lags, ci, ls=':', color='k')
ax1.plot(lags, -1*np.array(ci), ls=':', color='k')
ax1.grid(True)
ax1.set(ylabel='Cross-correlation', xlabel='Lag [unit displacement of 3 days]')

plt.figure()
plt.imshow(smb_raw[311][0])
plt.scatter(xl1, yt1)
plt.scatter(xl1, yb1)
plt.scatter(xr1, yt1)
plt.scatter(xr1, yb1)
plt.show()
