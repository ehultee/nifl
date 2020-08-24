## Script to compare SMB with velocity changes
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import pyproj as pyproj
from scipy import interpolate


##Load in HIRHAM
gl_smb_path = '/Users/lizz/Documents/GitHub/Data_unsynced/HIRHAM5-SMB/DMI-HIRHAM5_GL2_ERAI_1980_2016_SMB_MM.nc'
fh2 = Dataset(gl_smb_path, mode='r')
x_lon = fh2.variables['lon'][:].copy() #x-coord (latlon)
y_lat = fh2.variables['lat'][:].copy() #y-coord (latlon)
#zs = fh2.variables['height'][:].copy() #height in m - is this surface elevation or SMB?
ts = fh2.variables['time'][:].copy()
smb_raw = fh2.variables['smb'][:].copy()
fh2.close()

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
SMB_dict = {} #set up a dictionary of surface mass balance fields indexed by year
time_indices = range(311, 444) # go from Jan 2006 to Dec 2016 in monthly series
# time_indices = range(25, 34) # go from 2006-2014 in yearly series
smb_dates = pd.date_range(start='2006-01-01', end='2016-12-31', periods=len(time_indices))
for t,d in zip(time_indices, smb_dates):
    smb_t = smb_raw[t][0][yt1:yb1, xl1:xr1]
    regridded_smb_t = interpolate.griddata((xs.ravel(), ys.ravel()), smb_t.ravel(), (Xmat, Ymat), method='nearest')
    SMB_dict[d] = interpolate.interp2d(x_hel, y_hel, regridded_smb_t, kind='linear')   
    
## Plot SMB time series at points
xy_1 = (308103., -2577200.) #polar stereo coordinates of a point near Helheim 2009 terminus, in m
xy_2 = (302026., -2566770.) # point up on North branch
xy_3 = (297341., -2571490.) # point upstream on main branch
xy_4 = (294809., -2577580.) # point on southern tributary

smb_series = [float(SMB_dict[d](xy_1[0],xy_1[1])) for d in smb_dates]
fig, ax = plt.subplots(1)
ax.plot(smb_dates, smb_series)
plt.show()


## Now interpolate time series and pull values coincident with satellite shots
smb_d = [d.utctimetuple() for d in smb_dates]
dates_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in smb_d]
smb_series_func = interpolate.interp1d(dates_interp, smb_series, bounds_error=False)
coincident_dates = t_grid[t_grid<=max(dates_interp)]
coincident_smb = smb_series_func(coincident_dates) # sample at same dates as helheim-tseries_decomp
fig, ax = plt.subplots(1)
ax.plot(dates_grid[t_grid<=max(dates_interp)], coincident_smb)
plt.show()

## Compute cross-correlation with smooth preds[i] from helheim-tseries
fig, ax1 = plt.subplots(1)
lags, c, _, _ = ax1.xcorr(coincident_smb, preds[0]['full'][t_grid<=max(dates_interp)], usevlines=False, maxlags=200, normed=True, lw=2)
ax1.grid(True)
ax1.set(ylabel='Cross-correlation', xlabel='Lag [unit displacement of 3 days]')
