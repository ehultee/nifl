#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make stack plots of velocity, catchment SMB, catchment runoff, termini
Created on Fri Dec 18 11:38:22 2020

@author: lizz
"""


from netCDF4 import Dataset
from scipy import interpolate
import pyproj as pyproj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iceutils as ice
import nifl_helper as nifl


# ### Define where the necessary data lives

# In[ ]:


flowline_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Felikson-flowlines/netcdfs/glaciera199.nc'
velocity_fpath='/Users/lizz/Documents/GitHub/Data_unsynced/Gld-Stack/'
# gl_bed_fpath ='/Users/lizz/Documents/GitHub/Data_unsynced/BedMachine-Greenland/BedMachineGreenland-2017-09-20.nc'
catchment_smb_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Helheim-processed/HIRHAM_integrated_SMB.csv'
runoff_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Helheim-processed/RACMO2_3p2_Helheimgletscher_runoff_1958-2017.csv'
termini_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Helheim-processed/HLM_terminus_widthAVE.csv'

# ### Define the domain of analysis

# We will analyse along flowlines defined by Denis Felikson in his previous work, saved and shared as NetCDF files.  The flowlines are numbered 01-10 across the terminus; flowline 05 is close to the middle.  Note that Helheim Glacier has two large branches.  For now we'll study the main trunk, `glaciera199.nc`.  The more southerly trunk is `glacierb199.nc`.

# In[ ]:


ncfile = Dataset(flowline_fpath, 'r')
xh = ncfile['flowline05'].variables['x'][:]
yh = ncfile['flowline05'].variables['y'][:]
ncfile.close()


# In[ ]:


## Define points at which to extract
upstream_max = 500 # index of last xh,yh within given distance of terminus--pts roughly 50m apart
xys = [(xh[i], yh[i]) for i in range(0, upstream_max, 25)]

# ## Import and invert velocity observations

# In[ ]:

## Set up combined hdf5 stack
hel_stack = ice.MagStack(files=[velocity_fpath+'vx.h5', velocity_fpath+'vy.h5'])
data_key = 'igram' # B. Riel convention for access to datasets in hdf5 stack


# In[ ]:


# Create an evenly spaced time array for time series predictions
t_grid = np.linspace(hel_stack.tdec[0], hel_stack.tdec[-1], 1000)

# First convert the time vectors to a list of datetime
dates = ice.tdec2datestr(hel_stack.tdec, returndate=True)
dates_grid = ice.tdec2datestr(t_grid, returndate=True)

# Build the collection
collection = nifl.build_collection(dates)

# Construct a priori covariance
Cm = nifl.computeCm(collection)
iCm = np.linalg.inv(Cm)

# Instantiate a model for inversion
model = ice.tseries.Model(dates, collection=collection)

# Instantiate a model for prediction
model_pred = ice.tseries.Model(dates_grid, collection=collection)

## Access the design matrix for plotting
G = model.G

# Create lasso regression solver that does the following:
# i) Uses an a priori covariance matrix for damping out the B-splines
# ii) Uses sparsity-enforcing regularization (lasso) on the integrated B-splines
solver = ice.tseries.select_solver('lasso', reg_indices=model.itransient, penalty=0.05,
                                   rw_iter=1, regMat=iCm)


# Now that we are set up with our data and machinery, we'll ask the inversion to make us a continuous time series of velocity at each point we wish to study.



# In[ ]:


preds = []
for j, xy in enumerate(xys):
    try:
        pred, st, lt = nifl.VSeriesAtPoint(xy, vel_stack=hel_stack, collection=collection, 
                                  model=model, model_pred=model_pred, solver=solver, 
                                  t_grid=t_grid, sigma=1.5, data_key='igram')
        preds.append(pred)
    except AssertionError: # catches failed inversion
        print('Insufficient data for point {}. Removing'.format(j))
        xys.remove(xy)
        continue

    
# ## Comparison data sets


# ### Catchment-integrated SMB

# We load in a 1D timeseries of surface mass balance integrated over the whole Helheim catchment.  This data is monthly surface mass balance from the HIRHAM5 model, integrated over the Helheim catchment defined by K. Mankoff, with processing steps (coordinate reprojection, Delaunay triangulation, nearest-neighbor search and area summing) in `catchment-integrate-smb.py`.

# In[ ]:


smb = pd.read_csv(catchment_smb_fpath, parse_dates=[0])
smb_d = [d.utctimetuple() for d in smb['Date']]
smb_d_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in smb_d]
smb_func = interpolate.interp1d(smb_d_interp, smb['SMB_int'])

# Now, we compute the normalized cross-correlation between catchment-integrated SMB and surface velocity at each point along the flowline.  We will draw on the inverted velocity series saved in `preds` above.  We save the value of the maximum normalized cross-correlation, and the value in days of the lag where it occurs, to compare with other variables later.

# In[ ]:

smb_corr_amax = []
smb_lag_amax = []
smb_corr_plag1 = []
smb_lag_plag1 = []

for xy, pred in zip(xys, preds):
    corr, lags, ci = nifl.RunoffXcorr(xy, runoff_func=smb_func, runoff_dates=smb_d_interp, 
                              velocity_pred=pred, t_grid=t_grid, diff=1, normalize=True)
    smb_corr_amax.append(corr[abs(corr).argmax()])
    smb_lag_amax.append(lags[abs(corr).argmax()])


# In[ ]:


c = corr[lags>=0]
l = lags[lags>=0]

from scipy.signal import argrelextrema
idxs = np.asarray(argrelextrema(c, np.less)).squeeze();
cplag1 = c[idxs[0]]
lplag1 = l[idxs[0]]

print(lplag1)


# ### Runoff

# We import monthly runoff from the RACMO model, integrated over the Helheim catchment and shared as a CSV by Denis Felikson.  Because this data is catchment-integrated, we interpolate a single 1D time series that will be used at all points.

# In[ ]:


runoff = np.loadtxt(runoff_fpath, delimiter=',') 
rnf = runoff[runoff[:,0]>=2006] # trim values from before the start of the velocity series
rf = rnf[rnf[:,0]<=2016] #trim values after the end of the velocity series

runoff_dates = pd.date_range(start='2006-01-01', end='2016-12-01', periods=len(rf))
runoff_d = [d.utctimetuple() for d in runoff_dates]
d_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in runoff_d]
runoff_func = interpolate.interp1d(d_interp, rf[:,2])


# We compute the normalized cross-correlation between catchment-integrated runoff and surface velocity at each same point.  Again we save the value of the maximum normalized cross-correlation, and the value in days of the lag where it occurs, to compare with other variables.

# In[ ]:


runoff_corr_amax = []
runoff_lag_amax = []
for xy, pred in zip(xys, preds):
    corr, lags, ci = nifl.RunoffXcorr(xy, runoff_func=runoff_func, runoff_dates=d_interp, 
                              velocity_pred=pred, t_grid=t_grid, diff=1, normalize=True)
    runoff_corr_amax.append(corr[abs(corr).argmax()])
    runoff_lag_amax.append(lags[abs(corr).argmax()])


# ### Terminus position change

# We import width-averaged terminus position change processed by Leigh Stearns.  These data give terminus position in km from a baseline, so they do not need to be processed into a coordinate system.

# In[ ]:


termini = pd.read_csv(termini_fpath, parse_dates=True, usecols=[0,1])
termini['date'] = pd.to_datetime(termini['date'])
trmn = termini.loc[termini['date'].dt.year >= 2006]
tm = trmn.loc[trmn['date'].dt.year <=2016]

termini_d = [d.utctimetuple() for d in tm['date']]
tm_d_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in termini_d]
termini_func = interpolate.interp1d(tm_d_interp, tm['term_km'])


# In[ ]:


terminus_corr_amax = []
terminus_lag_amax = []
for xy, pred in zip(xys, preds):
    corr, lags, ci = nifl.RunoffXcorr(xy, runoff_func=termini_func, runoff_dates=tm_d_interp, 
                              velocity_pred=pred, t_grid=t_grid, diff=1, normalize=True)
    terminus_corr_amax.append(corr[abs(corr).argmax()])
    terminus_lag_amax.append(lags[abs(corr).argmax()])


# In[]:
# ## Plot point velocity against SMB, runoff, terminus position change

series = [hel_stack.timeseries(xy=xyi, key=data_key) for xyi in xys]
idx_to_plot = 5
t_end = min(tm_d_interp[-1], smb_d_interp[-1], d_interp[-1])
t_idx_end = np.argwhere(t_grid<t_end)[-1].squeeze()
t_grid_plot = t_grid[:t_idx_end]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(6, 16), sharex=True)
ax1.scatter(hel_stack.tdec, series[idx_to_plot])
ax1.plot(t_grid_plot, preds[idx_to_plot]['full'][:t_idx_end], label='Ice surface speed')
ax2.scatter(smb_d_interp, smb['SMB_int'])
ax2.plot(t_grid_plot, smb_func(t_grid_plot), label='SMB')
ax3.scatter(d_interp, rf[:,2])
ax3.plot(t_grid_plot, runoff_func(t_grid_plot), label='Runoff')
ax4.scatter(tm_d_interp, tm['term_km'])
ax4.plot(t_grid_plot, termini_func(t_grid_plot), label='Terminus position')
ax1.set(xlim=(2009, 2017), ylabel='Ice surface speed [km/a]')
ax2.set(ylabel='Catchment SMB [m3]')
ax3.set(ylabel='Catchment runoff [m3]')
ax4.set(ylabel='Terminus position [km]')
plt.savefig('/Users/lizz/Desktop/20201218-tseries_stack_pt5.png')
