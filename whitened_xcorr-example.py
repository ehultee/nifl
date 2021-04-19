#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use prewhitening functions in nifl_helper to recompute all xcorrs

Created on Thu Mar 18 17:29:30 2021

@author: lizz
"""
from netCDF4 import Dataset
from scipy import interpolate
import pyproj as pyproj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import iceutils as ice
import nifl_helper as nifl


# ### Define where the necessary data lives

# In[ ]:


flowline_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Felikson-flowlines/netcdfs/glaciera199.nc'
velocity_fpath='/Users/lizz/Documents/GitHub/Data_unsynced/Gld-Stack/'
gl_bed_fpath ='/Users/lizz/Documents/GitHub/Data_unsynced/BedMachine-Greenland/BedMachineGreenland-2017-09-20.nc'
catchment_smb_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Helheim-processed/smb_rec._.BN_RACMO2.3p2_ERA5_3h_FGRN055.1km.MM.csv'
runoff_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Helheim-processed/runoff._.BN_RACMO2.3p2_ERA5_3h_FGRN055.1km.MM.csv'
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
xys = [(xh[i], yh[i]) for i in range(0, upstream_max, 20)][2::]

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
                                  t_grid=t_grid, sigma=2.5, data_key='igram')
        preds.append(pred)
    except AssertionError: # catches failed inversion
        print('Insufficient data for point {}. Removing'.format(j))
        xys.remove(xy)
        continue


# ## Comparison data sets

# In[ ]:
# ### Catchment-integrated runoff

## Read in RACMO monthly int from Denis
runoff_racmo = pd.read_csv(runoff_fpath, index_col=0, parse_dates=True)
runoff_tr = runoff_racmo.loc[runoff_racmo.index.year >= 2006]
runoff = runoff_tr.loc[runoff_tr.index.year <2018]   

runoff_mf, runoff_rf, runoff_resid = nifl.Arima_ResidFunc(runoff)
resid_d = [d.utctimetuple() for d in runoff_resid.index]
resid_d_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in resid_d]

whitened_runoff_corr_amax = []
whitened_runoff_lag_amax = []
for xy, pred in zip(xys, preds):
    corr, lags, ci = nifl.Xcorr1D_prewhitened(resid_func=runoff_rf, series_dates=resid_d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              mod_fit=runoff_mf, normalize=True, pos_only=True)
    whitened_runoff_corr_amax.append(corr[abs(corr).argmax()])
    whitened_runoff_lag_amax.append(lags[abs(corr).argmax()])


# In[ ]:
# ### Catchment-integrated SMB
# We load in a 1D timeseries of surface mass balance integrated over the whole Helheim catchment.  This data is monthly surface mass balance from the HIRHAM5 model, integrated over the Helheim catchment defined by K. Mankoff, with processing steps (coordinate reprojection, Delaunay triangulation, nearest-neighbor search and area summing) in `catchment-integrate-smb.py`.

## Read in RACMO monthly int from Denis
smb_racmo = pd.read_csv(catchment_smb_fpath, index_col=0, parse_dates=True)
smb_tr = smb_racmo.loc[smb_racmo.index.year >= 2006]
smb = smb_tr.loc[smb_tr.index.year <2018] 

smb_mf, smb_rf, smb_resid = nifl.Arima_ResidFunc(smb)
resid_d = [d.utctimetuple() for d in smb_resid.index]
resid_d_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in resid_d]

whitened_smb_corr_amax = []
whitened_smb_lag_amax = []
for xy, pred in zip(xys, preds):
    corr, lags, ci = nifl.Xcorr1D_prewhitened(resid_func=smb_rf, series_dates=resid_d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              mod_fit=smb_mf, normalize=True, pos_only=True)
    whitened_smb_corr_amax.append(corr[abs(corr).argmax()])
    whitened_smb_lag_amax.append(lags[abs(corr).argmax()])
    

# In[ ]:
# ### Terminus position change
# We import width-averaged terminus position change processed by Leigh Stearns.  These data give terminus position in km from a baseline, so they do not need to be processed into a coordinate system.


termini = pd.read_csv(termini_fpath, index_col=0, parse_dates=True, usecols=[0,1])
trmn = termini.loc[termini.index.year >= 2006]
tm = trmn.loc[trmn.index.year <2017]

## smooth to make more comparable with SMB and runoff--but not too much for ARIMA
td = tm.rolling('10D').mean()
# td1 = td.resample('M').mean()

##
termini_mf, termini_rf, termini_resid = nifl.Arima_ResidFunc(td)
# termini_resid.plot()

termini_d = [d.utctimetuple() for d in tm.index]
tm_d_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in termini_d]

whitened_term_corr_amax = []
whitened_term_lag_amax = []
for xy, pred in zip(xys, preds):
    corr, lags, ci = nifl.Xcorr1D_prewhitened(resid_func=termini_rf, series_dates=tm_d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              mod_fit=termini_mf, normalize=True, pos_only=True)
    whitened_term_corr_amax.append(corr[abs(corr).argmax()])
    whitened_term_lag_amax.append(lags[abs(corr).argmax()])


# In[ ]:
    ## plotting
ls = LightSource(azdeg=225, altdeg=80)

    
div_colors = 'RdBu' # choose divergent colormap for xcorr
lag_colors = 'PiYG' # choose divergent colormap for lag
corrnorm_min, corrnorm_max = -0.3, 0.3
lagnorm_min, lagnorm_max = -365, 365

## set matplotlib font size defaults
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

## black-white hillshade topo underneath
rgb2 = ls.shade(np.asarray(b_hel), cmap=plt.get_cmap('gray'), blend_mode='overlay',
               dx=np.mean(np.diff(x_hel)), dy=np.mean(np.diff(y_hel)), vert_exag=5.)

fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(nrows=2,ncols=3, figsize=(12, 8), 
                                                     # constrained_layout=True, 
                                                     sharex=True, sharey=True,
                                                     gridspec_kw={'wspace':0.01})
    
ax1.imshow(rgb2, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc1 = ax1.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=whitened_smb_corr_amax, cmap=div_colors,
                 vmin=corrnorm_min, vmax=corrnorm_max)
# ## set up correctly scaled colorbar
# div1 = make_axes_locatable(ax1)
# cax1 = div1.append_axes("right", size="5%", pad=0.1)
# plt.colorbar(sc1, cax=cax1)
# cb1.ax.set_title('AMax. xcorr')
ax1.set(xlim=(278000, 320000), xticks=(280000, 300000, 320000), 
        ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
       ylabel='Northing [km]', title='Catchment SMB')

ax2.imshow(rgb2, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc2 = ax2.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=whitened_runoff_corr_amax, cmap=div_colors,
                 vmin=corrnorm_min, vmax=corrnorm_max)
# ## set up correctly scaled colorbar
# div2 = make_axes_locatable(ax2)
# cax2 = div2.append_axes("right", size="5%", pad=0.1)
# fig.colorbar(sc2, cax=cax2)
# cb2.ax.set_title('AMax. xcorr')
ax2.set(xlim=(278000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
       title='Catchment runoff')

ax3.imshow(rgb2, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc3 = ax3.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=whitened_term_corr_amax, cmap=div_colors,
                 vmin=corrnorm_min, vmax=corrnorm_max)
## set up correctly scaled colorbar - one for all xcorr plots
div3 = make_axes_locatable(ax3)
cax3 = div3.append_axes("right", size="5%", pad=0.1)
cb3 = fig.colorbar(sc3, cax=cax3)
cb3.ax.set_ylabel('AMax. xcorr')
ax3.set(xlim=(278000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
       title='Terminus position', aspect=1.)

## SECOND ROW
ax4.imshow(rgb2, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc4 = ax4.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=whitened_smb_lag_amax, cmap=lag_colors,
                  vmin=lagnorm_min, vmax=lagnorm_max)
# ## set up correctly scaled colorbar
# div4 = make_axes_locatable(ax4)
# cax4 = div4.append_axes("right", size="5%", pad=0.1)
# plt.colorbar(sc4, cax=cax4)
# cb1.ax.set_title('Lag [d] at peak xcorr')
ax4.set(xlim=(278000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
      xlabel='Easting [km]', ylabel='Northing [km]')

ax5.imshow(rgb2, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc5 = ax5.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=whitened_runoff_lag_amax, cmap=lag_colors,
                  vmin=lagnorm_min, vmax=lagnorm_max)
# ## set up correctly scaled colorbar
# div5 = make_axes_locatable(ax5)
# cax5 = div5.append_axes("right", size="5%", pad=0.1)
# fig.colorbar(sc5, cax=cax5)
# cb2.ax.set_title('Lag [d] at peak xcorr')
ax5.set(xlim=(278000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
      xlabel='Easting [km]')

ax6.imshow(rgb2, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc6 = ax6.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=whitened_term_lag_amax, cmap=lag_colors,
                  vmin=lagnorm_min, vmax=lagnorm_max)
## set up correctly scaled colorbar
div6 = make_axes_locatable(ax6)
cax6 = div6.append_axes("right", size="5%", pad=0.1)
cb6 = fig.colorbar(sc6, cax=cax6)
cb6.ax.set_ylabel('Lag [d] at peak xcorr')
ax6.set(xlim=(278000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
      xlabel='Easting [km]', aspect=1.)
# plt.tight_layout()
# plt.show()
plt.savefig('/Users/lizz/Desktop/20210204-helheim-xcorr_lag_composite')