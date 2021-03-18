#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-whiten data with ARIMA model before cross-correlation

Created on Wed Mar 17 11:45:37 2021

@author: lizz
"""

from netCDF4 import Dataset
from scipy import interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import iceutils as ice
import nifl_helper as nifl


# ### Define where the necessary data lives

# In[ ]:


flowline_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Felikson-flowlines/netcdfs/glaciera199.nc'
velocity_fpath='/Users/lizz/Documents/GitHub/Data_unsynced/Gld-Stack/'
gl_bed_fpath ='/Users/lizz/Documents/GitHub/Data_unsynced/BedMachine-Greenland/BedMachineGreenland-2017-09-20.nc'
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
xys = [(xh[i], yh[i]) for i in range(0, upstream_max, 20)][2::]

## Import and invert velocity observations

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


# ### Catchment-integrated SMB

# We load in a 1D timeseries of surface mass balance integrated over the whole Helheim catchment.  This data is monthly surface mass balance from the HIRHAM5 model, integrated over the Helheim catchment defined by K. Mankoff, with processing steps (coordinate reprojection, Delaunay triangulation, nearest-neighbor search and area summing) in `catchment-integrate-smb.py`.

# In[ ]:

## Read in RACMO monthly int from Denis
smb_fpath = '/Users/lizz/Documents/GitHub/helheim-fiesta/smb_rec._.BN_RACMO2.3p2_ERA5_3h_FGRN055.1km.MM.csv'
smb_racmo = pd.read_csv(smb_fpath, index_col=0, parse_dates=True)
smb_tr = smb_racmo.loc[smb_racmo.index.year >= 2006]
smb = smb_tr.loc[smb_tr.index.year <2018]

# ## normalize
# smb_normed = (smb-np.mean(smb)) / (np.std(smb)*len(smb))

# ## Prewhitening data: fit ARIMA model to SMB data
# lag_acf = sm.tsa.stattools.acf(smb_normed, nlags=20)
# lag_pacf = sm.tsa.stattools.pacf(smb_normed, nlags=20, method='ols')
# upper_confidence = 1.96/np.sqrt(len(smb_normed))
# p_candidates = np.argwhere(lag_pacf > upper_confidence).squeeze()
# p = p_candidates[p_candidates >0][0] # choose first nonzero value that exceeds upper CI
# q_candidates = np.argwhere(lag_acf > upper_confidence).squeeze()
# q = p_candidates[p_candidates >0][0] 
# d = 1 # order of differencing to apply in ARIMA model -- choose 1 for consistency with previous processing

# mod = sm.tsa.arima.ARIMA(smb_normed[' smb_rec (m3WE)'], order=(p,d,q), freq='M')
# mod_fit = mod.fit()

# smb_resid = pd.DataFrame(mod_fit.resid).squeeze()
# resid_d = [d.utctimetuple() for d in smb_resid.index]
# resid_d_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in resid_d]
# resid_func = interpolate.interp1d(resid_d_interp, smb_resid) # compute the residuals, used for xcorr
# # smb_resid.plot()

# ## Construct new filtered velocity series
# test_series = preds[5]['full']
# t_min = 2006
# t_max = 2018
# coincident_dates = np.asarray([t for t in t_grid if (t>=t_min and t<t_max)])
# vel_series_0 = test_series[np.where(t_grid>=t_min)]
# vel_series = vel_series_0[np.where(t_grid[np.where(t_grid>=t_min)]<t_max)] # trim dates to match t_limits
# coincident_resid = resid_func(coincident_dates) # sample at same dates as velocity series

# ## normalize series
# # resid_normd = (coincident_resid-np.mean(coincident_resid)) / (np.std(coincident_resid)*len(coincident_resid))
# ## don't normalize resid if already normalized SMB above
# vel_normd = (vel_series-np.mean(vel_series)) / (np.std(vel_series))

# v_filt = sm.tsa.filters.convolution_filter(vel_normd, filt=mod_fit.params)[1:-1]  #trim nans from ends
# # TODO: confirm whether this filters correctly

# whitened_corr = np.correlate(coincident_resid[1:-1], v_filt, mode='full')
# whitened_lags = range(int(-0.5*len(whitened_corr)), int(0.5*len(whitened_corr)+1))
# whitened_ci = [2/np.sqrt(len(coincident_resid)-abs(k)) for k in whitened_lags]


# ## plot to examine
# fig, ax = plt.subplots(1)
# ax.plot(whitened_lags, whitened_corr) 
# ax.plot(whitened_lags, whitened_ci, color='k', ls=':')
# ax.plot(whitened_lags, -1*np.array(whitened_ci), color='k', ls=':')
# ax.fill_between(whitened_lags, y1=whitened_corr, y2=0, where=abs(whitened_corr)>whitened_ci)
# plt.show()


def Arima_ResidFunc(series, normalize=True, diff=1):
    """
    Fit an ARIMA model of appropriate order to the input series, compute the 
    residuals we will use for the cross-correlation with velocity.

    Parameters
    ----------
    series : pd.Series
        The "input variable" (SMB, runoff, terminus position...) expressed as a
        one-dimensional time series.  The timestamp of each data point 
        should be the "index" of the pandas Series.
    normalize : bool, optional
        Whether to normalize series before fitting ARIMA. Default is True.
        Normalized residuals will make the output of XCorr1D inter-comparable 
        with normalized output for other variables.  If set to False, the signal 
        amplitude will be larger but the correlation values may exceed 1.
    diff : int, optional
        Number of discrete differences to apply to data. Default is 1.
        We do this to hone in on the effect of our variables on each other,
        rather than a mutual response (background trend) to a shared 
        un-modelled factor.

    Returns
    -------
    mod_fit: statsmodels.tsa.arima.model.ARIMAResultsWrapper
        The ARIMA model fit to the series, which we will use in filtering
        the velocity data for cross-correlation.
        (Ref: https://online.stat.psu.edu/stat510/lesson/9/9.1)
    resid_func : interpolate.interp1d
        1D-interpolated function with values of residuals (series-ARIMA) over time.
        We will use this to sample the residuals at the same time intervals
        as the velocity, which is necessary for the cross-correlation.
    series_resid : pd.Series , optional
        A DataFrame of the series residuals

    """
    if normalize:
        series_normed = (series-np.mean(series)) / (np.std(series)*len(series))
    else:
        series_normed = series
    
    ## Prewhitening data: fit ARIMA model to series data
    d = diff # order of differencing to apply in ARIMA model 
    ## partial autocorrelation function helps us choose the AR term (p) in ARIMA
    lag_pacf = sm.tsa.stattools.pacf(series_normed, nlags=20, method='ols')
    upper_confidence = 1.96/np.sqrt(len(series_normed))
    p_candidates = np.argwhere(lag_pacf > upper_confidence).squeeze()
    p = p_candidates[p_candidates >0][0] # choose first nonzero value that exceeds upper CI
    ## autocorrelation function helps us choose the MA term (q) in ARIMA
    lag_acf = sm.tsa.stattools.acf(series_normed, nlags=20) 
    q_candidates = np.argwhere(lag_acf > upper_confidence).squeeze()
    q = q_candidates[q_candidates >0][0] # choose first nonzero value that exceeds upper CI
    print('Fitting ARIMA of order ({},{},{})'.format(p,d,q))
    
    mod = sm.tsa.arima.ARIMA(series_normed, order=(p,d,q))
    mod_fit = mod.fit()

    series_resid = pd.DataFrame(mod_fit.resid).squeeze()
    resid_d = [d.utctimetuple() for d in series_resid.index]
    resid_d_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in resid_d]
    resid_func = interpolate.interp1d(resid_d_interp, series_resid)
    
    return mod_fit, resid_func, series_resid



    

def Xcorr1D_prewhitened(resid_func, series_dates, velocity_pred, t_grid, t_limits, 
            mod_fit, normalize=True, pos_only=False):
    """
    Compute cross-correlation on coincident series of a 1D time series
    (e.g. catchment-integrated runoff or SMB) versus velocity at a point.
    
    Note that unlike XCorr1D, we do not include the option to difference the 
    velocity time series here.  Rather, we apply a convolution filter defined
    by the parameters of the ARIMA model mod_fit.

    Parameters
    ----------
    resid_func : interpolate.interp1d
        1D-interpolated function with values of data series over time.
        We assume that this is already normalized.
    series_dates : list
        Decimal dates of data points
    velocity_series : dict
        Output of iceutils prediction.
    t_grid : ndarray
        Evenly spaced decimal times at which spline-fit velocity is sampled
    t_limits : tuple
        Start and end dates (decimal) of the time period to study
    mod_fit : statsmodels.tsa.arima.model.ARIMAResultsWrapper
        An ARIMA model fit to the input variable, which we will use in filtering
        the velocity data for cross-correlation.
        (Ref: https://online.stat.psu.edu/stat510/lesson/9/9.1)
    normalize : bool, optional
        Whether to normalize for a cross-correlation in [-1,1]. Default is True.
        This makes the output inter-comparable with normalized output for other
        variables.  If set to False, the signal amplitude will be larger but
        the correlation values may exceed 1.
    pos_only : bool, optional
    	Whether to analyse only xcorrs with positive lag values.  Default is False.
    	This allows a bidirectional causal relationship.  For a causal relationship 
    	hypothesised to be single-directional, choose True to display only positive
    	lag values.

    Returns
    -------
    corr : array
        Cross-correlation coefficients between SMB, velocity
    lags : array
        Time lag for each correlation value
    ci : array
        Confidence intervals for evaluation

    """
    t_min = max(min(series_dates), t_limits[0])
    t_max = min(max(series_dates), t_limits[1])
    coincident_dates = np.asarray([t for t in t_grid if (t>=t_min and t<t_max)])
    coincident_resid = resid_func(coincident_dates) # sample at same dates as velocity series
    
    vel_series_0 = velocity_pred['full'][np.where(t_grid>=t_min)]
    vel_series = vel_series_0[np.where(t_grid[np.where(t_grid>=t_min)]<t_max)] # trim dates to match t_limits
    if normalize:
        vel_series = (vel_series-np.mean(vel_series)) / (np.std(vel_series))
    v_filt = sm.tsa.filters.convolution_filter(vel_series, filt=mod_fit.params)[1:-1]  #trim nans from ends
    # TODO: confirm whether this whitens correctly

        
    corr = np.correlate(coincident_resid[1:-1], v_filt, mode='full')
    lags = range(int(-0.5*len(corr)), int(0.5*len(corr)+1))
    ci = [2/np.sqrt(len(coincident_resid)-abs(k)) for k in lags]

    ## convert lags to physical units
    lags = np.mean(np.diff(t_grid))*365.26*np.asarray(lags)
    
    if pos_only:
    	corr = corr[np.argwhere(lags>=0)].squeeze()
    	ci = np.asarray(ci)[np.argwhere(lags>=0)].squeeze()
    	lags = lags[np.argwhere(lags>=0)].squeeze()
    
    return corr, lags, ci



# In[ ]:
## Now test this pre-whitened approach along the flowline
mf, rf, ser_resid = Arima_ResidFunc(smb)
resid_d = [d.utctimetuple() for d in ser_resid.index]
resid_d_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in resid_d]

whitened_smb_corr_amax = []
whitened_smb_lag_amax = []

for xy, pred in zip(xys, preds):
    corr, lags, ci = Xcorr1D_prewhitened(resid_func=rf, series_dates=resid_d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              mod_fit=mf, normalize=True, pos_only=True)
    whitened_smb_corr_amax.append(corr[abs(corr).argmax()])
    whitened_smb_lag_amax.append(lags[abs(corr).argmax()])



# In[ ]:
## Test plot

## Read in and interpolate BedMachine topography
fh = Dataset(gl_bed_fpath, mode='r')
xx = fh.variables['x'][:].copy() #x-coord (polar stereo (70, 45))
yy = fh.variables['y'][:].copy() #y-coord
s_raw = fh.variables['surface'][:].copy() #surface elevation
h_raw=fh.variables['thickness'][:].copy() # Gridded thickness
b_raw = fh.variables['bed'][:].copy() # bed topo
thick_mask = fh.variables['mask'][:].copy()
ss = np.ma.masked_where(thick_mask !=2, s_raw)#mask values: 0=ocean, 1=ice-free land, 2=grounded ice, 3=floating ice, 4=non-Greenland land
hh = np.ma.masked_where(thick_mask !=2, h_raw) 
bb = b_raw #don't mask, to allow bed sampling from modern bathymetry (was subglacial in ~2006)
fh.close()


# In[ ]:


## Interpolate in area of Helheim
xl, xr = 6100, 6600
yt, yb = 12700, 13100
x_hel = xx[xl:xr]
y_hel = yy[yt:yb]
s_hel = ss[yt:yb, xl:xr]
b_hel = bb[yt:yb, xl:xr]
S_helheim = interpolate.RectBivariateSpline(x_hel, y_hel[::-1], s_hel.T[::,::-1]) #interpolating surface elevation provided
B_helheim = interpolate.RectBivariateSpline(x_hel, y_hel[::-1], b_hel.T[::,::-1]) #interpolating surface elevation provided

# ## Plotting




# First, we plot the max correlation at each point for a single variable.

# In[ ]:
ls = LightSource(azdeg=225, altdeg=80)

fig, ax = plt.subplots(1)
# ax.contourf(x_hel, y_hel, b_hel, cmap='gist_earth', alpha=0.5)
rgb = ls.shade(np.asarray(b_hel), cmap=plt.get_cmap('gist_earth'), blend_mode='overlay',
               dx=np.mean(np.diff(x_hel)), dy=np.mean(np.diff(y_hel)), vert_exag=5.)


div_colors = 'RdBu' # choose divergent colormap
corrnorm_min, corrnorm_max = -0.3, 0.3

ax.imshow(rgb, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc = ax.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=whitened_smb_corr_amax, cmap='RdBu', vmin=-0.5, vmax=0.5)
cb = fig.colorbar(sc, ax=ax)
cb.ax.set_title('Max. xcorr')
ax.set(xlim=(270000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
      xlabel='Easting [km]', ylabel='Northing [km]')
plt.show()