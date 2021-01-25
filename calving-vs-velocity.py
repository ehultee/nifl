#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare calving-related quantities against velocity variability

Created on Thu Sep  3 16:56:15 2020

@author: lizz
"""

import numpy as np
import pandas as pd
import csv
import iceutils as ice
from scipy import interpolate

# fn1 = '/Users/lizz/GitHub/Data_unsynced/Helheim-processed/HLM_terminus_widthAVE.csv'
# termini = pd.read_csv(fn1, parse_dates=True, usecols=[0,1])
# termini['date'] = pd.to_datetime(termini['date'])
# trmn = termini.loc[termini['date'].dt.year >= 2006]
# tm = trmn.loc[trmn['date'].dt.year <=2016]

# termini_d = [d.utctimetuple() for d in tm['date']]
# tm_d_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in termini_d]
# termini_func = interpolate.interp1d(tm_d_interp, tm['term_km'])

## Test low-frequency variability
def Xcorr1D_lt(pt, series_func, series_dates, velocity_pred, t_grid, t_limits, diff=1, normalize=True):
    """
    Compute cross-correlation on coincident series of a 1D time series
    (e.g. catchment-integrated runoff or SMB) versus velocity at a point.

    Parameters
    ----------
    pt : tuple
        Position (x,y) at which to pull velocity series.
    series_func : interpolate.interp1d
        1D-interpolated function with values of data series over time.
    series_dates : list
        Decimal dates of data points
    velocity_series : dict
        Output of iceutils prediction.
    t_grid : ndarray
        Evenly spaced decimal times at which spline-fit velocity is sampled
    t_limits : tuple
        Start and end dates (decimal) of the time period to study
    diff : int, optional
        Number of discrete differences to apply to data. Default is 1.
        Setting diff=0 will process the input data as-is.
    normalize : bool, optional
        Whether to normalize for a cross-correlation in [-1,1]. Default is True.
        This makes the output inter-comparable with normalized output for other
        variables.  If set to False, the signal amplitude will be larger but
        the correlation values may exceed 1.

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
    coincident_series = series_func(coincident_dates) # sample at same dates as velocity series
    
    vel_longterm = velocity_pred['secular'] + velocity_pred['transient']
    series_diff = np.diff(coincident_series, n=diff)
    vel_series_0 = vel_longterm[np.where(t_grid>=t_min)]
    vel_series = vel_series_0[np.where(t_grid[np.where(t_grid>=t_min)]<t_max)] # trim dates to match t_limits
    vel_diff = np.diff(vel_series, n=diff)
    if normalize:
        series_diff = (series_diff-np.mean(series_diff)) / (np.std(series_diff)*len(series_diff))
        vel_diff = (vel_diff-np.mean(vel_diff)) / (np.std(vel_diff))
    corr = np.correlate(series_diff, vel_diff, mode='full')
    lags = range(int(-0.5*len(corr)), int(0.5*len(corr)+1))
    ci = [2/np.sqrt(len(coincident_series)-abs(k)) for k in lags]

    ## convert lags to physical units
    lags = np.mean(np.diff(t_grid))*365.26*np.asarray(lags)
    
    return corr, lags, ci

tf_spl = interpolate.UnivariateSpline(tm_d_interp, tm['term_km']) #fit to multi-annual trend in terminus pos.

term_lt_corr_amax = []
term_lt_lag_amax = []
for xy, pred in zip(xys, preds):
    corr, lags, ci = Xcorr1D_lt(xy, series_func=tf_spl, series_dates=tm_d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              diff=1, normalize=True)
    term_lt_corr_amax.append(corr[abs(corr).argmax()])
    term_lt_lag_amax.append(lags[abs(corr).argmax()])

ls = LightSource(azdeg=225, altdeg=80)

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
# ax.contourf(x_hel, y_hel, b_hel, cmap='gist_earth', alpha=0.5)
rgb = ls.shade(np.asarray(b_hel), cmap=plt.get_cmap('gist_earth'), blend_mode='overlay',
               dx=np.mean(np.diff(x_hel)), dy=np.mean(np.diff(y_hel)), vert_exag=5.)
ax1.imshow(rgb, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc = ax1.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=term_lt_corr_amax, cmap='RdBu', 
                 vmin=-0.5, vmax=0.5)
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes("right", size="5%", pad=0.1)
cb1 = plt.colorbar(sc1, cax=cax1)
cax1.set_title('Max. xcorr')
ax1.set(xlim=(270000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
      xlabel='Easting [km]', ylabel='Northing [km]',
      title='Low-freq v, term. pos.')
ax2.imshow(rgb, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc2 = ax2.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=term_lt_lag_amax, cmap='RdBu', 
                 vmin=lagnorm_min, vmax=lagnorm_max)
div2 = make_axes_locatable(ax2)
cax2 = div2.append_axes("right", size="5%", pad=0.1)
cb2 = fig.colorbar(sc2, cax=cax2)
cb2.ax.set_title('Lag at max. xcorr')
ax2.set(xlim=(270000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
      xlabel='Easting [km]', ylabel='Northing [km]',
      title='Low-freq v, term. pos.')
plt.tight_layout()
plt.show()