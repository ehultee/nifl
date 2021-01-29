#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:34:57 2021
Cross-correlation of isolated low frequency or high frequency variability
Stack plot of these signals

@author: lizz
"""

import numpy as np
import pandas as pd
import csv
import iceutils as ice
from scipy import interpolate


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

# In[ ]:
## Low-frequency terminus variability
tm_evensampled = termini_func(t_grid_trimmed).squeeze()
window = np.int(3/np.mean(np.diff(t_grid_trimmed.squeeze()))) # set window size to the number of time steps in 1.5 years
tm_filtered = ndimage.uniform_filter1d(tm_evensampled, size=window)
tf_lowfreq = interpolate.UnivariateSpline(t_grid_trimmed, tm_filtered, s=0)


term_lt_corr_amax = []
term_lt_lag_amax = []
for xy, pred in zip(xys, preds):
    corr, lags, ci = Xcorr1D_lt(xy, series_func=tf_lowfreq, series_dates=tm_d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              diff=1, normalize=True)
    term_lt_corr_amax.append(corr[abs(corr).argmax()])
    term_lt_lag_amax.append(lags[abs(corr).argmax()])

# In[ ]:
## Low-frequency runoff variability
rnf_evensampled = runoff_func(t_grid_trimmed).squeeze()
rnf_filtered = ndimage.uniform_filter1d(rnf_evensampled, size=window)
rf_lowfreq = interpolate.UnivariateSpline(t_grid_trimmed, rnf_filtered, s=0)

rf_lt_corr_amax = []
rf_lt_lag_amax = []
for xy, pred in zip(xys, preds):
    corr, lags, ci = Xcorr1D_lt(xy, series_func=rf_lowfreq, series_dates=d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              diff=1, normalize=True)
    rf_lt_corr_amax.append(corr[abs(corr).argmax()])
    rf_lt_lag_amax.append(lags[abs(corr).argmax()])

# In[ ]:
## Low-frequency SMB variability
smb_evensampled = 1E-12*np.array(smb_func(t_grid_trimmed).squeeze())
smb_filtered = ndimage.uniform_filter1d(smb_evensampled, size=window)
smb_lowfreq = interpolate.UnivariateSpline(t_grid_trimmed, smb_filtered, s=0)

smb_lt_corr_amax = []
smb_lt_lag_amax = []
for xy, pred in zip(xys, preds):
    corr, lags, ci = Xcorr1D_lt(xy, series_func=smb_lowfreq, series_dates=smb_d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              diff=1, normalize=True)
    smb_lt_corr_amax.append(corr[abs(corr).argmax()])
    smb_lt_lag_amax.append(lags[abs(corr).argmax()])

# In[ ]:
## Plot the low-frequency signals in stack
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
for i in range(len(xys)):
    # ax1.plot(hel_stack.tdec, series[i], '.')
    ax1.plot(t_grid, preds[i]['full'], label='Point {}'.format(i), color=clrs[i], lw=1.0, alpha=0.3)
    ax1.plot(t_grid, preds[i]['secular']+preds[i]['transient'], color=clrs[i], lw=2.0)
ax1.set(ylabel='Surf. speed [km/a]',
        yticks=(4, 6, 8), xlim=(2009,2017))

ax2.scatter(smb_d_interp, 0.001*np.array(smb['SMB_int']), color='k', alpha=0.3) # raw SMB data
ax2.plot(t_grid_trimmed, 0.001*np.array(smb_func(t_grid_trimmed)), color='k', alpha=0.3)
ax2.plot(t_grid_trimmed, 1E9*np.array(smb_lowfreq(t_grid_trimmed)), color='k', alpha=0.7)
ax2.set(ylabel='Int. SMB [m3 w.e.]')

ax3.scatter(d_interp, 1000*np.array(rf[:,2]), color='k', alpha=0.3) # raw runoff data
ax3.plot(t_grid_trimmed, 1000*np.array(runoff_func(t_grid_trimmed)), color='k', alpha=0.3)
ax3.plot(t_grid_trimmed, 1000*np.array(rf_lowfreq(t_grid_trimmed)), color='k', alpha=0.7)
ax3.set(ylabel='Int. runoff [m3 w.e.]')
ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

ax4.scatter(tm_d_interp, tm['term_km'], color='k', alpha=0.3) # raw terminus data
ax4.plot(t_grid_trimmed, termini_func(t_grid_trimmed), color='k', alpha=0.3)
ax4.plot(t_grid_trimmed, tf_lowfreq(t_grid_trimmed), color='k', alpha=0.7)
ax4.set(ylabel='Term. pos. [km]')
ax4.set(xlim=(2009,2017), xlabel='Year')
for ax in (ax1, ax2, ax3, ax4):
    ax.grid(True, which='major', axis='x', ls=':', color='k', alpha=0.5)
plt.tight_layout()

# In[ ]:
## Plot the xcorr and lag for long-term terminus variability

ls = LightSource(azdeg=225, altdeg=80)

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
# ax.contourf(x_hel, y_hel, b_hel, cmap='gist_earth', alpha=0.5)
rgb = ls.shade(np.asarray(b_hel), cmap=plt.get_cmap('gist_earth'), blend_mode='overlay',
               dx=np.mean(np.diff(x_hel)), dy=np.mean(np.diff(y_hel)), vert_exag=5.)
ax1.imshow(rgb, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc1 = ax1.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=term_lt_corr_amax, cmap='RdBu', 
                 vmin=-0.5, vmax=0.5)
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes("right", size="5%", pad=0.1)
cb1 = plt.colorbar(sc1, cax=cax1)
cax1.set_title('Max. xcorr')
ax1.set(xlim=(270000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
      xlabel='Easting [km]', ylabel='Northing [km]',
      title='Low-freq v vs. term. pos.')
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
      title='Low-freq v vs. term. pos.')
plt.tight_layout()
plt.show()

# In[ ]:
## Plot xcorr and lag for long-term runoff variability

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
# ax.contourf(x_hel, y_hel, b_hel, cmap='gist_earth', alpha=0.5)
rgb = ls.shade(np.asarray(b_hel), cmap=plt.get_cmap('gist_earth'), blend_mode='overlay',
               dx=np.mean(np.diff(x_hel)), dy=np.mean(np.diff(y_hel)), vert_exag=5.)
ax1.imshow(rgb, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc1 = ax1.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=rf_lt_corr_amax, cmap='RdBu', 
                 vmin=-0.5, vmax=0.5)
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes("right", size="5%", pad=0.1)
cb1 = plt.colorbar(sc1, cax=cax1)
cax1.set_title('Max. xcorr')
ax1.set(xlim=(270000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
      xlabel='Easting [km]', ylabel='Northing [km]',
      title='Low-freq v vs. runoff')
ax2.imshow(rgb, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc2 = ax2.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=rf_lt_lag_amax, cmap='RdBu', 
                 vmin=lagnorm_min, vmax=lagnorm_max)
div2 = make_axes_locatable(ax2)
cax2 = div2.append_axes("right", size="5%", pad=0.1)
cb2 = fig.colorbar(sc2, cax=cax2)
cb2.ax.set_title('Lag at max. xcorr')
ax2.set(xlim=(270000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
      xlabel='Easting [km]', ylabel='Northing [km]',
      title='Low-freq v vs. runoff')
plt.tight_layout()
plt.show()

# In[ ]:
## Plot xcorr and lag for long-term SMB variability

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
# ax.contourf(x_hel, y_hel, b_hel, cmap='gist_earth', alpha=0.5)
rgb = ls.shade(np.asarray(b_hel), cmap=plt.get_cmap('gist_earth'), blend_mode='overlay',
               dx=np.mean(np.diff(x_hel)), dy=np.mean(np.diff(y_hel)), vert_exag=5.)
ax1.imshow(rgb, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc1 = ax1.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=smb_lt_corr_amax, cmap='RdBu', 
                 vmin=-0.5, vmax=0.5)
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes("right", size="5%", pad=0.1)
cb1 = plt.colorbar(sc1, cax=cax1)
cax1.set_title('Max. xcorr')
ax1.set(xlim=(270000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
      xlabel='Easting [km]', ylabel='Northing [km]',
      title='Low-freq v vs. SMB')
ax2.imshow(rgb, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc2 = ax2.scatter(np.asarray(xys)[:,0], np.asarray(xys)[:,1], c=smb_lt_lag_amax, cmap='RdBu', 
                 vmin=lagnorm_min, vmax=lagnorm_max)
div2 = make_axes_locatable(ax2)
cax2 = div2.append_axes("right", size="5%", pad=0.1)
cb2 = fig.colorbar(sc2, cax=cax2)
cb2.ax.set_title('Lag at max. xcorr')
ax2.set(xlim=(270000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
      xlabel='Easting [km]', ylabel='Northing [km]',
      title='Low-freq v vs. SMB')
plt.tight_layout()
plt.show()