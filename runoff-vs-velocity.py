#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare time series of catchment runoff with velocity timeseries
Created on Tue Oct 20 08:22:25 2020

@author: lizz
"""
import pandas as pd
from scipy import interpolate
import numpy as np
 
fn = '/Users/lizz/Desktop/RACMO2_3p2_Helheimgletscher_runoff_1958-2017.csv'
runoff = np.loadtxt(fn, delimiter=',')
rnf = runoff[runoff[:,0]>=2006]
rf = rnf[rnf[:,0]<=2016]

runoff_dates = pd.date_range(start='2006-01-01', end='2016-12-01', periods=len(rf))
runoff_d = [d.utctimetuple() for d in runoff_dates]
d_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in runoff_d]
runoff_func = interpolate.interp1d(d_interp, rf[:,2])

fig, ax = plt.subplots(1)
ax.plot(runoff_dates, rf[:,2])
plt.title('Catchment-integrated runoff')
plt.show()

t_grid = np.linspace(hel_stack.tdec[0], hel_stack.tdec[-1], 1000)
coincident_dates = t_grid[t_grid<=max(d_interp)]

for j, pt in enumerate(xys):
    coincident_runoff = runoff_func(coincident_dates) # sample at same dates as helheim-tseries_decomp
    runoff_diff = np.diff(coincident_runoff)
    vel_series = preds[j]['full'][t_grid<=max(d_interp)]
    vel_diff = np.diff(vel_series)
    rd_norm = (runoff_diff-np.mean(runoff_diff)) / (np.std(runoff_diff)*len(runoff_diff))
    vd_norm = (vel_diff-np.mean(vel_diff)) / (np.std(vel_diff))

    ## Compute cross-correlation with smooth preds[i] from helheim-tseries
    corr = np.correlate(rd_norm, vd_norm, mode='full')
    lags = range(int(-0.5*len(corr)), int(0.5*len(corr)+1))
    fig1, ax1 = plt.subplots(1)
    # lags, c, _, _ = ax1.xcorr(coincident_smb, vel_series, usevlines=False, maxlags=200, normed=True, lw=2)
    # lags, c, _, _ = ax1.xcorr(smb_diff, vel_diff, usevlines=False, maxlags=200, normed=True, lw=2)
    ax1.plot(lags, corr)
    # ax1.set(xticks=[0,int(0.5*len(corr)), len(corr)], xticklabels=[str(-1*int(0.5*len(corr))), '0', str(int(0.5*len(corr)))])
    ci = [2/np.sqrt(len(coincident_smb)-abs(k)) for k in lags]
    ax1.plot(lags, ci, ls=':', color='k')
    ax1.plot(lags, -1*np.array(ci), ls=':', color='k')
    ax1.grid(True)
    ax1.set(ylabel='Cross-correlation', xlabel='Lag [unit displacement of 3 days]')
