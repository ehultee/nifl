#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 18:33:20 2021

@author: lizz
"""
import statsmodels.api as sm


a_vel = sm.tsa.stattools.acf(np.diff(preds[0]['full']))[1]
b_runoff = sm.tsa.stattools.acf(np.diff(runoff))[1]
F_runoff = np.sqrt((1+(a_vel*b_runoff))/(1-(a_vel*b_runoff)))
a_vel_lt = sm.tsa.stattools.acf(preds[0]['secular']+preds[0]['transient'])[1]
b_runoff_lt = sm.tsa.stattools.acf(rf_lowfreq(t_grid))[1]
F_runoff_lt = np.sqrt((1+(a_vel_lt*b_runoff_lt))/(1-(a_vel_lt*b_runoff_lt)))

## modify CCSL for terminus
b_terminus = sm.tsa.stattools.acf(np.diff(td))[1]
F_terminus = np.sqrt((1+(a_vel*b_terminus))/(1-(a_vel*b_terminus)))
b_terminus_lt = sm.tsa.stattools.acf(tf_lowfreq(t_grid))[1]
F_terminus_lt = np.sqrt((1+(a_vel_lt*b_terminus_lt))/(1-(a_vel_lt*b_terminus_lt)))


## modify CCSL for smb
b_smb = sm.tsa.stattools.acf(np.diff(smb))[1]
F_smb = np.sqrt((1+(a_vel*b_smb))/(1-(a_vel*b_smb)))
b_smb_lt = sm.tsa.stattools.acf(smb_lowfreq(t_grid))[1]
F_smb_lt = np.sqrt((1+(a_vel_lt*b_smb_lt))/(1-(a_vel_lt*b_smb_lt)))


runoff_corr_amax = []
runoff_lag_amax = []
runoff_significance = []

for xy, pred in zip(xys, preds):
    corr, lags, ci = nifl.Xcorr1D(xy, series_func=runoff_func, series_dates=d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              diff=1, normalize=True, pos_only=True)
    ci_mod = F_runoff*np.asarray(ci)
    runoff_corr_amax.append(corr[abs(corr).argmax()])
    runoff_lag_amax.append(lags[abs(corr).argmax()])
    runoff_significance.append(abs(corr[abs(corr).argmax()]) > ci_mod[abs(corr).argmax()])


rf_lt_corr_amax = []
rf_lt_lag_amax = []
rf_lt_significance = []
for xy, pred in zip(xys, preds):
    corr, lags, ci = Xcorr1D_lt(xy, series_func=rf_lowfreq, series_dates=d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              diff=0, normalize=True, pos_only=False)
    ci_mod = F_runoff_lt * np.asarray(ci)
    rf_lt_corr_amax.append(corr[abs(corr).argmax()])
    rf_lt_lag_amax.append(lags[abs(corr).argmax()])
    rf_lt_significance.append(abs(corr[abs(corr).argmax()]) > ci_mod[abs(corr).argmax()])


terminus_corr_amax = []
terminus_lag_amax = []
terminus_significance = []

for xy, pred in zip(xys, preds):
    corr, lags, ci = nifl.Xcorr1D(xy, series_func=termini_func, series_dates=tm_d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              diff=1, normalize=True, pos_only=True)
    ci_mod = F_terminus *np.asarray(ci)
    terminus_corr_amax.append(corr[abs(corr).argmax()])
    terminus_lag_amax.append(lags[abs(corr).argmax()])
    terminus_significance.append(abs(corr[abs(corr).argmax()]) > ci_mod[abs(corr).argmax()])

term_lt_corr_amax = []
term_lt_lag_amax = []
terminus_lt_significance = []
for xy, pred in zip(xys, preds):
    corr, lags, ci = Xcorr1D_lt(xy, series_func=tf_lowfreq, series_dates=tm_d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              diff=0, normalize=True)
    ci_mod = F_terminus_lt *np.asarray(ci)
    term_lt_corr_amax.append(corr[abs(corr).argmax()])
    term_lt_lag_amax.append(lags[abs(corr).argmax()])
    terminus_lt_significance.append(abs(corr[abs(corr).argmax()]) > ci_mod[abs(corr).argmax()])



smb_corr_amax = []
smb_lag_amax = []
smb_significance = []

for xy, pred in zip(xys, preds):
    corr, lags, ci = nifl.Xcorr1D(xy, series_func=smb_func, series_dates=smb_d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              diff=1, normalize=True, pos_only=True)
    ci_mod = F_smb*np.asarray(ci)
    smb_corr_amax.append(corr[abs(corr).argmax()])
    smb_lag_amax.append(lags[abs(corr).argmax()])
    smb_significance.append(abs(corr[abs(corr).argmax()]) > ci_mod[abs(corr).argmax()])


smb_lt_corr_amax = []
smb_lt_lag_amax = []
smb_lt_significance = []
for xy, pred in zip(xys, preds):
    corr, lags, ci = Xcorr1D_lt(xy, series_func=smb_lowfreq, series_dates=smb_d_interp, 
                              velocity_pred=pred, t_grid=t_grid, t_limits=(2009,2017), 
                              diff=0, normalize=True, pos_only=False)
    ci_mod = F_smb_lt * np.asarray(ci)
    smb_lt_corr_amax.append(corr[abs(corr).argmax()])
    smb_lt_lag_amax.append(lags[abs(corr).argmax()])
    smb_lt_significance.append(abs(corr[abs(corr).argmax()]) > ci_mod[abs(corr).argmax()])