#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test stationarity

Created on Thu Mar 25 16:46:31 2021

@author: lizz
"""
import statsmodels.api as sm

def adf_test(timeseries):
	print('A timeseries ready for xcorr analysis should have ADF test statistic more negative than critical value (reject the null hypothesis).')
    print ('Results of Dickey-Fuller Test:')
    dftest = sm.tsa.stattools.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
def kpss_test(timeseries):
	print('A timeseries ready for xcorr analysis should have KPSS statistic lower than the critical value (fail to reject the null hypothesis).')
    print ('Results of KPSS Test:')
    kpsstest = sm.tsa.stattools.kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
    
adf_test(np.diff(td,n=1))
kpss_test(np.diff(td, n=1))