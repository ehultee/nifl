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

fn1 = '/Users/lizz/GitHub/Data_unsynced/Helheim-processed/HLM_terminus_widthAVE.csv'
termini = pd.read_csv(fn1, parse_dates=True, usecols=[0,1])
termini['date'] = pd.to_datetime(termini['date'])
trmn = termini.loc[termini['date'].dt.year >= 2006]
tm = trmn.loc[trmn['date'].dt.year <=2016]

termini_d = [d.utctimetuple() for d in tm['date']]
tm_d_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in termini_d]
termini_func = interpolate.interp1d(tm_d_interp, tm['term_km'])