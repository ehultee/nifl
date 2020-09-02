#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and save calving fronts
Created on Tue Sep  1 11:50:41 2020

@author: lizz
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import datetime
import iceutils as ice

## Set up combined hdf5 stack
fpath='/Users/lizz/Documents/GitHub/Data_unsynced/Gld-Stack/'
hel_stack = ice.MagStack(files=[fpath+'vx.h5', fpath+'vy.h5'])
data_key = 'igram' # B. Riel convention for access to datasets in hdf5 stack

with open('/Users/lizz/Documents/GitHub/Data_unsynced/Helheim-selected_fronts.csv', 'w', newline='') as csvfile:
    fieldnames = ['Time', 'Front_points']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(series[0]): # slice for each available point in timeseries
        s = hel_stack.slice(index=i, key=data_key)
        t = hel_stack.tdec[i]
        
        ## Plot the slice
        fig, ax = plt.subplots()
        ax.contourf(hel_stack.stacks[0]._datasets['x'], hel_stack.stacks[0]._datasets['y'], s, 50)
        ax.set(
        	xlim=(292000, 320500), ylim=(-2581400, -2555000),
        	xticks=[295000, 300000, 305000, 310000, 315000, 320000], 
            xticklabels=['295', '300', '305', '310', '315', '320'], xlabel='Easting [km]',
        	yticks=[-2580000, -2575000, -2570000, -2565000, -2560000, -2555000], 
            yticklabels=['-2580', '-2575', '-2570', '-2565', '-2560', '-2555'], ylabel='Northing [km]',
            aspect=1
        	)
        plt.tight_layout()
        plt.show()
        
        ## Call ginput to get points
        pts = plt.ginput(n=-1, show_clicks=True)
        ## Write the selected points to CSV
        writer.writerow({'time': t, 'Front_points': pts})
        plt.close()



