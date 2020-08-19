#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:26:34 2020
Flowline profiles for Helheim analysis

@author: lizz
"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy import interpolate
import sys
sys.path.insert(0, '/Users/lizz/Documents/GitHub/nifl')
import nifl_helper


## Read in network of flowlines
filename = '/Users/lizz/Documents/GitHub/Data_unsynced/Auto_selected-networks/Gld-autonetwork-GID175-date_2018-10-04.csv'
flowlines_foranalysis = []
coords_list = nifl_helper.Flowline_CSV(filename, has_width=True, flip_order=False)


## Read in and interpolate BedMachine topography
gl_bed_path ='/Users/lizz/Documents/GitHub/Data_unsynced/BedMachine-Greenland/BedMachineGreenland-2017-09-20.nc'
fh = Dataset(gl_bed_path, mode='r')
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

## Interpolate in area of Helheim
xl, xr = 6100, 6400
yt, yb = 12700, 13100
x_hel = xx[xl:xr]
y_hel = yy[yt:yb]
s_hel = ss[yt:yb, xl:xr]
b_hel = bb[yt:yb, xl:xr]
S_helheim = interpolate.RectBivariateSpline(x_hel, y_hel[::-1], s_hel.T[::,::-1]) #interpolating surface elevation provided
B_helheim = interpolate.RectBivariateSpline(x_hel, y_hel[::-1], b_hel.T[::,::-1]) #interpolating surface elevation provided

## Extract along flowlines
bed_vals = [float(B_helheim(x[0],x[1])) for x in coords_list[0]]
surface_vals = [float(S_helheim(x[0],x[1])) for x in coords_list[0]]
xvals = 0.001*np.array(nifl_helper.ArcArray(coords_list[0]))

## Plot
fig, ax = plt.subplots(1)
ax.plot(xvals, bed_vals, color='saddlebrown')
ax.plot(xvals, surface_vals, color='darkgrey')
plt.fill_between(xvals, surface_vals, bed_vals, color='darkgrey', alpha=0.5)
plt.fill_between(xvals, bed_vals, y2=-1300, color='saddlebrown', alpha=0.5, hatch='/')
ax.set_xlim(xvals[-1], xvals[0]) #flip x-axis
ax.set_ylim(-1300, 1900)
ax.set_aspect(0.01)
plt.show()

