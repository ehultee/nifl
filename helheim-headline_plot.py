#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headline fig components
Created on Wed Jan  6 12:08:12 2021

@author: lizz
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import datetime # for labelling

## set matplotlib font size defaults
SMALL_SIZE = 10
MEDIUM_SIZE = 11
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# In[]:
## Import calving fronts to display
## hand-picked from MEaSUREs tiles
fronts_fpath = '/Users/lizz/Documents/GitHub/Data_unsynced/Helheim-processed/Helheim-selected_fronts.csv' 
fronts = pd.read_csv(fronts_fpath)
front09 = fronts['Front_points'][0].split()
f09 = np.array([(float(front09[i].strip('()[],')), float(front09[i+1].strip('()[],'))) 
       for i in range(0, len(front09), 2)]) # create an array of points from the list of strings

# # In[]:
# ## COMPONENT PLOTS

# ## Flowline to plot
# # xy_plot = [(xh[i], yh[i]) for i in range(0, upstream_max)]
# xy_plot = np.array(xys)

# clrs = plt.get_cmap('plasma')(np.array(range(len(xys)))/len(xys))
# t_grid_trimmed = t_grid[np.argwhere(t_grid<2017)] ## valid range for interpolated funcs


# ## Plot velocity stack and forcing series
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
# for i in range(len(xys)):
#     # ax1.plot(hel_stack.tdec, series[i], '.')
#     ax1.plot(t_grid, preds[i]['full'], label='Point {}'.format(i), color=clrs[i], lw=2.0)
#     # ax1.plot(t_grid, preds[i]['secular']+preds[i]['transient'], color=clrs[i], lw=2.0, alpha=0.3)
# ax1.set(ylabel='Surf. speed [km/a]',
#         yticks=(4, 6, 8), xlim=(2009,2017))
# ax2.scatter(smb_d_interp, np.array(smb), color='k', alpha=0.5) # raw SMB data
# ax2.plot(t_grid_trimmed, np.array(smb_func(t_grid_trimmed)), color='k', lw=2.0)
# ax2.plot(t_grid_trimmed, 1E9*np.array(smb_lowfreq(t_grid_trimmed)), color='k', lw=2.0, alpha=0.3)
# ax2.set(ylabel='Int. SMB [m$^3$ w.e.]')
# ax3.scatter(d_interp, np.array(runoff), color='k', alpha=0.5) # raw runoff data
# ax3.plot(t_grid_trimmed, np.array(runoff_func(t_grid_trimmed)), color='k', lw=2.0)
# ax3.plot(t_grid_trimmed, np.array(rf_lowfreq(t_grid_trimmed)), color='k', lw=2.0, alpha=0.3)
# ax3.set(ylabel='Int. runoff [m$^3$ w.e.]')
# ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
# ax4.scatter(tm_d_interp, tm, color='k', alpha=0.5) # raw terminus data
# ax4.plot(t_grid_trimmed, termini_func(t_grid_trimmed), color='k', lw=2.0)
# ax4.plot(t_grid_trimmed, tf_lowfreq(t_grid_trimmed), color='k', lw=2.0, alpha=0.3)
# ax4.set(ylabel='Term. pos. [km]')
# ax4.set(xlim=(2009,2017), xlabel='Year')
# for ax in (ax1, ax2, ax3, ax4):
#     ax.grid(True, which='major', axis='x', ls=':', color='k', alpha=0.5)
# plt.tight_layout()


# ## plot splines with data from two points
# highlight_idx = (5, 15)
# fig, ax = plt.subplots(1, figsize=(6,2))
# for j in highlight_idx:
#     pt = xys[j]
#     series = hel_stack.timeseries(xy=pt, key=data_key)
#     ax.plot(hel_stack.tdec, series, '.', color=clrs[j], alpha=0.5)
#     ax.plot(t_grid, preds[j]['full'], label='Point {}'.format(j), color=clrs[j], lw=2.0)
#     ax.plot(t_grid, preds[j]['secular']+preds[j]['transient'], color=clrs[j], lw=1.0, alpha=0.5)
# ax.set(xlabel='Year', ylabel='Surface speed [km/a]',
#     yticks=(4, 6, 8), xlim=(2009,2017))
# plt.tight_layout()
# plt.show()
    

# ## Plot map of flowline
# ls = LightSource(azdeg=225, altdeg=80)

# fig, ax = plt.subplots(1, figsize=(6,6))
# rgb = ls.shade(np.asarray(b_hel), cmap=plt.get_cmap('gist_earth'), blend_mode='overlay',
#                 dx=np.mean(np.diff(x_hel)), dy=np.mean(np.diff(y_hel)), vert_exag=5.)
# ax.imshow(rgb, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
# sc = ax.scatter(xy_plot[:,0], xy_plot[:,1], c=clrs)
# ax.plot(f09[:,0], f09[:,1], color='k')
# for j in highlight_idx:
#     ax.plot(xy_plot[j,0], xy_plot[j,1], marker='*', ms=15., mec='w', c=clrs[j], lw=0.)
# ax.set(xlim=(270000, 320000), xticks=(280000, 300000, 320000), 
#       ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
#         xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
#       xlabel='Easting [km]', ylabel='Northing [km]')
# im = ax.imshow(b_hel, cmap=plt.get_cmap('gist_earth'))
# im.remove()
# cbaxes = inset_axes(ax, width="33%", height="4%", loc='lower left', borderpad=1.5)
# cb = plt.colorbar(im, cax=cbaxes, orientation='horizontal')
# cb.set_ticks([-1000,0,2000])
# cb.ax.set_xticklabels(['-1000', '0', '2000 m a.s.l.'])
# cb.ax.xaxis.set_ticks_position('top')
# cb.ax.xaxis.set_label_position('top')
# plt.tight_layout()
# plt.show()
# # plt.savefig('/Users/lizz/Desktop/{}-helheim_map_with_colorbar'.format(
# #    datetime.date.today().strftime('%Y%m%d')))


# In[]:
## COMPOSITE FIGURE

## GridSpec version
fig3 = plt.figure(figsize=(12,8))
gs = fig3.add_gridspec(nrows=4, ncols=2)
f3_ax1 = fig3.add_subplot(gs[0:3,0])
f3_ax2 = fig3.add_subplot(gs[3,0])
f3_ax3 = fig3.add_subplot(gs[0,1])
f3_ax4 = fig3.add_subplot(gs[1,1])
f3_ax5 = fig3.add_subplot(gs[2,1])
f3_ax6 = fig3.add_subplot(gs[3,1])

## map
f3_ax1.imshow(rgb, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc = f3_ax1.scatter(xy_plot[:,0], xy_plot[:,1], c=clrs)
f3_ax1.plot(f09[:,0], f09[:,1], color='w')
for j in highlight_idx:
    f3_ax1.plot(xy_plot[j,0], xy_plot[j,1], marker='*', ms=15., mec='w', c=clrs[j], lw=0.)
f3_ax1.set(xlim=(270000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
        xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
      xlabel='Easting [km]', ylabel='Northing [km]')
im = f3_ax1.imshow(b_hel, cmap=plt.get_cmap('gist_earth'))
im.remove()
cbaxes = inset_axes(f3_ax1, width="33%", height="4%", loc='lower left', borderpad=1.5)
cb = plt.colorbar(im, cax=cbaxes, orientation='horizontal')
cb.set_ticks([-1000,0,2000])
cb.ax.set_xticklabels(['-1000', '0', '2000 m a.s.l.'])
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')
## splines with data
for j in highlight_idx:
    pt = xys[j]
    series = hel_stack.timeseries(xy=pt, key=data_key)
    f3_ax2.plot(hel_stack.tdec, series, '.', color=clrs[j], alpha=0.5)
    f3_ax2.plot(t_grid, preds[j]['full'], label='Point {}'.format(j), color=clrs[j], lw=2.0)
    f3_ax2.plot(t_grid, preds[j]['secular']+preds[j]['transient'], color=clrs[j], lw=1.0, alpha=0.5)
f3_ax2.set(xlabel='Year', ylabel='Surface speed [km/a]',
    yticks=(4, 6, 8), xlim=(2009,2017))
## Plot velocity stack and forcing series
for i in range(len(xys)):
    # ax1.plot(hel_stack.tdec, series[i], '.')
    f3_ax3.plot(t_grid, preds[i]['full'], label='Point {}'.format(i), color=clrs[i], lw=2.0)
    # f3_ax3.plot(t_grid, preds[i]['secular']+preds[i]['transient'], color=clrs[i], lw=2.0, alpha=0.3)
f3_ax3.set(ylabel='Surf. speed [km/a]',
        yticks=(4, 6, 8))
f3_ax4.scatter(smb_d_interp, np.array(smb), color='k', alpha=0.5) # raw SMB data
f3_ax4.plot(t_grid_trimmed, np.array(smb_func(t_grid_trimmed)), color='k', lw=2.0)
f3_ax4.plot(t_grid_trimmed, 1E9*np.array(smb_lowfreq(t_grid_trimmed)), color='k', lw=2.0, alpha=0.3)
f3_ax4.set(ylabel='Int. SMB [m$^3$ w.e.]')
f3_ax5.scatter(d_interp, np.array(runoff), color='k', alpha=0.5) # raw runoff data
f3_ax5.plot(t_grid_trimmed, np.array(runoff_func(t_grid_trimmed)), color='k', lw=2.0)
f3_ax5.plot(t_grid_trimmed, np.array(rf_lowfreq(t_grid_trimmed)), color='k', lw=2.0, alpha=0.3)
f3_ax5.set(ylabel='Int. runoff [m$^3$ w.e.]', yticks=(0, 1E9))
f3_ax6.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
f3_ax6.scatter(tm_d_interp, tm, color='k', alpha=0.5) # raw terminus data
f3_ax6.plot(t_grid_trimmed, termini_func(t_grid_trimmed), color='k', lw=2.0)
f3_ax6.plot(t_grid_trimmed, tf_lowfreq(t_grid_trimmed), color='k', lw=2.0, alpha=0.3)
f3_ax6.set(ylabel='Term. pos. [km]')
f3_ax6.set(xlim=(2009,2017), xlabel='Year')
for ax in (f3_ax3, f3_ax4, f3_ax5, f3_ax6):
    ax.grid(True, which='major', axis='x', ls=':', color='k', alpha=0.5)
    ax.set(xlim=(2009,2017))
plt.tight_layout()
plt.show()