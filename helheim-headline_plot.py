#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headline fig components
Created on Wed Jan  6 12:08:12 2021

@author: lizz
"""

## Flowline to plot
# xy_plot = [(xh[i], yh[i]) for i in range(0, upstream_max)]
xy_plot = np.array(xys)

clrs = plt.get_cmap('plasma')(np.array(range(len(xys)))/len(xys))
t_grid_trimmed = t_grid[np.argwhere(t_grid<2017)] ## valid range for interpolated funcs


## Plot velocity stack and forcing series
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
for i in range(len(xys)):
    # ax1.plot(hel_stack.tdec, series[i], '.')
    ax1.plot(t_grid, preds[i]['full'], label='Point {}'.format(i), color=clrs[i])
ax1.set(ylabel='Surf. speed [km/a]',
        yticks=(4, 6, 8), xlim=(2009,2017))
ax2.scatter(smb_d_interp, 0.001*np.array(smb['SMB_int']), color='k') # raw SMB data
ax2.plot(t_grid_trimmed, 0.001*np.array(smb_func(t_grid_trimmed)), color='k', alpha=0.7)
ax2.set(ylabel='Int. SMB [m3 w.e.]')
ax3.scatter(d_interp, 1000*np.array(rf[:,2]), color='k') # raw runoff data
ax3.plot(t_grid_trimmed, 1000*np.array(runoff_func(t_grid_trimmed)), color='k', alpha=0.7)
ax3.set(ylabel='Int. runoff [m3 w.e.]')
ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax4.scatter(tm_d_interp, tm['term_km'], color='k') # raw terminus data
ax4.plot(t_grid_trimmed, termini_func(t_grid_trimmed), color='k', alpha=0.7)
ax4.set(ylabel='Term. pos. [km]')
ax4.set(xlim=(2009,2017), xlabel='Year')
for ax in (ax1, ax2, ax3, ax4):
    ax.grid(True, which='major', axis='x', ls=':', color='k', alpha=0.5)
plt.tight_layout()


## plot splines with data from two points
highlight_idx = (5, 15)
fig, ax = plt.subplots(1, figsize=(6,2))
for j in highlight_idx:
    pt = xys[j]
    series = hel_stack.timeseries(xy=pt, key=data_key)
    ax.plot(hel_stack.tdec, series, '.', color=clrs[j], alpha=0.5)
    ax.plot(t_grid, preds[j]['full'], label='Point {}'.format(j), color=clrs[j])
ax.set(xlabel='Year', ylabel='Surface speed [km/a]',
    yticks=(4, 6, 8), xlim=(2009,2017))
plt.tight_layout()
plt.show()
    

## Plot map of flowline
ls = LightSource(azdeg=225, altdeg=80)

fig, ax = plt.subplots(1, figsize=(6,6))
rgb = ls.shade(np.asarray(b_hel), cmap=plt.get_cmap('gist_earth'), blend_mode='overlay',
               dx=np.mean(np.diff(x_hel)), dy=np.mean(np.diff(y_hel)), vert_exag=5.)
ax.imshow(rgb, origin='lower', extent=(x_hel[0], x_hel[-1], y_hel[0], y_hel[-1]))
sc = ax.scatter(xy_plot[:,0], xy_plot[:,1], c=clrs)
for j in highlight_idx:
    ax.plot(xy_plot[j,0], xy_plot[j,1], marker='*', ms=15., mec='w', c=clrs[j], lw=0.)
ax.set(xlim=(270000, 320000), xticks=(280000, 300000, 320000), 
      ylim=(-2590000, -2550000), yticks=(-2590000, -2570000, -2550000), 
       xticklabels=('280', '300', '320'), yticklabels=('-2590', '-2570', '-2550'),
      xlabel='Easting [km]', ylabel='Northing [km]')
plt.tight_layout()
plt.show()