#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:55:12 2026

@author: tomasferreyrahauchar
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as snd

import glob
import h5py

from matplotlib.path import Path
from skimage.filters import gaussian
from skimage.measure import find_contours
from tqdm import tqdm

from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata, interp1d
from scipy.spatial import cKDTree

from skimage.morphology import local_minima, disk, remove_small_holes, binary_erosion, binary_dilation, binary_closing, binary_opening 

# import skimage as ski
# from scipy.optimize import root_scalar
# from scipy.integrate import quad

# import os
# import json

# from datetime import timedelta
from time import time

# from calibration import conversion_function_ROI
# from ice_detection import get_contour

def nangauss(altus, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):    
        V = altus.copy()
        V[np.isnan(altus)] = 0
        W = 0 * altus.copy() + 1
        W[np.isnan(altus)] = 0
        VV,WW = snd.gaussian_filter(V, sigma), snd.gaussian_filter(W, sigma)
        gdp = VV/WW
        gdp[np.isnan(altus)] = np.nan
        return gdp

def circle_fit(params, x, y):
    xc, yc, r = params
    return np.sqrt((x - xc)**2 + (y - yc)**2) - r

def grad4(f, dx):
    df = np.empty_like(f)

    # Interior points
    df[2:-2] = ( -f[4:] + 8*f[3:-1] - 8*f[1:-3] + f[:-4] ) / (12*dx)

    # One-sided 4th-order stencils at boundaries
    df[0] = (-25*f[0] + 48*f[1] - 36*f[2] + 16*f[3] - 3*f[4]) / (12*dx)
    df[1] = (-3*f[0] - 10*f[1] + 18*f[2] - 6*f[3] + f[4]) / (12*dx)

    df[-2] = (3*f[-1] + 10*f[-2] - 18*f[-3] + 6*f[-4] - f[-5]) / (12*dx)
    df[-1] = (25*f[-1] - 48*f[-2] + 36*f[-3] - 16*f[-4] + 3*f[-5]) / (12*dx)

    return df


def interpolate_gaussian( xdata, ydata, zdata, xn, yn, dist_threshold=3, sigmas=[5,10] ):
    """
    Interpolate scattered data onto a regular grid using anisotropic Gaussian
    weighted averaging.

    Each grid point is assigned a value computed as the weighted average of all
    input data, where the weights decrease according to a Gaussian function of
    the distance to the grid point. The Gaussian may have different standard
    deviations along the z- and y-directions.

    Grid points farther than `dist_threshold` from the nearest input point are
    masked and assigned NaN.

    Parameters
    ----------
    xdata : (N,) array_like
        x-coordinates of the input data.
    ydata : (N,) array_like
        y-coordinates of the input data.
    zdata : (N,) array_like
        Values to interpolate.
    zn : (M, K) ndarray
        z-coordinates of the output grid.
    yn : (M, K) ndarray
        y-coordinates of the output grid.
    dist_threshold : float, optional
        Maximum allowed Euclidean distance from a grid point to the nearest
        input sample. Grid points farther away are set to NaN.
    sigmas : sequence of two floats, optional
        Standard deviations (sigma_z, sigma_y) of the Gaussian kernel. Larger
        values produce smoother interpolation.

    Returns
    -------
    gg : (M, K) ndarray
        Interpolated grid with NaNs outside the specified distance threshold.

    Notes
    -----
    - Every grid point is computed using all input samples.
    - The interpolation is smooth but computationally expensive, with
      complexity O(N * M * K), where N is the number of input points.
    - The cKDTree is used only to determine which grid points should be masked,
      not for the interpolation itself.
    """
    sig_x, sig_y = sigmas
    sig_x2, sig_y2 = sig_x**2, sig_y**2
    values = zdata
    
    points = np.column_stack((xdata, ydata))
    tree = cKDTree(points)
    grid_points = np.column_stack((xn.ravel(), yn.ravel()))
    dist, _ = tree.query(grid_points, k=1)
    gdist = dist.reshape(xn.shape)

    gg = np.full_like(xn, np.nan) * 1.

    ny,nx = np.shape(xn)
    for l in tqdm(range(ny)):
        for j in range(nx):
            xp,yp = xn[l,j], yn[l,j]
            dist2 = (xdata - xp)*(xdata - xp) / sig_x2 + (ydata - yp)*(ydata - yp) / sig_y2
            wiegh = np.exp( -dist2/2 )

            try: gg[l,j] = np.average(values, weights=wiegh)
            except ZeroDivisionError: gg[l,j] = 0.
            
    gg[ gdist > dist_threshold ] = np.nan    
    return gg


def interpolate_idw(xdata, ydata, zdata, xn, yn, dist_threshold=3, power=2, eps=1e-12):
    """
    Interpolate scattered data onto a regular grid using Inverse Distance
    Weighting (IDW).

    Parameters
    ----------
    xdata : (N,) array_like
        Values to interpolate.
    ydata : (N,) array_like
        y-coordinates of the input data.
    zdata : (N,) array_like
        z-coordinates of the input data.
    zn, yn : 2D ndarray
        Coordinates of the output grid.
    dist_threshold : float, optional
        Grid points farther than this from the nearest input point are set
        to NaN.
    power : float, optional
        Power parameter of the IDW interpolation.
        Typical values are:
            1 : smooth
            2 : standard
            3-4 : more local
    eps : float, optional
        Small number preventing division by zero.

    Returns
    -------
    gg : ndarray
        Interpolated grid.
    """

    values = zdata

    points = np.column_stack((xdata, ydata))
    tree = cKDTree(points)

    grid_points = np.column_stack((xn.ravel(), yn.ravel()))
    gdist, _ = tree.query(grid_points, k=1)
    gdist = gdist.reshape(xn.shape)

    gg = np.full_like(xn, np.nan, dtype=float)

    ny, nx = xn.shape

    for l in tqdm(range(ny)):
        for j in range(nx):

            xp = xn[l, j]
            yp = yn[l, j]

            dist = np.sqrt((xdata - xp)**2 + (ydata - yp)**2)

            # Exact data point
            idx = np.argmin(dist)
            if dist[idx] < eps:
                gg[l, j] = values[idx]
                continue

            weights = 1.0 / (dist + eps)**power

            gg[l, j] = np.sum(weights * values) / np.sum(weights)

    gg[gdist > dist_threshold] = np.nan

    return gg


def get_data(file):
    with h5py.File(file, 'r') as f:

        cha_t = f['Sensors/Time'][:]
        cha_s = f['Sensors/Seconds'][:]
        cha_v = f['Sensors/Velocity'][:]
        cha_Tb = f['Sensors/T_bot'][:]
        cha_Tt = f['Sensors/T_top'][:]

        hol_x = f['Holder/X contour'][:]
        hol_y = f['Holder/Y contour'][:]

        ice_t = f['Ice/Time'][:]
        ice_x, ice_y = f['Ice/X contour'][:], f['Ice/Y contour'][:]
        
    return cha_t, cha_s, cha_v, cha_Tb, cha_Tt, hol_x, hol_y, ice_t, ice_x, ice_y



def mask_ice_pos( xgr, ygr, ice_sx, ice_sy ):

    x_mask = np.concatenate( [ice_sx[0], ice_sx[-1][::-1], [ice_sx[0][0]] ] ) 
    y_mask = np.concatenate( [ice_sy[0], ice_sy[-1][::-1], [ice_sy[0][0]] ] ) 
    
    mask = np.zeros_like(ygr, dtype=bool)
    
    for i in tqdm(range(len(xgr[0,:]))):
        
        xl = xgr[0,i] 
        diff = x_mask - xl 
        cross = np.where( (diff[1:] * diff[:-1]) < 0 )[0]
        
        crosses = y_mask[cross] + (xl - x_mask[cross]) * (y_mask[cross+1] - y_mask[cross]) / (x_mask[cross+1] - x_mask[cross])
        
        for j in range(len(crosses)):
            fil = ygr[:,0]>crosses[j]
            mask[fil,i] = ~ mask[fil,i]
    
    return mask, (x_mask,y_mask)


def order_data(ice_x, ice_y, fourier_smooth=False, modes=20, p0_circ = (30,-30,30) ):
    
    xc0, yc0, r0 = p0_circ
    result = least_squares(circle_fit, [xc0, yc0, r0], args=(ice_x[0], ice_y[0]))
    cx, cy, rc = result.x[0], result.x[1], result.x[2] 

    ice_xo, ice_yo = [], []
    ice_xs, ice_ys = [], []
    for n in range(0, len(ice_t), 1):
        r = np.sqrt( (ice_x[n] - cx)**2 + (ice_y[n] - cy)**2 )
        theta = np.mod( np.arctan2( ice_x[n] - cx, ice_y[n] - cy ), np.pi * 2 ) 
        ordert = np.argsort(theta)
        
        theta, r = theta[ordert], r[ordert]
        coco = np.fft.rfft( r, n=len(r) )
        coco[modes:] = 0
        ir = np.fft.irfft( coco, n=len(r) )


        if np.min( ice_y[n][ordert] - cy ) > 0: break
        
        ice_xo.append( ice_x[n][ordert] - cx )
        ice_yo.append( ice_y[n][ordert] - cy )
        ice_xs.append( ir * np.sin(theta) )
        ice_ys.append( ir * np.cos(theta) )

    if fourier_smooth:
        return ice_xs, ice_ys, cx,cy 

    else:
        return ice_xo, ice_yo, cx,cy 



def order_holder(x, y):
    points = np.column_stack((x, y))

    # Top-left point: largest y, then smallest x
    start = np.lexsort((points[:, 0], -points[:, 1]))[0]

    tree = cKDTree(points)

    visited = np.zeros(len(points), dtype=bool)
    order = []

    current = start

    for _ in range(len(points)):
        order.append(current)
        visited[current] = True

        # Ask for all neighbours ordered by distance
        _, idx = tree.query(points[current], k=len(points))

        # Pick the nearest unvisited one
        for j in idx:
            if not visited[j]:
                current = j
                break

    return x[order], y[order]
    
    
def area_vol( x, y, order_grad=2 ):
    if len(y)>1:
        if order_grad == 2: dy, dx = np.gradient(y), np.gradient(x)
        elif order_grad == 4: dy, dx = grad4(y,1), grad4(x, 1)
        else: 
            print('Use order 2 or 4') 
            return np.nan,np.nan
    else: return np.nan,np.nan    
        
    fil1, fil2 = x>=0, x<=0
    V1 = np.pi * np.trapezoid( x[fil1]**2 * dy[fil1] )
    A1 = 2*np.pi * np.trapezoid( x[fil1] * np.sqrt( dx[fil1]**2 + dy[fil1]**2) )

    V2 = np.pi * np.trapezoid( x[fil2]**2 * dy[fil2] )
    A2 = 2*np.pi * np.trapezoid( x[fil2] * np.sqrt( dx[fil2]**2 + dy[fil2]**2) )
        
    return (np.abs(V1) + np.abs(V2)) / 2, (np.abs(A1) + np.abs(A2)) / 2

def show_contours( xgr, ygr, tgr, levels, fmt='-', plot=True ):
    extent = ( np.min(xgr), np.max(xgr), np.min(ygr), np.max(ygr) )
    
    plt.figure()
    
    plt.imshow( tgr, extent=extent )    
    
    xcs, ycs = [],[] 
    for lev in levels:
        cont = find_contours( tgr, level = lev )
        if len(cont)>0:
            yc, xc = np.vstack(cont).T
        
            x_contour = np.interp(xc, np.arange(len(xgr[0,:])), xgr[0,:])
            y_contour = np.interp(yc, np.arange(len(ygr[:,0])), ygr[:,0])
            xcs.append(x_contour); ycs.append(y_contour)
            
            plt.plot( x_contour, y_contour, fmt )
    
    plt.show()
    if not plot: plt.close()
    return xcs,ycs

def calculate_areas_volumes_im( xgr, ygr, tgr, levels, order_grad=2 ):
    
    Vs, As = [],[]
    for lev in levels:
        cont = find_contours( tgr, level = lev )
        if len(cont)>0:
            yc, xc = np.vstack(cont).T
        
            x_contour = np.interp(xc, np.arange(len(xgr[0,:])), xgr[0,:])
            y_contour = np.interp(yc, np.arange(len(ygr[:,0])), ygr[:,0])
    
            V,A = area_vol(x_contour, y_contour, order_grad=order_grad)
        else: V, A = np.nan,np.nan
        Vs.append(V); As.append(A)

    return np.array(Vs), np.array(As)


def curve_intersections(xc, yc, xh, yh):
    """
    Find intersections between a closed curve (xc, yc) and a second
    sampled curve (xh, yh).

    The curves are linearly interpolated between their data points.

    Returns
    -------
    intersections : (N, 2) ndarray
        Intersection coordinates.
    """

    # Closed curve
    boundary = Path(np.column_stack((xc, yc)))

    # Determine whether each point of the second curve is inside/outside
    points = np.column_stack((xh, yh))
    inside = boundary.contains_points(points)

    # Find segments where the second curve changes from inside to outside
    crossings = np.where(inside[:-1] != inside[1:])[0]

    intersections = []

    for i in crossings:
        # Linear interpolation between points i and i+1.
        # We need to find where the segment crosses the boundary.
        p1 = points[i]
        p2 = points[i + 1]

        # Binary search along the segment using Path.contains_point
        lo, hi = 0.0, 1.0

        for _ in range(40):
            t = 0.5 * (lo + hi)
            p = p1 + t * (p2 - p1)

            if boundary.contains_point(p) == inside[i]: lo = t
            else: hi = t

        t = 0.5 * (lo + hi)
        intersections.append(p1 + t * (p2 - p1))

    return np.array(intersections), inside


def area_vol_holder(xh_i, yh_i):
    vol,_ = area_vol(xh_i, yh_i)

    r = np.abs( xh_i[-1]- xh_i[0] ) / 2
    are = np.pi * r**2
    
    return vol, are


def calculate_areas_volumes_im2( xgr, ygr, tgr, levels, order_grad=2 ):
    
    Vs, As, Vh, Ah = [],[], [], []
    for lev in levels:
        cont = find_contours( tgr, level = lev )
        if len(cont)>0:
            yc, xc = np.vstack(cont).T
        
            x_contour = np.interp(xc, np.arange(len(xgr[0,:])), xgr[0,:])
            y_contour = np.interp(yc, np.arange(len(ygr[:,0])), ygr[:,0])
    
            cc, inside = curve_intersections(x_contour, y_contour, hol_xo, hol_yo)
            if len(cc)>1:
                xh_i, yh_i = np.concatenate(( [cc[0,0]], hol_xo[inside], [cc[1,0]] )), np.concatenate(( [cc[0,1]], hol_yo[inside], [cc[1,1]] ))
                vh, ah = area_vol_holder(xh_i, yh_i)
                
            else: vh, ah = np.nan, np.nan 
    
            V,A = area_vol(x_contour, y_contour, order_grad=order_grad)

        else: 
            V, A = np.nan,np.nan

        Vs.append(V); As.append(A)
        Vh.append(vh); Ah.append(ah)

    return np.array(Vs), np.array(As), np.array(Vh), np.array(Ah)


# def 2d_area( xgr, ygr, tgr ):
#     Vs, As, Vh, Ah = [],[], [], []
#     for lev in levels:
#         cont = find_contours( tgr, level = lev )
#         if len(cont)>0:
#             yc, xc = np.vstack(cont).T
        
#             x_contour = np.interp(xc, np.arange(len(xgr[0,:])), xgr[0,:])
#             y_contour = np.interp(yc, np.arange(len(ygr[:,0])), ygr[:,0])


#%%
path = '/Volumes/Ice blocks/Sphere channel/Results/'
files = glob.glob( path + '*.hdf5' )


t1 = time()

# cha_t, cha_s, cha_v, cha_Tb, cha_Tt, hol_x, hol_y, ice_t, ice_x, ice_y = get_data(files[1])
# cha_t, cha_s, cha_v, cha_Tb, cha_Tt, hol_x, hol_y, ice_t, ice_x, ice_y = get_data(files[4])
cha_t, cha_s, cha_v, cha_Tb, cha_Tt, hol_x, hol_y, ice_t, ice_x, ice_y = get_data(files[-1])

t2 = time()
print(t2-t1)

print( cha_v )
#%%

fourier_smooth = False

dt = np.diff(ice_t)[0]

ice_xo, ice_yo, cx, cy = order_data(ice_x, ice_y)
ice_xs, ice_ys, cx, cy = order_data(ice_x, ice_y, fourier_smooth=True , modes=20)



fig, ax = plt.subplots(1,2,figsize=(13,6), layout='constrained')

for n in range(len(ice_xo)):
    ax[0].plot( ice_xo[n], ice_yo[n], '-', markersize=1, color=(n/len(ice_xo),0,1-n/len(ice_xo)) )

for n in range(len(ice_xs)):
    ax[1].plot( ice_xs[n], ice_ys[n], '-', markersize=1, color=(n/len(ice_xo),0,1-n/len(ice_xo)) )

ax[0].grid(True)    
ax[0].axis('equal')
ax[1].grid(True)    
ax[1].axis('equal')

ax[0].set_xlabel(r'$x$ (mm)')
ax[0].set_ylabel(r'$y$ (mm)')
ax[1].set_xlabel(r'$x$ (mm)')
ax[1].set_ylabel(r'$y$ (mm)')

filename = './Documents/Sphere_biblio/Figs/contours.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()

if fourier_smooth:
    ice_xo, ice_yo = ice_xs, ice_ys


#%%

t1 = time()

max_val = 30
ppmm = 10

x = np.linspace( -max_val, max_val, 2*max_val*ppmm+1  ,endpoint=True )

xgr, ygr = np.meshgrid( x,x[::-1] )
rgr, thegr = np.sqrt( xgr**2 + ygr**2 ), np.arctan2( ygr, xgr )

flat_t = np.array( [t for x, t in zip(ice_xo, ice_t) for _ in x] )
flat_x = np.concatenate(ice_xo)
flat_y = np.concatenate(ice_yo)

points = np.vstack(  (flat_x, flat_y) ).T

thing = griddata(points, flat_t-ice_t[0], (xgr,ygr), method='linear')

# thing2 = interpolate_gaussian(flat_x, flat_y, flat_t-ice_t[0], xgr, ygr, dist_threshold=10, sigmas=[0.6,0.6] )
# thing2 = interpolate_idw(flat_x, flat_y, flat_t-ice_t[0], xgr, ygr, dist_threshold=100, power=3)

 
mask, (x_mask,y_mask) = mask_ice_pos(xgr, ygr, ice_xo, ice_yo) 

t2 = time()

t2-t1

#%%

hol_xo, hol_yo = order_holder(hol_x, hol_y)
hol_xo, hol_yo = hol_xo-cx, hol_yo-cy

thing0 = np.copy(thing)
thing0[ np.isnan(thing0) ] = 0.

gthing = nangauss(thing0, 7 )
# gthing = nangauss(thing, 7 )

gy, gx = np.gradient( gthing, x, -x )
grad_abs = np.sqrt( gx**2 + gy**2 )

mthing = np.copy(gthing)
m_thing = np.copy(thing)
mgrad_abs = np.copy(grad_abs)

# mthing[~mask] = np.nan 
# m_thing[~mask] = np.nan 
# mgrad_abs[~mask] = np.nan 

fig, ax = plt.subplots(1,3, figsize=(16,6), sharey=True)

# ax[1].imshow( m_thing , extent=(-30,30,-30,30) )
ax[1].imshow( mthing , extent=(-30,30,-30,30) )

# ax[1].plot( hol_xo-cx, hol_yo-cy, 'g-' )
ax[1].plot( hol_xo, hol_yo, 'g-' )

ax[2].imshow( mgrad_abs , extent=(-30,30,-30,30) )

for n in range(0, len(ice_xo), 1):    
# for n in [0,len(ice_xo)-1]:    
    ax[0].plot( ice_xo[n], ice_yo[n], '-', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)) )
ax[0].imshow( mthing , extent=(-30,30,-30,30), alpha=0 )


# plt.savefig('./Documents/Sphere figures/natural_prof.pdf',dpi=400, bbox_inches='tight')
plt.show()
#%%

V0, A0 = 4/3 * np.pi * 30**3, 4 * np.pi * 30**2

vols, ars = [],[]
for i in range(len(ice_xo)):
    va,aa = area_vol(ice_xo[i], ice_yo[i])
    vols.append( va ); ars.append( aa )

vols, ars = np.array(vols) , np.array(ars) 


levels = ice_t[0:len(ice_xo)-0] - ice_t[0]
svo, sar, hov, har = calculate_areas_volumes_im2(xgr, ygr, mthing, levels )
svo[0], sar[0] = np.nan, np.nan
final = -1
svo[final:], sar[final:] = np.nan, np.nan

fig,ax = plt.subplots(1,3, layout='constrained' , figsize=(10,4) )

time_ice = ice_t - ice_t[0]
ax[0].plot( time_ice, (vols-hov)/V0, '.-' )
ax[0].plot( time_ice, (svo-hov)/V0, '.-' )

ax[1].plot( time_ice, (ars-har)/A0, '.-' )
ax[1].plot( time_ice, (sar-har)/A0, '.-' )

ax[2].plot( time_ice, np.gradient( (vols-hov)/V0, time_ice ), '.-' )
ax[2].plot( time_ice, np.gradient( (svo-hov)/V0, time_ice ), '.-' )

ax[0].set_ylim(-0.1,1.1)
ax[1].set_ylim(-0.1,1.1)

ax[0].grid()
ax[1].grid()
ax[2].grid()
plt.plot()

#%%
latent = 334e3 # m^2 / s^2 o J/kg
cp = 4184 # J/(kg°C)
rho_w = 997.98 # kg/m3
rho_i = 916.8 # kg / m^3 
k_th = 0.143e-6 # m^2/s
nu = 1e-6 #m^2/s

# This two values I should take from data
T0 = 21 #K or °C (it is difference with Tm=0°C)
U = 0.4 #m/s

time_ice = ice_t - ice_t[0]
V_ice, A_ice = (svo-hov) * 1e-9, (sar-har) * 1e-6
dV_ice = np.gradient( V_ice, time_ice )

Nu = - rho_i/rho_w * latent/(cp*T0) * 1/k_th * np.cbrt(V_ice) * dV_ice / A_ice
Re = U/nu * np.cbrt(V_ice)

plt.figure()

plt.plot(Re, Nu, '.-')

# plt.plot(Re, Re**(0.33), '-')
plt.plot(Re, Re**(0.5) * 1.2, '-')
# plt.plot(Re, Re**(1) * 0.01, '-')

plt.xscale('log')
plt.yscale('log')
plt.show()

#%%








#%%
from scipy.integrate import cumulative_trapezoid

def transform_angle( theta ):
    t_theta = theta + np.pi/2
    t_theta = np.where(t_theta > np.pi, t_theta - 2*np.pi, t_theta)
    return t_theta

def arc_length(x, y):
    if type(x) is list:
        s = []
        for i in range(len(x)):
            xg,yg = np.gradient( x[i] ), np.gradient(y[i] )
            ss = cumulative_trapezoid( np.sqrt(xg**2 + yg**2), initial=0 )
            s.append(ss)
        
    else:
        xg,yg = np.gradient( x ), np.gradient(y)
        s = cumulative_trapezoid( np.sqrt(xg**2 + yg**2), initial=0 )
    
    return s

def get_contours( xgr, ygr, tgr, levels ):

    x_conts, y_conts, th_conts = [],[],[]
    for lev in levels:
        cont = find_contours( tgr, level = lev )
        if len(cont)>0:
            yc, xc = np.vstack(cont).T
        
            x_contour = np.interp(xc, np.arange(len(xgr[0,:])), xgr[0,:])
            y_contour = np.interp(yc, np.arange(len(ygr[:,0])), ygr[:,0])
            theta = np.arctan2( y_contour, x_contour )
            
            sort = np.argsort( theta )
            x_conts.append(x_contour[sort]); y_conts.append(y_contour[sort]); th_conts.append(theta[sort]) 
        else: x_conts.append( np.array([np.nan]) ); y_conts.append( np.array([np.nan]) ), th_conts.append( np.array([np.nan]) )
    
    return x_conts, y_conts, th_conts

def divide_in_angle_time(n_a, n_t, thegr, tgr):
    """
    Divide a 2D grid into angular and time bins.

    Parameters
    ----------
    n_a : int
        Number of angular bins.
    n_t : int
        Number of time bins.
    thegr : ndarray
        2D array of angles in radians.
    tgr : ndarray
        2D array of time values.

    Returns
    -------
    div : ndarray
        Same shape as thegr/tgr. Each element contains the combined
        bin number, from 1 to n_a*n_t. NaN where tgr is NaN.

    acent : ndarray
        Centre angle of each angular bin.

    tcent : ndarray
        Centre time of each time bin.
    """

    div_a = np.floor( (thegr + np.pi) / (2 * np.pi / n_a) + 0.5 ).astype(int) % n_a 

    acent_a = np.linspace( -np.pi, np.pi, n_a, endpoint=False )

    # Time bins
    tmin, tmax = np.nanmin(tgr), np.nanmax(tgr)

    levels = np.linspace(tmin, tmax, n_t + 1)

    div_t = np.digitize( tgr, levels[1:-1], right=True )

    tcent_t = (levels[:-1] + levels[1:]) / 2
    
    # Create all combinations of (time, angle)
    acent, tcent = np.meshgrid( acent_a, tcent_t )
    acent = acent.ravel()
    tcent = tcent.ravel()

    # Combine angular and time bins
    div = div_t * n_a + div_a 

    # Preserve NaNs
    div = div.astype(float)
    div[np.isnan(tgr)] = np.nan

    return div, acent, tcent


def bin_stats(div, ar, acent, extra=5):
    """
    Calculate mean and standard deviation of `ar` for every bin.

    Parameters
    ----------
    div : ndarray
        Array containing the bin index for each element.
        Expected values are 0 ... n_bins-1.
    ar : ndarray
        Array of values to calculate statistics for.
        Same shape as div.
    acent : ndarray
        Angular centre for each bin.
    tcent : ndarray
        Time centre for each bin.
    extra: float
        Sets the angles at the top, between (90±extra)°, to np.nan

    Returns
    -------
    means : ndarray
        Mean of ar in each bin.
    stds : ndarray
        Standard deviation of ar in each bin.
    acent : ndarray
        Angular centre of each bin.
    tcent : ndarray
        Time centre of each bin.
    """
    n_bins = len(acent)

    bins, values = div.ravel(), ar.ravel()

    # Ignore NaNs in ar and div
    valid = (~np.isnan(values)) & (~np.isnan(bins))

    bins, values = bins[valid].astype(int), values[valid]

    # Number of values in each bin
    counts = np.bincount( bins, minlength=n_bins )

    # Sum of values and squares in each bin
    sums = np.bincount( bins, weights=values, minlength=n_bins )
    sums_sq = np.bincount( bins, weights=values**2, minlength=n_bins )

    # Mean
    means = np.divide( sums, counts, out=np.full(n_bins, np.nan), where=counts > 0 )

    # Variance
    variance = np.divide( sums_sq, counts, out=np.full(n_bins, np.nan), where=counts > 0 ) - means**2

    # Avoid tiny negative values due to floating-point errors
    variance = np.maximum(variance, 0)
    stds = np.sqrt(variance)

    top = (acent >= (90-extra)/180*np.pi) * (acent <= (90+extra)/180*np.pi) 
    means[top], stds[top] = np.nan, np.nan

    return means, stds 


#%%
hol_xo, hol_yo = order_holder(hol_x, hol_y)
hol_xo, hol_yo = hol_xo-cx, hol_yo-cy

thing0 = np.copy(thing)
thing0[ np.isnan(thing0) ] = 0.

gthing = nangauss(thing0, 7 )

#gradient
x = np.linspace( -max_val, max_val, 2*max_val*ppmm+1  ,endpoint=True )
gy, gx = np.gradient( gthing, x, -x, edge_order=2 )
grad_abs = np.sqrt( gx**2 + gy**2 )

# kurv
mgx, mgy = np.gradient(gthing, x, -x, edge_order=2)
mgxx, mgxy = np.gradient(mgx, x, -x, edge_order=2)
mgyx, mgyy = np.gradient(mgy, x, -x, edge_order=2)

kurv = -( mgxx * mgy**2 + mgyy * mgx**2 - 2 * mgxy * mgx * mgy ) / np.sqrt( mgx**2 + mgy**2 )**3
# kurv = np.abs( mgxx * mgy**2 + mgyy * mgx**2 - 2 * mgxy * mgx * mgy ) / np.sqrt( mgx**2 + mgy**2 )**3


m_thing = np.copy(gthing)
m_kurv = np.copy(kurv)
m_grad_abs = np.copy(grad_abs)

m_thing[~mask] = np.nan 
m_kurv[~mask] = np.nan 
m_grad_abs[~mask] = np.nan 


fig, ax = plt.subplots(1,3, figsize=(16,6), sharey=False, layout='constrained')

# ax[1].imshow( m_thing , extent=(-30,30,-30,30) )
im0 = ax[0].imshow( m_thing , extent=(-30,30,-30,30) )
# im0 = ax[0].imshow( gthing , extent=(-30,30,-30,30) )

im1 = ax[1].imshow( m_kurv, extent=(-30,30,-30,30) )

im2 = ax[2].imshow( 1/m_grad_abs, extent=(-30,30,-30,30) )


for i in range(3):
    ax[i].set_xlabel(r'$x$ (mm)')
    ax[i].set_ylabel(r'$y$ (mm)')

ax[0].set_title('Time (s)')
ax[1].set_title('Curvature (1/mm)')
ax[2].set_title('Melt rate (mm/s)')

fig.colorbar(im0, ax=ax[0], shrink=0.7, pad=0.03)
fig.colorbar(im1, ax=ax[1], shrink=0.7, pad=0.03)
fig.colorbar(im2, ax=ax[2], shrink=0.7, pad=0.03)

filename = './Documents/Sphere_biblio/Figs/quantities.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()


#%%

U_inf = np.median( cha_v ) # m/s
nu = 1e-6 # m^2/s

n_a = 96
n_t = 12

ar = m_kurv

k_div_all, k_acent, k_tcent = divide_in_angle_time(n_a, n_t, thegr, m_thing)
k_means, k_stds = bin_stats(k_div_all, ar, k_acent, extra=5)

unique_t, step_t = np.unique(k_tcent, return_inverse=True)
x_cont, y_cont, th_cont = get_contours( xgr, ygr, gthing, unique_t )


fig, axs = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': 'polar'}, layout='constrained')

axs[0].pcolormesh( thegr, rgr, m_thing, shading='auto' )
axs[0].set_ylim(0,32)

axs[1].pcolormesh( thegr, rgr, k_div_all, shading='auto', cmap='prism' )
axs[1].set_ylim(0,32)
# axs[1].set_rlabel_position(90)

# plt.show()

# fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': 'polar'}, layout='constrained')

axs[2].pcolormesh( thegr, rgr, ar, shading='auto' )
axs[2].set_ylim(0,32)

# axs[3].errorbar( acent, means, yerr=stds, fmt='.-', capsize=1 )
# axs[3].scatter( k_acent, k_means, s=15, c=k_tcent, cmap='viridis', zorder=5 )
for i in range(1,n_t-1,2):
    tval, argus = unique_t[i], np.where( step_t==i )[0]
    axs[3].plot( k_acent[argus], k_means[argus], '.-', color=(i/n_t,0,1-i/n_t) )

# axs[3].set_rlim(0,0.159)

# axs[3].set_rlabel_position(90)
for label in axs[3].get_yticklabels():
    label.set_horizontalalignment('center')

axs[0].set_title('Time (s)')
axs[1].set_title('Bins')
axs[2].set_title('Curvature (1/mm)')
axs[3].set_title('Mean Curvature (1/mm)')

for ax in axs:
    ax.set_rlabel_position(90)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('center')

    ax.set_xticks( np.arange(0,360,45) * np.pi/180 )
    ax.set_xticklabels(['90°','135°','180°','-135°','-90°','-45°','0°','45°'])

filename = './Documents/Sphere_biblio/Figs/kurv_spherical.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()

fig, ax = plt.subplots(1,3,layout='constrained', figsize=(15,5))

ax[0].axis('equal')
# for i in range(n_t):
for i in range(1,n_t-1,2):
    ax[0].plot( x_cont[i], y_cont[i], '-', color=(i/n_t,0,1-i/n_t) )
    
    tval, argus = unique_t[i], np.where( step_t==i )[0]
    t_k_acent, t_k_means = k_acent[argus], k_means[argus]

    ct_k_acent = transform_angle(t_k_acent)
    side1, side2 = (ct_k_acent <= 0 ), (ct_k_acent >= 0 )

    ang_side1, ang_side2 = np.abs(ct_k_acent[side1]), ct_k_acent[side2]
    t_means_side1, t_means_side2 = t_k_means[side1], t_k_means[side2]

    sort1, sort2 = np.argsort(ang_side1), np.argsort(ang_side2) 
    side_mean = ( t_means_side2[sort2][:-1] + t_means_side1[sort1]) / 2
    side_mean = np.concatenate( (side_mean, [t_means_side2[sort2][-1]]) )
    
    # ax[1].plot( ang_side1[sort1] * 180/np.pi, t_means_side1[sort1], '.-', color=(i/n_t,0,1-i/n_t) )
    # ax[1].plot( ang_side2[sort2] * 180/np.pi, t_means_side2[sort2], '.-', color=(i/n_t,0,1-i/n_t) )
    ax[1].plot( ang_side2[sort2] * 180/np.pi, side_mean, '.-', color=(i/n_t,0,1-i/n_t) ) #color=(0,i/n_t,0) )

    R = (np.nanmax(x_cont[i]) - np.nanmin(x_cont[i])) / 2
    a_k1 = t_means_side1 * R 
    a_k2 = t_means_side2 * R 
    side_ak = ( a_k2[sort2][:-1] + a_k1[sort1] ) / 2
    side_ak = np.concatenate( (side_ak, [a_k2[sort2][-1]]) )
    Re_t = U_inf * (R/1000) / nu # Now with transverse radius, could be diameter

    # ax[2].plot( ang_side1[sort1] * 180/np.pi, a_k1[sort1], '.-', color=(i/n_t,0,1-i/n_t) )
    # ax[2].plot( ang_side2[sort2] * 180/np.pi, a_k2[sort2], '.-', color=(i/n_t,0,1-i/n_t) )
    ax[2].plot( ang_side2[sort2] * 180/np.pi, side_ak, '.-', color=(i/n_t,0,1-i/n_t), label=f"Re = {Re_t:.0f}" )


ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[2].legend(loc=(1.,.2))
# fig.colorbar( im, ax=ax[1] )

ax[0].set_xlabel(r'$x$ (mm)')
ax[0].set_ylabel(r'$y$ (mm)')
ax[1].set_xlabel(r'$\theta$ (°)')
ax[1].set_ylabel(r'$k$ (1/mm)')
ax[2].set_xlabel(r'$\theta$ (°)')
ax[2].set_ylabel(r'$k R $')

filename = './Documents/Sphere_biblio/Figs/kurvatures.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()


#%%
"""
Should take the angle (for second plot) with the same center always, 
or with the center of each contour?
"""
U_inf = np.median( cha_v ) # m/s
nu = 1e-6 # m^2/s

n_a = 96 #36
n_t = 12

ar = 1/m_grad_abs

mr_div_all, mr_acent, mr_tcent = divide_in_angle_time(n_a, n_t, thegr, m_thing)
mr_means, mr_stds = bin_stats(mr_div_all, ar, mr_acent, extra=5)

unique_t, step_t = np.unique(mr_tcent, return_inverse=True)
x_cont, y_cont, th_cont = get_contours( xgr, ygr, gthing, unique_t )


fig, axs = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': 'polar'}, layout='constrained')

axs[0].pcolormesh( thegr, rgr, m_thing, shading='auto' )
axs[0].set_ylim(0,32)

axs[1].pcolormesh( thegr, rgr, mr_div_all, shading='auto', cmap='prism' )
axs[1].set_ylim(0,32)

# plt.show()

# fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': 'polar'}, layout='constrained')

axs[2].pcolormesh( thegr, rgr, ar, shading='auto' )
axs[2].set_ylim(0,32)

# axs[3].errorbar( acent, means, yerr=stds, fmt='.', capsize=1 )
# axs[3].scatter( mr_acent, mr_means, s=10, c=mr_tcent, cmap='viridis', zorder=5 )
for i in range(1,n_t-1,2):
    tval, argus = unique_t[i], np.where( step_t==i )[0]
    axs[3].plot( mr_acent[argus], mr_means[argus], '.-', color=(i/n_t,0,1-i/n_t) )


# axs[3].set_rlim(0,0.159)

axs[0].set_title('Time (s)')
axs[1].set_title('Bins')
axs[2].set_title('Melt rate (mm/s)')
axs[3].set_title('Mean melt rate (mm/s)')

for ax in axs:
    ax.set_rlabel_position(90)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('center')

    ax.set_xticks( np.arange(0,360,45) * np.pi/180 )
    ax.set_xticklabels(['90°','135°','180°','-135°','-90°','-45°','0°','45°'])

filename = './Documents/Sphere_biblio/Figs/mr_spherical.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()


fig, ax = plt.subplots(1,3,layout='constrained', figsize=(15,5))

ax[0].axis('equal')
# for i in range(n_t):
for i in range(1,n_t-1,2):
    ax[0].plot( x_cont[i], y_cont[i], '-', color=(i/n_t,0,1-i/n_t) )
    
    tval, argus = unique_t[i], np.where( step_t==i )[0]
    t_mr_acent, t_mr_means = mr_acent[argus], mr_means[argus]

    ct_mr_acent = transform_angle(t_mr_acent)
    side1, side2 = (ct_mr_acent <= 0 ), (ct_mr_acent >= 0 )

    ang_side1, ang_side2 = np.abs(ct_mr_acent[side1]), ct_mr_acent[side2]
    t_means_side1, t_means_side2 = t_mr_means[side1], t_mr_means[side2]

    sort1, sort2 = np.argsort(ang_side1), np.argsort(ang_side2) 
    side_mean = ( t_means_side2[sort2][:-1] + t_means_side1[sort1]) / 2
    side_mean = np.concatenate( (side_mean, [t_means_side2[sort2][-1]]) )
    
    # ax[1].plot( ang_side1[sort1] * 180/np.pi, t_means_side1[sort1], '.-', color=(i/n_t,0,1-i/n_t) )
    # ax[1].plot( ang_side2[sort2] * 180/np.pi, t_means_side2[sort2], '.-', color=(i/n_t,0,1-i/n_t) )
    ax[1].plot( ang_side2[sort2] * 180/np.pi, side_mean, '.-', color=(i/n_t,0,1-i/n_t) ) #color=(0,i/n_t,0) )

    R = (np.nanmax(x_cont[i]) - np.nanmin(x_cont[i])) / 2
    a_mr1 = t_means_side1 * R / nu / 1000000
    a_mr2 = t_means_side2 * R / nu / 1000000
    side_amr = ( a_mr2[sort2][:-1] + a_mr1[sort1] ) / 2
    side_amr = np.concatenate( (side_amr, [a_mr2[sort2][-1]]) )
    Re_t = U_inf * (R/1000) / nu # Now with transverse radius, could be diameter

    # ax[2].plot( ang_side1[sort1] * 180/np.pi, a_mr1[sort1], '.-', color=(i/n_t,0,1-i/n_t) )
    # ax[2].plot( ang_side2[sort2] * 180/np.pi, a_mr2[sort2], '.-', color=(i/n_t,0,1-i/n_t) )
    ax[2].plot( ang_side2[sort2] * 180/np.pi, side_amr, '.-', color=(i/n_t,0,1-i/n_t), label=f"Re = {Re_t:.0f}" )
    
    print(R)


ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[2].legend(loc=(1.,.2))
# fig.colorbar( im, ax=ax[1] )

ax[0].set_xlabel(r'$x$ (mm)')
ax[0].set_ylabel(r'$y$ (mm)')
ax[1].set_xlabel(r'$\theta$ (°)')
ax[1].set_ylabel(r'$\dot{R}$ (mm/s)')
ax[2].set_xlabel(r'$\theta$ (°)')
ax[2].set_ylabel(r'$\dot{R} R / \nu$')

filename = './Documents/Sphere_biblio/Figs/melt_rates.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()




#%%










#%%
from skimage.filters import sato
from skimage.morphology import skeletonize, binary_erosion, remove_small_objects
from skimage.measure import label, regionprops, regionprops_table


kurv = np.copy(m_kurv)
kurv[np.isnan(kurv)] = 0.


def find_streaks(A, sigmas=(1, 2, 3), threshold=0.1):
    # Work on log scale because your values span many orders of magnitude
    B = np.log1p(A / A.max())

    # Remove small-scale noise
    B = gaussian(B, sigma=0.5)

    # Detect bright ridge/tubular structures
    R = sato(B, sigmas=sigmas, black_ridges=False)

    # Normalize ridge response
    R /= R.max()

    # Keep significant ridges
    mask = R > threshold

    # Thin them to approximately one-pixel-wide centerlines
    skeleton = skeletonize(mask)

    return skeleton, R


eroded_mask = binary_erosion( mask, disk(7) )

ske,rr = find_streaks(kurv, sigmas=(1, 2, 3), threshold=0.07)
lske = label(ske)
area = regionprops_table(lske, properties={'area'} )['area']

l_branch = np.argsort(area)[-2:] +1
hku = (lske == l_branch[0]) + (lske == l_branch[1])

hku = remove_small_objects( hku * eroded_mask, 10, connectivity=2)

fig, ax = plt.subplots(2,2, layout='constrained', figsize=(8,8))

ax[0,0].imshow( kurv )
ax[0,1].imshow( eroded_mask*1. + mask )

ax[1,0].imshow( kurv )
ax[1,0].imshow( ske, alpha=0.2, cmap='gray' )

ax[1,1].imshow( hku )
# ax[1,1].imshow(  )

# ax[1].imshow( ukurv == mkh )
# ax[2].imshow( ukurv == mkv )

# plt.colorbar()
plt.show()



area

#%%









#%%









#%%
fig, ax = plt.subplots(1,2,layout='constrained', figsize=(10,5))

im = ax[0].scatter( k_acent * 180/np.pi, k_means, c=k_tcent, cmap='viridis' )
ax[0].set_xticks([-180,-90,0,90,180])

im = ax[1].scatter( mr_acent * 180/np.pi, mr_means, c=mr_tcent, cmap='viridis' )
ax[1].set_xticks([-180,-90,0,90,180])

fig.colorbar( im, ax=ax[1] )

ax[0].grid()
ax[1].grid()
ax[0].set_ylabel('Curvature (1/mm)')
ax[1].set_ylabel('Melt rate (mm/s)')
ax[0].set_xlabel('Angle (°)')
ax[1].set_xlabel('Angle (°)')

ax[1].set_yscale('log')

plt.show()


# plt.figure()
# plt.imshow(div_all, extent=(-30,30,-30,30), cmap='prism')
# plt.colorbar()
# plt.show()



#%%
# means = \dot{r}, Re = U_\infty 
U_inf = np.median( cha_v ) # m/s
nu = 1e-6 # m^2/s

unique_t, step_t = np.unique(mr_tcent, return_inverse=True)
x_cont, y_cont, th_cont = get_contours( xgr, ygr, gthing, unique_t )
s = arc_length(x_cont, y_cont)


fig, ax = plt.subplots(1,4, layout='constrained', figsize=(16,4) )
ax[0].axis('equal')

for i in range(n_t):
# for i in range(1):
    ax[0].plot( x_cont[i], y_cont[i], '-', color=(i/n_t,0,1-i/n_t) )

    f1, f2, f0 = th_cont[i] <= -np.pi/2, (th_cont[i] >= -np.pi/2) * (th_cont[i] <= 0), np.argmin( np.abs(th_cont[i] + np.pi/2) )
    s0 = s[i][f0]
    s1, s2 = s0 - s[i][f1], s[i][f2] - s0
    
    th1 = th_cont[i][f1]
    th2 = th_cont[i][f2]
    Re_s1 = U_inf * (s1/1000) / nu
    Re_s2 = U_inf * (s2/1000) / nu

    # ax[1].plot( th_cont[i]*180/np.pi, s[i], '-', color=(i/n_t,0,1-i/n_t) )
    # ax[1].plot( Re_s1, th1*180/np.pi, '-', color=(i/n_t,0,1-i/n_t) )
    # ax[1].plot( Re_s2, th2*180/np.pi, '-', color=(i/n_t,0,1-i/n_t) )
    
    # ax[1].plot( th1*180/np.pi, Re_s1, '-', color=(i/n_t,0,1-i/n_t) )
    # ax[1].plot( th2*180/np.pi, Re_s2, '-', color=(i/n_t,0,1-i/n_t) )
    
    tval, argus = unique_t[i], np.where( step_t==i )[0]
    R = (np.nanmax(x_cont[i]) - np.nanmin(x_cont[i])) / 2
    fil1, fil2 = mr_acent[argus] <= -np.pi/2, (mr_acent[argus] >= -np.pi/2) * (mr_acent[argus] <= 0)
    
    s_f1, s_f2 = np.interp( mr_acent[argus][fil1], th1, s1 ), np.interp( mr_acent[argus][fil2], th2, s2 )
    Re_sf1 = U_inf * (s_f1/1000) / nu 
    Re_sf2 = U_inf * (s_f2/1000) / nu

    a_mr1 = mr_means[argus][fil1] #* R / nu / 1000000
    a_mr2 = mr_means[argus][fil2] #* R / nu / 1000000
    
    # ax[1].plot( Re_sf1, a_mr1, 'o-', color=(i/n_t,0,1-i/n_t) )
    # ax[1].plot( Re_sf2, a_mr2, 's-', color=(i/n_t,0,1-i/n_t) )
    ax[1].plot( mr_acent[argus][fil1], a_mr1, 'o-', color=(i/n_t,0,1-i/n_t) )
    ax[1].plot( mr_acent[argus][fil1], a_mr2, 's-', color=(i/n_t,0,1-i/n_t) )

    a_mr1 = mr_means[argus][fil1] * R / nu / 1000000
    a_mr2 = mr_means[argus][fil2] * R / nu / 1000000
    
    ax[2].plot( mr_acent[argus][fil1], a_mr1, 'o-', color=(i/n_t,0,1-i/n_t) )
    ax[2].plot( mr_acent[argus][fil1], a_mr2, 's-', color=(i/n_t,0,1-i/n_t) )

    a_mr1 = mr_means[argus][fil1] * s_f1 / nu / 1000000
    a_mr2 = mr_means[argus][fil2] * s_f2 / nu / 1000000
    
    ax[3].plot( Re_sf1, a_mr1, 'o-', color=(i/n_t,0,1-i/n_t) )
    ax[3].plot( Re_sf2, a_mr2, 's-', color=(i/n_t,0,1-i/n_t) )

    # ax[1].scatter( tval, R, c=(i/n_t,0,1-i/n_t) )
    # ax[1].plot( mr_acent[argus][fil1], mr_means[argus][fil1], 'o-', color=(i/n_t,0,1-i/n_t) )
    # ax[1].plot( mr_acent[argus][fil2], mr_means[argus][fil2], 's-', color=(i/n_t,0,1-i/n_t) )

    # ax[1].plot( mr_acent[argus][fil1]*180/np.pi, Re_sf1, 'o', color=(i/n_t,0,1-i/n_t) )
    # ax[1].plot( mr_acent[argus][fil2]*180/np.pi, Re_sf2, 's', color=(i/n_t,0,1-i/n_t) )
    
dre = np.linspace(0,15000,100)
ax[3].plot( dre, (dre) /2000, 'k--', label=r'Re$^1$' )
ax[3].legend()

# ax[1].set_xscale('log')
# ax[1].set_yscale('log')
# # ax[2].set_xscale('log')
# ax[2].set_yscale('log')
# ax[3].set_xscale('log')
# ax[3].set_yscale('log')

ax[0].set_xlabel(r'$x$ (mm)')
ax[0].set_ylabel(r'$y$ (mm)')
ax[1].set_ylabel(r'$\dot{R}$')
ax[1].set_xlabel(r'$\theta$')
# ax[1].set_xlabel(r'Re')
ax[2].set_ylabel(r'$\dot{R} R / \nu$')
ax[2].set_xlabel(r'$\theta$')
# ax[2].set_xlabel(r'Re')
ax[3].set_ylabel(r'$\dot{R} s / \nu$')
ax[3].set_xlabel(r'Re')


plt.show()




#%%




#%%












#%%

# Boundary layer thickness, following Kundu 6th edition page 653 (Monkewit et al (2007) model?)

U=1
a=1

x = np.linspace(-5,5,51)
x,y = np.meshgrid(x,x[::-1])

r, th = np.sqrt(x**2+ y**2), np.arctan2( y, x ) 

ur, u0 = U * (1-a**2/r**2) * np.cos(th), -U * (1+a**2/r**2) * np.sin(th) 

# ux = ur * np.cos(th) - u0 * np.sin(th)
# uy = ur * np.sin(th) + u0 * np.cos(th)
ux = U - a**2/r**2 * U * np.cos(2*th)
uy = -a**2/r**2 * U * np.sin(2*th)

ux[r<2], uy[r<2] =0,0

plt.figure()
# plt.imshow( u0, extent=(-10,10,-10,10) )
# plt.imshow( ux, extent=(-10,10,-10,10) )

plt.quiver( x, y, ux, uy , np.sqrt(ux**2+uy**2), scale=50 )

circ =np.linspace(-np.pi,np.pi,10000)
plt.plot( a*np.cos(circ),  a*np.sin(circ), 'r-' )

plt.axis('equal')
plt.colorbar()
plt.show()

circx = np.linspace(0,2*a,1000)
curvx = np.linspace(0, np.pi*a, 10000)

circy = np.sqrt( 2*a*circx-circx**2 )
circym = -np.sqrt( 2*a*circx-circx**2 )
r0 = a * np.sin( curvx/a ) 

plt.figure()

plt.plot( circx, circy, '-' )
# plt.plot( circx, circym, '-' )
plt.plot( curvx, r0, '-' )

plt.axis('equal')
plt.grid()
plt.show()

#%%









#%%



#%%
# -------Functions-------
def distance(A, B):
    xA, yA = A
    xB, yB = B
    return np.sqrt((xB - xA)**2 + (yB - yA)**2)

def pixel_width(i, j, coeffs, ROI):
    x_left, _ = conversion_function_ROI(j-.5, i, coeffs, ROI)
    x_right, _ = conversion_function_ROI(j+.5, i, coeffs, ROI)
    return distance((x_left, 0), (x_right, 0))

def pixel_height(i, j, coeffs, ROI):
    _, y_bot = conversion_function_ROI(j, i-.5, coeffs, ROI)
    _, y_top = conversion_function_ROI(j, i+.5, coeffs, ROI)
    return distance((0, y_bot), (0, y_top))

def get_centroid2D(seg, coeffs, ROI):
    I, J = np.nonzero(seg)
    vol = np.zeros_like(I, dtype=np.float64)
    xC = np.zeros_like(I, dtype=np.float64)
    yC = np.zeros_like(I, dtype=np.float64)
    for k in range(len(I)):
        i, j = I[k], J[k]
        vol[k] = pixel_height(i, j, coeffs, ROI) * pixel_width(i, j, coeffs, ROI)
        xC[k], yC[k] = conversion_function_ROI(j, i, coeffs, ROI)
    tot_vol = np.sum(vol)
    return np.sum(xC*vol) / tot_vol, np.sum(yC*vol) / tot_vol

def get_volume(seg, coeffs, ROI, x0=None):
    I, J = np.nonzero(seg)
    if x0 == None:
        x0, y0 = get_centroid2D(seg, coeffs, ROI)
    vol = 0.
    for k in range(len(I)):
        i, j = I[k], J[k]
        x, y = conversion_function_ROI(j, i, coeffs, ROI)
        # Calculate local (cylindrical) radius
        r = distance((x, 0), (x0, 0))
        vol += np.pi * r * pixel_height(i, j, coeffs, ROI) * pixel_width(i, j, coeffs, ROI)
    return vol

def get_volume_v2(seg, coeffs, ROI, x0=None):
    if x0 == None:
        x0, y0 = get_centroid2D(seg, coeffs, ROI)

    vol = 0.
    seg = ski.morphology.binary_opening(seg, footprint=ski.morphology.footprint_rectangle((1, 2)))
    seg = ski.morphology.binary_closing(seg, footprint=ski.morphology.footprint_rectangle((1, 2)))
    float_seg = ski.util.img_as_float32(seg)
    for i, line in enumerate(float_seg):
        grad = np.gradient(line)
        edge = grad[:-1] + grad[1:]
        j_start = np.nonzero(edge > 0.7)[0]
        j_stop = np.nonzero(edge < -0.7)[0]
        if len(j_start) != len(j_stop):
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(line)), grad)
            ax.plot(np.arange(len(line)), line)
            plt.show()
        for k in range(len(j_start)):
            x_start, _ = conversion_function_ROI(j_start[k], i, coeffs, ROI)
            x_start = x_start - x0
            r_start = np.abs(x_start)
            x_stop, _ = conversion_function_ROI(j_stop[k], i, coeffs, ROI)
            x_stop = x_stop - x0
            r_stop = np.abs(x_stop)

            # Test if both points are on the same side of the center line
            if x_start * x_stop > 0:
                # Test wich side
                if x_start > 0:
                    vol += (r_stop**2 - r_start**2) * pixel_height(i, j_stop[k], coeffs, ROI)
                else:
                    vol += (r_start**2 - r_stop**2) * pixel_height(i, j_start[k], coeffs, ROI)
            else:
                vol += (r_stop**2 * pixel_height(i, j_stop[k], coeffs, ROI) + r_start**2 * pixel_height(i, j_start[k], coeffs, ROI))
    return 0.5 * np.pi * vol

def get_surface_area(contour, coeffs, ROI, x0):
    I, J = contour
    area = 0.
    if not I.size:
        return area
    for k in range(len(I)):
        i, j = I[k], J[k]
        x, y = conversion_function_ROI(j, i, coeffs, ROI)
        area += np.abs(x - x0) * pixel_height(i, j, coeffs, ROI) # not really satisfying
    return np.pi * area

def get_surface_area_v2(spl, coeffs, ROI, x0):
    def fx(u):
        x, y = spl(u)
        return x - x0
    # Test if the ice is on both sides of the center line
    # If not, the detected contour might just be a bubble on the holder
    if fx(0.25) * fx(0.75) > 0:
        return 0.
    # Find the curvilign absiss where the contour crosses the center line
    sol = root_scalar(fx, bracket=[0.25, 0.75], method='brentq')
    u0 = sol.root

    def fct2int(u):
        x, y = spl(u)
        dx, dy = spl.derivative(1)(u)
        return np.abs(x - x0) * np.sqrt(dx**2 + dy**2)
    int1, _ = quad(fct2int, 0, u0)
    int2, _ = quad(fct2int, u0, 1)
    return np.pi * (int1 + int2)

def get_centroid3D(seg, coeffs, ROI, x0=None):
    I, J = np.nonzero(seg)
    if x0 == None:
        x0, y0 = get_centroid2D(seg, coeffs, ROI)
    vol = np.zeros_like(I, dtype=np.float64)
    weighted_x = np.zeros_like(I, dtype=np.float64)
    weighted_y = np.zeros_like(I, dtype=np.float64)
    for k in range(len(I)):
        i, j = I[k], J[k]
        x, y = conversion_function_ROI(j, i, coeffs, ROI)
        # Calculate local (cylindrical) radius
        r = distance((x, 0), (x0, 0))
        dx = pixel_width(i, j, coeffs, ROI)
        dy = pixel_height(i, j, coeffs, ROI)
        vol[k] = np.pi * r * dx * dy
        weighted_x[k] = x * vol[k]
        weighted_y[k] = y * vol[k]
    tot_vol = np.sum(vol)
    return np.sum(weighted_x)/tot_vol, np.sum(weighted_y)/tot_vol


#%%
# -------Main program-------
if __name__ == "__main__":
    pass
    
    # The following lines are part of a test, and should not be of any use anymore
    exp = 'exp_25_07_01_04'
    run = 1
    img_name = 'DSC_6310.tiff'
    res_path = os.path.join('..', 'RESULTS', '{}_{:02d}'.format(exp, run))

    coeffs_array = np.load(os.path.join('..', 'DATA', exp, 'calibration', 'coeffs.npy'))
    ROI_path = os.path.join(res_path, 'ROI.json')
    with open(ROI_path, 'r') as f:
        current_ROI = json.load(f)

    seg_path = os.path.join(res_path, 'segmentation')
    

    image = ski.util.img_as_bool(ski.io.imread(os.path.join(seg_path, img_name)))
    """plt.imshow(image)
    test_line = image[1092]
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(test_line)), np.gradient(test_line))
    ax.plot(np.arange(len(test_line)), test_line)
    plt.show()"""
    holder = ski.util.img_as_bool(ski.io.imread(os.path.join(seg_path, 'holder.tiff')))
    holder = ski.morphology.binary_dilation(holder, footprint=ski.morphology.disk(8))

    ice = image & ~holder
    ice = ski.morphology.remove_small_objects(ice, min_size=10000)
    closed = ski.morphology.binary_closing(ice, footprint=ski.morphology.footprint_rectangle((1, 200)))

    """props = ski.measure.regionprops(ski.measure.label(closed))
    ic, jc = props[0].centroid
    xc1, yc1 = conversion_function_ROI(jc, ic, coeffs_array, current_ROI)"""
    xc2, yc2 = get_centroid2D(closed, coeffs_array, current_ROI)
    #xc3, yc3 = get_centroid3D(closed, coeffs_array, current_ROI)

    t1 = time()
    volume = get_volume(closed, coeffs_array, current_ROI, x0=xc2)
    print(volume)
    t2 = time()
    print(f'Get volume v1: {t2 - t1}')

    t3 = time()
    volume = get_volume_v2(closed, coeffs_array, current_ROI, x0=xc2)
    print(volume)
    t4 = time()
    print(f'Get volume v2: {t4 - t3}')

    I, J = np.nonzero(get_contour(closed))
    X, Y = conversion_function_ROI(J, I, coeffs_array, current_ROI)
    fig, ax = plt.subplots()
    ax.scatter(X, Y, c=['b' for k in X])
    #ax.scatter(xc1, yc1, c='r')
    #ax.scatter(xc2, yc2, c='k')
    #ax.scatter(xc3, yc3, c='g')
    ax.axis('equal')
    plt.show()
    