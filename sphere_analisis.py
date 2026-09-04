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
    df[2:-2] = (
        -f[4:] + 8*f[3:-1] - 8*f[1:-3] + f[:-4]
    ) / (12*dx)

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



fig, ax = plt.subplots(1,2,figsize=(10,6))

for n in range(len(ice_xo)):
    ax[0].plot( ice_xo[n], ice_yo[n], '-', markersize=1, color=(n/len(ice_xo),0,1-n/len(ice_xo)) )

for n in range(len(ice_xs)):
    ax[1].plot( ice_xs[n], ice_ys[n], '-', markersize=1, color=(n/len(ice_xo),0,1-n/len(ice_xo)) )

ax[0].grid(True)    
ax[0].axis('equal')
ax[1].grid(True)    
ax[1].axis('equal')

plt.show()

if fourier_smooth:
    ice_xo, ice_yo = ice_xs, ice_ys


#%%



t1 = time()

max_val = 30
ppmm = 10

x = np.linspace( -max_val, max_val, 2*max_val*ppmm+1  ,endpoint=True )

xgr, ygr = np.meshgrid( x,x[::-1] )


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

gthing = nangauss(thing, 7 )
gy, gx = np.gradient( gthing, x, -x )
grad_abs = np.sqrt( gx**2 + gy**2 )

mthing = np.copy(gthing)
m_thing = np.copy(thing)
mgrad_abs = np.copy(grad_abs)

mthing[~mask] = np.nan 
m_thing[~mask] = np.nan 
mgrad_abs[~mask] = np.nan 

fig, ax = plt.subplots(1,3, figsize=(16,6), sharey=True)

# ax[1].imshow( m_thing , extent=(-30,30,-30,30) )
ax[1].imshow( mthing , extent=(-30,30,-30,30) )

ax[1].plot( hol_xo-cx, hol_yo-cy, 'g-' )

ax[2].imshow( mgrad_abs , extent=(-30,30,-30,30) )

for n in range(0, len(ice_xo), 1):    
# for n in [0,len(ice_xo)-1]:    
    ax[0].plot( ice_xo[n], ice_yo[n], '-', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)) )
ax[0].imshow( mthing , extent=(-30,30,-30,30), alpha=0 )


# plt.savefig('./Documents/Sphere figures/natural_prof.pdf',dpi=400, bbox_inches='tight')
plt.show()

#%%

plt.figure()
plt.plot( hol_x, hol_y, '-')
plt.show()

hol_xo, hol_yo = order_holder(hol_x, hol_y)
plt.figure()
plt.plot( hol_xo, hol_yo, '-')
plt.show()


#%%

vols, ars = [],[]
for i in range(len(ice_xo)):
    va,aa = area_vol(ice_xo[i], ice_yo[i])
    vols.append( va ); ars.append( aa )

vols, ars = np.array(vols) , np.array(ars) 

levels = ice_t[1:len(ice_xo)-1] - ice_t[0]
svo, sar = calculate_areas_volumes_im(xgr, ygr, mthing, levels )
svo, sar = svo, sar 

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].plot( ice_t[:len(ice_xo)] - ice_t[0], vols / vols[0], '.-'  )
ax[0].plot( ice_t[1:len(ice_xo)-1] - ice_t[0], svo / vols[0], '.-'  )

ax[1].plot( ice_t[:len(ice_xo)] - ice_t[0], ars / ars[0] , '.-'  )
ax[1].plot( ice_t[1:len(ice_xo)-1] - ice_t[0], sar / ars[0] , '.-'  )

# op = np.linspace(18,200,4)
# ax[1].plot( op, -0.004*(op-18)+0.9 )

ax[0].set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$V/V_0$')

ax[1].set_xlabel(r'$t$ (s)')
ax[1].set_ylabel(r'$A/A_0$')
ax[1].set_ylim(-0.01,1.01)

# plt.savefig('./Documents/Sphere figures/natural_va.pdf',dpi=400, bbox_inches='tight')
plt.show() 

#%%


levels = [30*5]
num = 30

xcs, ycs = show_contours(xgr, ygr, mthing, levels , fmt='-')

plt.figure()
# plt.imshow( mthing, extent=(-30,30,-30,30)  )

plt.plot( ice_xo[num], ice_yo[num], '-')
plt.plot( xcs[0], ycs[0], '-')

plt.axis('equal')

plt.xlabel(r'$x$ (mm)')
plt.ylabel(r'$y$ (mm)')
# plt.savefig('./Documents/Sphere figures/natural_cont.pdf',dpi=400, bbox_inches='tight')
plt.show()


print( calculate_areas_volumes_im(xgr, ygr, mthing, levels ) )
print( area_vol(ice_xo[num], ice_yo[num]) )


#%%
mask, (x_mask,y_mask) = mask_ice_pos(xgr, ygr, ice_xo, ice_yo) 
xl = xgr[0,i]

def crosses_holder( hol_x , hol_y, ytop ):

    diff = hol_y - ytop
    cross = np.where( (diff[1:] * diff[:-1]) < 0 )[0]
    
    crosses = y_mask[cross] + (xl - x_mask[cross]) * (y_mask[cross+1] - y_mask[cross]) / (x_mask[cross+1] - x_mask[cross])
    
    return crosses

# def area_vol_holder(hol_xo, hol_yo, ice_xo, ice_yo):
#     vh, ah = [],[]
#     for i in range(len(ice_xo)):
#         ytop = ( ice_yo[i][0] + ice_yo[i][-1] ) / 2
#         fil = hol_yo<=ytop
        
#         vol,_ = area_vol(hol_xo[fil], hol_yo[fil])
#         vh.append(vol)

#         r = np.abs( hol_xo[fil][-1]-hol_xo[fil][0] ) / 2
#         are = np.pi * r**2
#         ah.append(are)

#%%
hol_xo, hol_yo = order_holder(hol_x, hol_y)
hol_xo, hol_yo = hol_xo-cx, hol_yo-cy

plt.figure()
plt.plot( hol_xo, hol_yo, '-')

vh, ah = [],[]
yts = []
for i in range(len(ice_xo)):
    ytop = ( ice_yo[i][0] + ice_yo[i][-1] ) / 2
    fil = hol_yo<=ytop
    yts.append(ytop)
    
    vol,_ = area_vol(hol_xo[fil], hol_yo[fil])
    vh.append(vol)
    
    r = np.abs( hol_xo[fil][-1]-hol_xo[fil][0] ) / 2
    are = np.pi * r**2
    ah.append(are)
        
    plt.plot( hol_xo[fil], hol_yo[fil], '-')
    
# plt.axis('equal')
plt.show()


vh, ah = np.array(vh), np.array(ah)

fig,ax=plt.subplots(1,2)
ax[0].plot( vh , '.-')
ax[1].plot( ah, '.-' )
# ax[2].plot( yts, '.-' )
plt.show()


#%%
# # for i in range(len(ice_xo)):
# for i in range(0,5):
#     ytop = ( ice_yo[i][0] + ice_yo[i][-1] ) / 2

#     print(  ytop )

plt.figure()
plt.plot( hol_xo[fil], hol_yo[fil], '-')
plt.plot( hol_xo[fil][0], hol_yo[fil][0], 'y.')
plt.plot( hol_xo[fil][-1], hol_yo[fil][-1], 'r.')
plt.show()

#%%
i = 0

mask, (x_mask,y_mask) = mask_ice_pos(xgr, ygr, ice_xo, ice_yo) 
xl = xgr[0,i]

ytop = ( ice_yo[i][0] + ice_yo[i][-1] ) / 2
fil = hol_yo<=ytop

diff = hol_yo - ytop
cross = np.where( (diff[1:] * diff[:-1]) < 0 )[0]

crosses = y_mask[cross] + (xl - x_mask[cross]) * (y_mask[cross+1] - y_mask[cross]) / (x_mask[cross+1] - x_mask[cross])

plt.figure()
plt.plot( x_mask, y_mask )
plt.imshow( mask, extent=(-30,30,-30,30) )
plt.show()



cross

#%%

# pruebas calculo area y coso

# t = np.linspace()
xp, d_x = np.linspace(-2,2,1001, retstep=True)
yp, d_y = np.linspace(-2,2,1001, retstep=True)
xp, yp = np.meshgrid(xp,-yp)

ey = 2.
tp = np.sqrt( xp**2 + (yp/ey)**2  )
mat = np.max(np.sqrt( xp**2 + yp**2 ))
tp[tp>1.5], tp[tp<0.3] = np.nan, np.nan
tp = tp

fig, ax = plt.subplots(1,3, figsize=(12,5))
ax[0].imshow(tp, extent=(-2,2,-2,2))

# levs = [0.4,0.7,1,1.2,1.4]
levs = [0.6,1]


ax[1].imshow(tp, extent=(-2,2,-2,2), cmap='gray')
cs = ax[1].contour(xp, yp, tp, levels=levs ) #np.arange(0,2,0.1))


vols2, ars2 = [], []
for i in range(len(cs.get_paths())):
    path = cs.get_paths()[i]
    verts = path.vertices
    x, y = verts[:, 0], verts[:, 1]
    
    ax[2].plot( x, y, '.' )
    ax[2].axis('equal')
    ax[2].grid()
    
    # ay, ax = (find_contours( tp, level = 1 )[0]).T
    # ax[2].plot( ax, ay, '-' )    
    
    V, A = area_vol(x, y)
    
    # fil = x>0
    # x,y = x[fil], y[fil]
    
    # if len(y)>1:
    #     dy, dx = np.gradient(y), np.gradient(x)
    #     # dy, dx = grad4(y,1), grad4(x, 1)
    #     I2 = np.pi * np.trapezoid( x**2 * dy )
    #     A2 = 2*np.pi * np.trapezoid( x * np.sqrt( dx**2 + dy**2) )
    #     # ax[2].plot(  dy , '.-' )
    # else: I2, A2 = 0,0
    
    # # ax[3].plot( (x[:-1]**2 + x[1:]**2) * (y[1:] - y[:-1]), '.-' )

    vols2.append(V)    
    ars2.append(A)    

plt.show()

vols2, ars2 = np.array(vols2), np.array(ars2)


# actual_vol = np.array( levs )**3 * 4/3 * np.pi
# actual_ar = np.array( levs )**2 * 4 * np.pi

actual_vol = np.array(levs)* np.array(levs)*1. * np.array(levs)*ey  * 4/3 * np.pi

p, a,b,c = 1.6075, np.array(levs), np.array(levs)*1., np.array(levs)*ey
actual_ar = 4*np.pi * np.power( (np.power(a,p)*np.power(b,p) + np.power(a,p)*np.power(c,p) + np.power(c,p)*np.power(b,p))/3, 1/p)
#np.array( levs ) * np.array( levs )*1. * np.array( levs )*1.2  * 4 * np.pi

V, A = calculate_areas_volumes_im(xp, yp, tp, levs) 

print(V)
print( vols2 )
print( actual_vol )

print()
print(A)
print( ars2 )
print( actual_ar )

print()
print( (1-vols2/actual_vol)*100, (1-ars2/actual_ar)*100  )



#%%

ay, ax = np.vstack(find_contours( tp, level = 1 )).T

x_contour = np.interp(ax, np.arange(len(xp[0])), xp[0])
y_contour = np.interp(ay, np.arange(len(yp[:,0])), yp[:,0])

plt.figure()
# plt.imshow(tp, extent=(-2,2,-2,2))

plt.plot(x, y, 's' )
plt.plot(x_contour, y_contour,'r' )
# plt.contour(xp, yp, tp, levels=[1], color='cyan' )

# plt.grid()
# plt.axis('equal')
plt.show()

#%%

show_contours(xp, yp, tp, [0.7,1.,0.9])








#%%





#%%













#%%

t1 = time()

max_val = 30
ppmm = 10

x = np.linspace( -max_val, max_val, 2*max_val*ppmm+1  ,endpoint=True )

xgr, ygr = np.meshgrid( x,x[::-1] )


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


# def area_vol_holder(hol_xo, hol_yo, ice_xo, ice_yo):
#     vh, ah = [],[]
#     for i in range(len(ice_xo)):
#         ytop = ( ice_yo[i][0] + ice_yo[i][-1] ) / 2
#         fil = hol_yo<=ytop
        
#         vol,_ = area_vol(hol_xo[fil], hol_yo[fil])
#         vh.append(vol)

#         r = np.abs( hol_xo[fil][-1]-hol_xo[fil][0] ) / 2
#         are = np.pi * r**2
#         ah.append(are)

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
# fig,ax = plt.subplots(1,2, figsize=(10,6), layout='constrained' )

# vh, ah = [],[]
# tgr = mthing
# for lev in levels:
#     cont = find_contours( tgr, level = lev )
#     if len(cont)>0:
#         yc, xc = np.vstack(cont).T

#         x_contour = np.interp(xc, np.arange(len(xgr[0,:])), xgr[0,:])
#         y_contour = np.interp(yc, np.arange(len(ygr[:,0])), ygr[:,0])

#         co = ( lev/levels[-1], 0, 1 - lev/levels[-1] )

#         ax[0].plot( x_contour, y_contour, '-', color = co, label=lev  )
        

#         cc, inside = curve_intersections(x_contour, y_contour, hol_xo, hol_yo)
#         if len(cc)>1:
#             xh_i, yh_i = np.concatenate(( [cc[0,0]], hol_xo[inside], [cc[1,0]] )), np.concatenate(( [cc[0,1]], hol_yo[inside], [cc[1,1]] ))
#             vol, are = area_vol_holder(xh_i, yh_i)
#             vh.append(vol); ah.append(are)
#         else: vh.append(np.nan); ah.append(np.nan)
#     else:
#         print(lev, 'no')

# ax[0].plot(hol_xo, hol_yo, 'g-')

# vh,ah = np.array(vh), np.array(ah)

# # ax[0].legend(loc=(0.9,0))
# ax[0].axis('equal')

# # ax[1].plot( vols, '.-', label='Original' )
# # ax[1].plot( svo, '.-' , label='Smooth')

# # ax[1].plot( vh/V0, '.-' , label='More Smooth', zorder=1)
# ax[1].plot( ah, '.-' , label='More Smooth', zorder=1)



# vh, ah = [],[]
# yts = []
# for i in range(len(ice_xo)):
#     ytop = ( ice_yo[i][0] + ice_yo[i][-1] ) / 2
#     fil = hol_yo<=ytop
#     yts.append(ytop)
    
#     vol,_ = area_vol(hol_xo[fil], hol_yo[fil])
#     vh.append(vol)
    
#     r = np.abs( hol_xo[fil][-1] - hol_xo[fil][0] ) / 2
#     are = np.pi * r**2
#     ah.append(are)
# vh,ah = np.array(vh), np.array(ah)


# # ax[1].plot( vh/V0, '.-' , label='Smooth', zorder=0)
# ax[1].plot( ah, '.-' , label='Smooth', zorder=0)
# ax[1].legend()

# plt.show()

#%%
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


t2 = time()

# cc, inside = curve_intersections(x_contour, y_contour, hol_xo, hol_yo)
# xh_i, yh_i = np.concatenate(( [cc[0,0]], hol_xo[inside], [cc[1,0]] )), np.concatenate(( [cc[0,1]], hol_yo[inside], [cc[1,1]] ))

t3 = time()

# print(inside)  # True = inside, False = outside


plt.figure()
plt.plot( x_contour, y_contour, 'b-' )

plt.plot( hol_xo, hol_yo, 'g-' )

# plt.plot( xh_i, yh_i, 'r-' )

# plt.plot( cc[:,0], cc[:,1], 'r.' )

plt.axis('equal')
plt.show()

# cc
t3-t2
#%%












#%%












#%%

xc0, yc0, r0 = 0, 0, 30
result = least_squares(circle_fit, [xc0, yc0, r0], args=(ice_x[0], ice_y[0]))

print(result)

cx, cy = result.x[0], result.x[1]
# cx, cy = np.median(ice_x[0]), np.median(ice_y[0]) 
# cxm, cym = np.median(ice_x[-1]), np.median(ice_y[-1]) 

rs = np.linspace(0, 35, 100)
thetas = np.array([0,15,30,45,60,75,90,105,120,135,150,165,180]) / 180*np.pi
lins = [ ( rs*np.cos(theta) , rs*np.sin(theta) ) for theta in thetas ]

plt.figure()
for n in range(0, len(ice_t), 3):
    plt.plot( ice_x[n] , ice_y[n] , '.', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t))  )

# plt.plot( hol_x, hol_y, 'g.', markersize=1  )

plt.plot( [0], [0], 'k.' )
# plt.plot( cx, cy, 'k.' )

for i in range(len(lins)):
    plt.plot( lins[i][0], lins[i][1], 'g-' )


plt.axis('equal')
plt.show()

#%%

print( np.max( ice_x[0] -cx), np.max( ice_y[0] -cy) )
print( np.min( ice_x[0] -cx), np.min( ice_y[0] -cy) )

t1 = time()

max_val = 30
ppmm = 5

x = np.linspace( -max_val, max_val, 2*max_val*ppmm+1  ,endpoint=True )

xgr, ygr = np.meshgrid( x,x[::-1] )

flat_t = np.array( [t for x, t in zip(ice_x, ice_t) for _ in x] )
flat_x = np.array( [x for sublist in ice_x for x in sublist] )
flat_y = np.array( [y for sublist in ice_y for y in sublist] )

points = np.vstack(  (flat_x-cx, flat_y-cy) ).T
thing = griddata(points, flat_t-ice_t[0], (xgr,ygr))

t2 = time()
t2-t1

#%%
gthing = nangauss(thing, 3 )

gy, gx = np.gradient( gthing, x, x )


plt.figure()
plt.imshow( gthing )
plt.colorbar()

plt.axis('equal')
plt.show()

# plt.figure()
# plt.imshow( gy, vmin=-50, vmax=50 )
# plt.colorbar()

# plt.axis('equal')
# plt.show()

# plt.figure()
# plt.imshow( gx, vmin=-50, vmax=50 )
# plt.colorbar()

# plt.axis('equal')
# plt.show()

plt.figure()
plt.imshow( np.sqrt(gx**2 + gy**2) )#, vmin=-50, vmax=50 )
plt.colorbar()

plt.axis('equal')
plt.show()



#%%

xc0, yc0, r0 = 0, 0, 30
result = least_squares(circle_fit, [xc0, yc0, r0], args=(ice_x[0], ice_y[0]))
cx, cy = result.x[0], result.x[1]

plt.figure()
for n in range(0, len(ice_t), 1):
    r = np.sqrt( (ice_x[n] - cx)**2 + (ice_y[n] - cy)**2 )
    theta = np.mod( np.arctan2( ice_y[n] - cy, ice_x[n] - cx ), np.pi * 2 )
    ordert = np.argsort(theta)
    theta, r = theta[ordert], r[ordert]
    
    coco = np.fft.rfft( r, n=len(r) )
    coco[20:] = 0

    ir = np.fft.irfft( coco, n=len(r) )
     

    # plt.plot( r*np.cos(theta), r*np.sin(theta), '.', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)) )
    # plt.plot( ice_x[n]-cx , ice_y[n]-cy , '.', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t))  )


    # plt.plot( theta, r, '-', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)) )
    plt.plot( theta, ir, '.', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)), alpha=0.3 )
    # plt.plot( np.real( coco ) , '-', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)) )

    # plt.plot( r*np.cos(theta) , r*np.sin(theta) , '.', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)) )
    # plt.plot( ir*np.cos(theta), ir*np.sin(theta), '.', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)), alpha=0.5  )



# plt.axis('equal')
plt.show()


#%%
import numpy as np

def normal_melt_rate(x_profiles, y_profiles, dt):
    """
    Compute normal melt rate between successive profiles.

    Parameters
    ----------
    x_profiles, y_profiles : list of ndarrays
        x_profiles[i] and y_profiles[i] define one interface.
    dt : float
        Time between profiles (scalar or list).

    Returns
    -------
    rates : list of ndarrays
        Normal melt rate for every profile except the last.
    """

    rates = []

    for k in tqdm(range(len(x_profiles)-1)):

        x1 = np.asarray(x_profiles[k])
        y1 = np.asarray(y_profiles[k])

        x2 = np.asarray(x_profiles[k+1])
        y2 = np.asarray(y_profiles[k+1])

        # Tangent
        # dx, dy = np.gradient(x1), np.gradient(y1)
        dx, dy = grad4(x1,1), grad4(y1,1)

        L = np.sqrt(dx**2 + dy**2)

        tx,ty = dx/L, dy/L

        # Unit normal
        nx, ny = -ty, tx

        rate = np.empty(len(x1))

        pts2 = np.column_stack((x2, y2))

        for i in range(len(x1)):

            p = np.array([x1[i], y1[i]])

            # Vector to every point on next profile
            d = pts2 - p

            # Distance along normal
            normal_dist = d @ np.array([nx[i], ny[i]])

            # Perpendicular distance from normal line
            tangent_dist = np.abs(d @ np.array([tx[i], ty[i]]))

            # Point closest to the normal ray
            j = np.argmin(tangent_dist)

            rate[i] = normal_dist[j] / dt

        rates.append(rate)

    return rates
#%%


dt = np.diff(ice_t)[0]

xc0, yc0, r0 = 0, 0, 30
result = least_squares(circle_fit, [xc0, yc0, r0], args=(ice_x[0], ice_y[0]))
cx, cy = result.x[0], result.x[1]

thetas, rs = [], []
for n in range(0, len(ice_t), 1):
    r = np.sqrt( (ice_x[n] - cx)**2 + (ice_y[n] - cy)**2 )
    theta = np.mod( np.arctan2( ice_y[n] - cy, ice_x[n] - cx ), np.pi * 2 )
    ordert = np.argsort(theta)
    theta, r = theta[ordert], r[ordert]
    
    coco = np.fft.rfft( r, n=len(r) )
    coco[20:] = 0

    ir = np.fft.irfft( coco, n=len(r) )
    thetas.append( theta )
    rs.append( ir )


melt_rates = normal_melt_rate(thetas, rs, dt) 

#%%

vmin,vmax = min(np.min(r) for r in melt_rates), max(np.max(r) for r in melt_rates)

fig, ax = plt.subplots()
for n in range(0, len(ice_t)-1, 1):

    # plt.plot( thetas[n], rs[n], '.', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)), alpha=0.3 )
    # plt.plot( rs[n]*np.cos(thetas[n]), rs[n]*np.sin(thetas[n]), '.', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)), alpha=0.5  )

    sc = plt.scatter( thetas[n], rs[n], s=1, c=melt_rates[n], cmap='jet', vmin=vmin, vmax=vmax )

cbar = fig.colorbar(sc, ax=ax)
# plt.axis('equal')
plt.show()

#%%



x1 = np.asarray(thetas[0])
y1 = np.asarray(rs[0])

x2 = np.asarray(thetas[1])
y2 = np.asarray(rs[1])

x1g, y1g = gaussian_filter1d(x1, 2), gaussian_filter1d(y1, 2)

# Tangent
dx, dy = np.gradient(x1g), np.gradient(y1g)
# dx, dy = grad4(x1,1), grad4(y1,1)

L = np.sqrt(dx**2 + dy**2)

tx,ty = dx/L, dy/L #tangential
nx, ny = -ty, tx #normal

rate = np.empty(len(x1))

pts2 = np.column_stack((x2, y2))

for i in range(len(x1)):

    p = np.array([x1[i], y1[i]])

    # Vector to every point on next profile
    d = pts2 - p

    # Distance along normal
    normal_dist = d @ np.array([nx[i], ny[i]])

    # Perpendicular distance from normal line
    tangent_dist = np.abs(d @ np.array([tx[i], ty[i]]))

    # Point closest to the normal ray
    j = np.argmin(tangent_dist)

    rate[i] = normal_dist[j] / dt


fig, ax = plt.subplots()
sc = plt.scatter( x1, y1, s=1, c=rate, cmap='jet') #, vmin=vmin, vmax=vmax )
sv = plt.scatter( x2, y2, s=1 ) #, vmin=vmin, vmax=vmax )
cbar = fig.colorbar(sc, ax=ax)
plt.show()


#%%
def normal_line(nx,ny, orig, ini=0,fin=1,ll=3):
    ls = np.linspace(ini,fin,ll)
    linesx,linesy = [],[]
    for i in range(len(nx)):
        xi,yi = orig[i]
        xn,yn = nx[i], ny[i]
        
        lx, ly = xn*ls + xi, yn*ls + yi
        linesx.append(lx); linesy.append(ly) 
    return linesx,linesy 

ponts = np.vstack((x1g,y1g)).T
linesx,linesy = normal_line(nx, ny, ponts, fin=0., ini=-0.1)

# x1g, y1g = gaussian_filter1d(x1, 1), gaussian_filter1d(y1, 1)


fig, ax = plt.subplots()
# sc = plt.scatter( x1, y1, s=1 ) #, c=rate, cmap='jet') #, vmin=vmin, vmax=vmax )
sc = plt.plot( x1, y1, '-' ) #, c=rate, cmap='jet') #, vmin=vmin, vmax=vmax )
sc = plt.plot( x2, y2, '-' ) #, c=rate, cmap='jet') #, vmin=vmin, vmax=vmax )

# plt.plot( x1g, y1g, '-', alpha=0.9 ) #, c=rate, cmap='jet') #, vmin=vmin, vmax=vmax )

for i in range(0,len(ponts),10):
# for i in [3]:
    plt.plot( linesx[i], linesy[i], 'r-' )

plt.axis('equal')
plt.show()

# plt.figure()
# plt.plot( ty, '.' )
# plt.show()

#%%







#%%



dt = np.diff(ice_t)[0]

xc0, yc0, r0 = 30, -30, 30
result = least_squares(circle_fit, [xc0, yc0, r0], args=(ice_x[0], ice_y[0]))
cx, cy, rc = result.x[0], result.x[1], result.x[2] 

thetas, rs = [], []
ice_sx, ice_sy = [], []
ice_xo, ice_yo = [], []
for n in range(0, len(ice_t), 1):
    r = np.sqrt( (ice_x[n] - cx)**2 + (ice_y[n] - cy)**2 )
    theta = np.mod( np.arctan2( ice_x[n] - cx, ice_y[n] - cy ), np.pi * 2 ) 
    ordert = np.argsort(theta)
    theta, r = theta[ordert], r[ordert]
    
    coco = np.fft.rfft( r, n=len(r) )
    coco[20:] = 0

    ir = np.fft.irfft( coco, n=len(r) )
    thetas.append( theta )
    rs.append( ir )
    ice_sx.append( ir * np.sin(theta) )
    ice_sy.append( ir * np.cos(theta) )
    ice_xo.append( ice_x[n][ordert] - cx )
    ice_yo.append( ice_y[n][ordert] - cy )



tt = np.linspace(0,2*np.pi, 10000)

fig, ax = plt.subplots(1,2, figsize=(8,5))
for n in range(0, len(ice_t), 1):
    
    # ax[0].plot( ice_x[n], ice_y[n], '.', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)) )
    ax[0].plot( ice_x[n]-cx, ice_y[n]-cy, '.', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)) )
    
    ax[1].plot( ice_sx[n], ice_sy[n], '-', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)) )
    # ax[1].plot( ice_sx[n]+cx, ice_sy[n]+cy, '-', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)) )


# ax[0].plot(cx,cy, 'c.')
# ax[0].plot(rc*np.cos(tt)+cx, rc*np.sin(tt)+cy, 'g-'  ) 

ax[0].grid(True)    
ax[0].axis('equal')

ax[1].grid(True)    
ax[1].axis('equal')


plt.show()



#%%

def mask_far_points(ice_sx, ice_sy, thing, xgr, ygr, max_dist):
    """
    Replace values in `thing` with np.nan if the corresponding grid point
    is farther than `max_dist` from any original data point.

    Parameters
    ----------
    ice_sx, ice_sy : list of arrays
        Original x and y coordinates.
    thing : 2D ndarray
        Interpolated grid.
    xgr, ygr : 2D ndarray
        Meshgrid corresponding to `thing`.
    max_dist : float
        Maximum allowed distance from an original point.

    Returns
    -------
    masked : 2D ndarray
        Copy of `thing` with distant points set to np.nan.
    """

    # Flatten the original coordinates
    flat_x = np.concatenate(ice_sx)
    flat_y = np.concatenate(ice_sy)
    points = np.column_stack((flat_x, flat_y))

    # Build nearest-neighbor search tree
    tree = cKDTree(points)

    # Query every grid point
    grid_points = np.column_stack((xgr.ravel(), ygr.ravel()))
    dist, _ = tree.query(grid_points)

    # Create mask
    mask = dist.reshape(thing.shape) > max_dist

    # Apply mask
    masked = thing.astype(float).copy()
    masked[mask] = np.nan

    return masked

def mask_ice_pos( xgr, ygr, ice_sx, ice_sy,   ):

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



t1 = time()

max_val = 30
ppmm = 10

x = np.linspace( -max_val, max_val, 2*max_val*ppmm+1  ,endpoint=True )

xgr, ygr = np.meshgrid( x,x[::-1] )

# flat_t = np.array( [t for x, t in zip(ice_sx, ice_t) for _ in x] )
# flat_x = np.concatenate(ice_sx)
# flat_y = np.concatenate(ice_sy)

flat_t = np.array( [t for x, t in zip(ice_xo, ice_t) for _ in x] )
flat_x = np.concatenate(ice_xo)
flat_y = np.concatenate(ice_yo)

points = np.vstack(  (flat_x, flat_y) ).T

thing = griddata(points, flat_t-ice_t[0], (xgr,ygr), method='linear')

# mask, (x_mask,y_mask) = mask_ice_pos(xgr, ygr, ice_sx, ice_sy) 
mask, (x_mask,y_mask) = mask_ice_pos(xgr, ygr, ice_xo, ice_yo) 

t2 = time()

t2-t1

#%%

gthing = nangauss(thing, 10 )
gy, gx = np.gradient( gthing, x, -x )
grad_abs = np.sqrt( gx**2 + gy**2 )

mthing = np.copy(gthing)
mgrad_abs = np.copy(grad_abs)

mthing[~mask] = np.nan 
mgrad_abs[~mask] = np.nan 

fig, ax = plt.subplots(1,4, figsize=(16,6), sharey=True)

ax[2].imshow( mthing , extent=(-30,30,-30,30) )

ax[3].imshow( np.log(1/mgrad_abs) , extent=(-30,30,-30,30) )

for n in range(0, len(ice_t), 1):    
    ax[1].plot( ice_sx[n], ice_sy[n], '-', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)) )
ax[1].imshow( mthing , extent=(-30,30,-30,30), alpha=0 )

for n in range(0, len(ice_t), 1):    
    ax[0].plot( ice_x[n]-cx, ice_y[n]-cy, '.', markersize=1, color=(n/len(ice_t),0,1-n/len(ice_t)) )
ax[0].imshow( mthing , extent=(-30,30,-30,30), alpha=0 )

plt.show()


# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# surf = ax.plot_surface(xgr, ygr, mthing, cmap='coolwarm', linewidth=0, antialiased=True )

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('t')

# plt.show()


#%%


x_mask = np.concatenate( [ice_xo[0], ice_xo[-1][::-1], [ice_xo[0][0]] ] ) 
y_mask = np.concatenate( [ice_yo[0], ice_yo[-1][::-1], [ice_yo[0][0]] ] ) 

mask = np.zeros_like(ygr, dtype=bool)

for i in tqdm(range(len(xgr[0,:]))):
    
    xl = xgr[0,i] 
    diff = x_mask - xl 
    cross = np.where( (diff[1:] * diff[:-1]) < 0 )[0]
    
    crosses = y_mask[cross] + (xl - x_mask[cross]) * (y_mask[cross+1] - y_mask[cross]) / (x_mask[cross+1] - x_mask[cross])
    
    for j in range(len(crosses)):
        fil = ygr[:,0]>crosses[j]
        mask[fil,i] = ~ mask[fil,i]
        
        

plt.figure()
plt.imshow( mask, extent=(-30,30,-30,30), alpha=1 )
plt.plot( x_mask, y_mask , '.-' )

plt.show()



#%%





#%%





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
    