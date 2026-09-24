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

from skimage.morphology import local_minima, disk, remove_small_holes, binary_erosion, binary_dilation, binary_closing, binary_opening 

from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import networkx as nx
from scipy.signal import convolve2d, savgol_filter

# import skimage as ski
# from scipy.optimize import root_scalar
# from scipy.integrate import quad

# import os
# import json

# from datetime import timedelta
from time import time

import os 
import sys
sys.path.append('./Documents/Projects/Sphere-melting/')
import bl_solver


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


def nangauss_boundary_preserving(altus, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):    
        # 1. Create masks for valid data and true interior data
        valid_mask = ~np.isnan(altus)
        
        # Shrink the valid mask by 1 pixel to find the interior
        # Pixels that are valid but NOT in the interior are the boundary edges
        interior_mask = snd.binary_erosion(valid_mask)
        edge_mask = valid_mask & ~interior_mask
        
        # 2. Standard normalized convolution (your original logic)
        V = altus.copy()
        V[~valid_mask] = 0
        W = np.zeros_like(altus) + 1
        W[~valid_mask] = 0
        
        VV = snd.gaussian_filter(V, sigma)
        WW = snd.gaussian_filter(W, sigma)
        gdp = VV / WW
        
        # 3. Apply the boundary preservation rules
        # Start with a copy of the original array (keeps NaNs and original values)
        result = altus.copy()
        
        # ONLY update the true interior pixels with the blurred data
        result[interior_mask] = gdp[interior_mask]
        
        # Explicitly ensure the edge_mask pixels keep their exact original values
        result[edge_mask] = altus[edge_mask]
        
        return result
    
def nangauss_smooth_boundary(altus, sigma, transition_width=None):
    """
    Blurs valid data while preserving the exact boundary values.
    The interior smoothly transitions to the boundary values to prevent jumps.
    
    transition_width: How many pixels wide the smooth transition zone should be.
                      Defaults to roughly 2 * sigma if not specified.
    """
    if transition_width is None:
        transition_width = max(1.0, 2.0 * sigma)
        
    with np.errstate(divide='ignore', invalid='ignore'):    
        # 1. Identify valid data regions
        valid_mask = ~np.isnan(altus)
        
        # 2. Standard normalized convolution (your original logic)
        V = altus.copy()
        V[~valid_mask] = 0
        W = np.zeros_like(altus) + 1
        W[~valid_mask] = 0
        
        VV = snd.gaussian_filter(V, sigma)
        WW = snd.gaussian_filter(W, sigma)
        gdp = VV / WW
        
        # 3. Calculate distance from the NaN boundary
        # distance_transform_edt computes the distance from the nearest False (NaN) pixel
        distance_from_nan = snd.distance_transform_edt(valid_mask)
        
        # 4. Create a blending weight matrix (0.0 at edge, 1.0 deep inside)
        # We subtract 1 so that the exact boundary pixel (distance == 1) has a weight of 0.0
        weight = (distance_from_nan - 1) / transition_width
        weight = np.clip(weight, 0.0, 1.0) # Clamp between 0 and 1
        
        # 5. Smoothly blend the original data and the blurred data
        # At weight = 0 (the edge): result is 100% original altus
        # At weight = 1 (deep interior): result is 100% blurred gdp
        result = (1.0 - weight) * altus + weight * gdp
        
        # Restore the original NaNs to the empty zones
        result[~valid_mask] = np.nan
        
        return result



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



def mask_ice_pos( xgr, ygr, ice_sx, ice_sy, hol_c=False, disable_t=True, min_y=0 ):

    if not hol_c:
        x_mask = np.concatenate( [ice_sx[0], ice_sx[-1][::-1], [ice_sx[0][0]] ] ) 
        y_mask = np.concatenate( [ice_sy[0], ice_sy[-1][::-1], [ice_sy[0][0]] ] ) 
        
        mask = np.zeros_like(ygr, dtype=bool)
    
        for i in tqdm(range(len(xgr[0,:])), disable=disable_t):
            
            xl = xgr[0,i] 
            diff = x_mask - xl 
            cross = np.where( (diff[1:] * diff[:-1]) < 0 )[0]
            
            crosses = y_mask[cross] + (xl - x_mask[cross]) * (y_mask[cross+1] - y_mask[cross]) / (x_mask[cross+1] - x_mask[cross])
            
            for j in range(len(crosses)):
                fil = ygr[:,0]>crosses[j]
                mask[fil,i] = ~ mask[fil,i]
                
        return mask, 0, (x_mask,y_mask)
                
    elif hol_c:
        x_mask_out = np.concatenate( [ice_sx[0], [ice_sx[0][0]] ] ) 
        y_mask_out = np.concatenate( [ice_sy[0], [ice_sy[0][0]] ] ) 
        
        x_mask_in = np.concatenate( [ice_sx[-1], [ice_sx[-1][0]] ] ) 
        y_mask_in = np.concatenate( [ice_sy[-1], [ice_sy[-1][0]] ] ) 
        
        mask = np.zeros_like(ygr, dtype=bool)
        mask_inner = np.zeros_like(ygr, dtype=bool)
        mask_outer = np.zeros_like(ygr, dtype=bool)
    
        for i in tqdm(range(len(xgr[0,:])), disable=disable_t):
            
            xl = xgr[0,i] 

            diff = x_mask_out - xl 
            cross = np.where( (diff[1:] * diff[:-1]) < 0 )[0]            
            crosses_out = y_mask_out[cross] + (xl - x_mask_out[cross]) * (y_mask_out[cross+1] - y_mask_out[cross]) / (x_mask_out[cross+1] - x_mask_out[cross])
            
            diff = x_mask_in - xl 
            cross = np.where( (diff[1:] * diff[:-1]) < 0 )[0]            
            crosses_in = y_mask_in[cross] + (xl - x_mask_in[cross]) * (y_mask_in[cross+1] - y_mask_in[cross]) / (x_mask_in[cross+1] - x_mask_in[cross])
            
            crosses = np.sort( np.concatenate((crosses_out, crosses_in)) )
                
            for j in range(len(crosses)):
                fil = ygr[:,0]>crosses[j]
                mask[fil,i] = ~ mask[fil,i]        

            for j in range(len(crosses_in)):
                fil = ygr[:,0]>crosses_in[j]
                mask_inner[fil,i] = ~ mask_inner[fil,i]        

            for j in range(len(crosses_out)):
                fil = ygr[:,0]>crosses_out[j]
                mask_outer[fil,i] = ~ mask_outer[fil,i]        

        mask1 = (xgr > -2.5) * (xgr < 2.5) * (ygr>min_y)
        mask = (mask*1. - mask1)>0
    
        return mask, mask_outer, mask_inner, (0,0)


def sort_contour_graph(x, y, k=5):
    points = np.column_stack((x, y))

    # Build k-nearest-neighbor graph
    G = kneighbors_graph( points, n_neighbors=k, mode="distance", include_self=False ).toarray()

    n = len(points)

    # Start from an arbitrary point
    current = 0
    order = [current]
    unused = set(range(n))
    unused.remove(current)

    while unused:
        # Neighbors of current that haven't been visited
        candidates = [j for j in unused if G[current, j] > 0]

        if candidates:
            # Take closest connected point
            next_point = min(candidates, key=lambda j: G[current, j])
        else:
            # If graph connection is exhausted, jump to nearest unused point
            next_point = min( unused, key=lambda j: np.sum((points[j] - points[current])**2) )

        order.append(next_point)
        unused.remove(next_point)
        current = next_point

    order = np.asarray(order)

    return x[order], y[order], order



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
        # _, _, ordert = sort_contour_graph( ice_x[n] - cx, ice_y[n] - cy  )
        
        
        theta, r = theta[ordert], r[ordert]
        coco = np.fft.rfft( r, n=len(r) )
        coco[modes:] = 0
        ir = np.fft.irfft( coco, n=len(r) )

        # if np.min( ice_y[n][ordert] - cy ) > 0: break
        
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


def calculate_areas_volumes_im2( xgr, ygr, tgr, hol_xo, hol_yo, levels, order_grad=2 ):
    
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


def get_contours( xgr, ygr, tgr, levels ):

    x_conts, y_conts, th_conts = [],[],[]
    for lev in levels:
        cont = find_contours( tgr, level = lev )
        if len(cont)>0:
            yc, xc = np.vstack(cont).T
        
            x_contour = np.interp(xc, np.arange(len(xgr[0,:])), xgr[0,:])
            y_contour = np.interp(yc, np.arange(len(ygr[:,0])), ygr[:,0])
            theta = np.arctan2( y_contour, x_contour )
            
            # sort = np.argsort( theta )
            _, _, sort = sort_contour_graph( x_contour, y_contour, k=5)
            
            x_conts.append(x_contour[sort]); y_conts.append(y_contour[sort]); th_conts.append(theta[sort]) 

        else: x_conts.append( np.array([np.nan]) ); y_conts.append( np.array([np.nan]) ), th_conts.append( np.array([np.nan]) )
    
    return x_conts, y_conts, th_conts


def make_circle_plot( axs, radii_ticks = [10,20,30], angle_tick_dist=45 ):

    for ax in axs.flatten(): 
        ax.axis('off')
        
        padding = 1
        ax.set_xlim(-max_val - padding, max_val + padding)
        ax.set_ylim(-max_val - padding, max_val + padding)
        
        angles = np.linspace(0, 2*np.pi, 200)
        
        for r in radii_ticks:
            ax.plot(r * np.cos(angles), r * np.sin(angles), color='gray', linestyle='--', linewidth=0.5)
            ax.text(0, r, f' {int(r)}', color='gray',  ha='center', va='top') #fontsize=8,
    
        theta_ticks = np.arange(180, -180, -angle_tick_dist)
        for deg in theta_ticks:
            rad = np.radians(deg - 90)
            ax.plot([0, max_val * np.cos(rad)], [0, max_val * np.sin(rad)], color='gray', linestyle='--', linewidth=0.5)
            ax.text(max_val * 1.08 * np.cos(rad), max_val * 1.08 * np.sin(rad), f'{deg}°',  color='black', ha='center', va='center') # fontsize=9
    
        circle_clip = plt.Circle((0, 0), max_val, transform=ax.transData, facecolor='none', edgecolor='black', linewidth=1.5)
        ax.add_patch(circle_clip)
    

# def 2d_area( xgr, ygr, tgr ):
#     Vs, As, Vh, Ah = [],[], [], []
#     for lev in levels:
#         cont = find_contours( tgr, level = lev )
#         if len(cont)>0:
#             yc, xc = np.vstack(cont).T
        
#             x_contour = np.interp(xc, np.arange(len(xgr[0,:])), xgr[0,:])
#             y_contour = np.interp(yc, np.arange(len(ygr[:,0])), ygr[:,0])


# os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
# plt.rcParams.update({ # 'axes.edgecolor': 'black',
#                         "text.usetex": True,
#                         "font.family": "serif",
#                         'font.size':12, })

#%%


path = '/Volumes/Ice blocks/Sphere channel/Results/'
files = glob.glob( path + '*.hdf5' )


t1 = time()

# cha_t, cha_s, cha_v, cha_Tb, cha_Tt, hol_x, hol_y, ice_t, ice_x, ice_y = get_data(files[1])
cha_t, cha_s, cha_v, cha_Tb, cha_Tt, hol_x, hol_y, ice_t, ice_x, ice_y = get_data(files[4])
# cha_t, cha_s, cha_v, cha_Tb, cha_Tt, hol_x, hol_y, ice_t, ice_x, ice_y = get_data(files[-1])

t2 = time()

truncate_smooth = True

dt = np.diff(ice_t)[0]

ice_xo, ice_yo, cx, cy = order_data(ice_x, ice_y)
ice_xo, ice_yo = ordenar_contornos(ice_xo, ice_yo)
ice_xs, ice_ys = keep_inside_previous(ice_xo, ice_yo, lag=1, tol=.2) 

t3 = time()
print(t3-t2, t2-t1, t3-t1)

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

if truncate_smooth:
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

# mask, (x_mask,y_mask) = mask_ice_pos(xgr, ygr, ice_xo, ice_yo) 
mask, mask_out, mask_in, (x_mask,y_mask) = mask_ice_pos(xgr, ygr, ice_xo, ice_yo,  hol_c=True, min_y=2) 

t2 = time()
print(t2-t1)
#%%
t2 = time()

hol_xo, hol_yo = order_holder(hol_x, hol_y)
hol_xo, hol_yo = hol_xo-cx, hol_yo-cy

# thing0 = np.copy(thing)
# thing0[ np.isnan(thing0) ] = -3.
# thing0[ mask_in ] = (ice_t[-1] - ice_t[0]) + 2

# gthing = nangauss(thing0, 3 )


# thing0 = np.copy(thing)
# thing0[ np.isnan(thing0) ] = -5.
# thing0[ mask_in ] = (ice_t[-1] - ice_t[0]) + 4

# gthing = nangauss(thing0, 7 )


thing0 = np.copy(thing)
thing0[ ~mask_out ] = -5.
thing0[ mask_in ] = (ice_t[-1] - ice_t[0]) + 3

gthing = nangauss(thing0, 9 )


mthing = np.copy(gthing)
mthing[~mask] = np.nan 

levels = ice_t-ice_t[0]
levels[0] += 0.1
x_conts, y_conts, th_conts = get_contours(xgr, ygr, gthing, levels)

t3 = time()

t3 - t2


#%%


fig, ax = plt.subplots(1,3, figsize=(16,6), sharey=True, layout='constrained')

im = ax[1].imshow( mthing , extent=(-30,30,-30,30), cmap='viridis' )
ax[1].plot( hol_xo, hol_yo, 'g-' )

cmap = im.cmap
norm = im.norm

for n in range(0, len(ice_xo), 2):    
    color = im.cmap(im.norm(ice_t[n]-ice_t[0]))
    ax[0].plot( ice_xo[n], ice_yo[n], '-', markersize=1, color=color )
ax[0].imshow( mthing , extent=(-30,30,-30,30), alpha=0 )


for n in range(0, len(ice_xo), 2):    
    color = im.cmap(im.norm(ice_t[n]-ice_t[0]))
    ax[2].plot( ice_xo[n], ice_yo[n], '-', markersize=1, color=color, alpha=0.3 )
    ax[2].plot( x_conts[n], y_conts[n], '--', markersize=1, color=color )
ax[2].imshow( mthing , extent=(-30,30,-30,30), alpha=0 )

for i in range(3):
    ax[i].set_xlabel(r'$x$ (mm)')
ax[0].set_ylabel(r'$y$ (mm)')

# fig.colorbar(im)

# plt.savefig('./Documents/Sphere figures/natural_prof.pdf',dpi=400, bbox_inches='tight')
plt.show()


x = np.linspace( -max_val, max_val, 2*max_val*ppmm+1  ,endpoint=True )
gy, gx = np.gradient( gthing, x, -x, edge_order=2 )
grad_abs = np.sqrt( gx**2 + gy**2 )

# kurv
mgx, mgy = np.gradient(gthing, x, -x, edge_order=2)
mgxx, mgxy = np.gradient(mgx, x, -x, edge_order=2)
mgyx, mgyy = np.gradient(mgy, x, -x, edge_order=2)

kurv = -( mgxx * mgy**2 + mgyy * mgx**2 - 2 * mgxy * mgx * mgy ) / np.sqrt( mgx**2 + mgy**2 )**3
# kurv = np.abs( mgxx * mgy**2 + mgyy * mgx**2 - 2 * mgxy * mgx * mgy ) / np.sqrt( mgx**2 + mgy**2 )**3


m_kurv = np.copy(kurv)
m_grad_abs = np.copy(grad_abs)
m_kurv[~mask] = np.nan 
m_grad_abs[~mask] = np.nan 

# fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=True, layout='constrained')

# # im1 = ax[1].imshow( 1/m_grad_abs , extent=(-30,30,-30,30), cmap='viridis') #, vmin=0, vmax=0.3 )
# im1 = ax[1].imshow( 1/m_grad_abs , extent=(-30,30,-30,30), cmap='viridis', vmin=0.08, vmax=0.5 )

# im0 = ax[0].imshow( m_kurv , extent=(-30,30,-30,30), cmap='viridis' )

# fig.colorbar(im1, ax=ax[1], shrink=0.8, pad=0.01)
# fig.colorbar(im0, ax=ax[0], shrink=0.8, pad=0.01)


# ax[0].set_title('Curvature')
# ax[1].set_title('Melt rate')

# fig.show()

#%%

# =============================================================================
# Figure example interpol
# =============================================================================

# In polar plot
# from matplotlib.image import NonUniformImage

plt.rcParams.update({'font.size':20})

fig, axs = plt.subplots(1, 3, figsize=(16, 7), layout='constrained')

mask_hol30 = hol_yo < 30

make_circle_plot(axs)

im = axs[1].imshow( mthing , extent=(-30,30,-30,30), cmap='viridis', vmin=0, )# vmax=100 )

cmap = im.cmap
norm = im.norm

axs[0].plot( hol_xo[mask_hol30], hol_yo[mask_hol30], 'r-' )

for n in range(0, len(ice_xo), 1):    
    color = im.cmap(im.norm(ice_t[n]-ice_t[0]))
    axs[0].plot( ice_xo[n], ice_yo[n], '-', markersize=1, color=color )
axs[0].imshow( mthing , extent=(-30,30,-30,30), alpha=0 )


for n in range(1, len(ice_xo), 3):    
    color = im.cmap(im.norm(ice_t[n]-ice_t[0]))
    axs[2].plot( ice_xo[n], ice_yo[n], '-', markersize=1, color=color, alpha=0.3 )
    axs[2].plot( x_conts[n], y_conts[n], '--', markersize=1, color=color )
axs[2].imshow( mthing , extent=(-30,30,-30,30), alpha=0 )


for i,lab in zip([0,1,2],[r'$a)$', r'$b)$', r'$c)$']):
    axs[i].text(-33,30, lab  )
    
ax_lims = [0.05,0.10,0.9,.02]
cbar_ax = fig.add_axes(ax_lims)
cbar = fig.colorbar(im, label=r'$t$ (s)', location='bottom', shrink=0.9, cax=cbar_ax , ticks=[0,20,40,60,80,100] )
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{int(x)}"))

# fig.tight_layout()
 
    
filename = './Documents/Sphere figures/interpolation.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

fig.show()

#%%

# =============================================================================
# Comparison contours
# =============================================================================

plt.rcParams.update({'font.size':14})

path = '/Volumes/Ice blocks/Sphere channel/Results/'
files = glob.glob( path + '*.hdf5' )

max_val = 30
ppmm = 10

x = np.linspace( -max_val, max_val, 2*max_val*ppmm+1  ,endpoint=True )

xgr, ygr = np.meshgrid( x,x[::-1] )
rgr, thegr = np.sqrt( xgr**2 + ygr**2 ), np.arctan2( ygr, xgr )

# opaque 10, opaque 40, clear 10, clear 40
files_use = [28, 30, 4, 6]
# files_use = [28]


ice_ts, gthings = [], [] 
hol_xos, hol_yos = [], []
ice_xos, ice_yos = [], []

for i,f in enumerate(files_use):

    t1 = time()
    
    cha_t, cha_s, cha_v, cha_Tb, cha_Tt, hol_x, hol_y, ice_t, ice_x, ice_y = get_data(files[f])
    
    t2 = time()
    print(t2-t1)
    
    t1 = time()
    dt = np.diff(ice_t)[0]
    ice_xo, ice_yo, cx, cy = order_data(ice_x, ice_y)

    ice_xo, ice_yo = ordenar_contornos(ice_xo, ice_yo)
    ice_xo, ice_yo = keep_inside_previous(ice_xo, ice_yo, lag=1, tol=.2) 

    ice_xos.append( ice_xo )
    ice_yos.append( ice_yo )
    
    # print( len(ice_t), len(ice_x), len(ice_y), len(ice_xo), len(ice_yo) )
    
    flat_t = np.array( [t for x, t in zip(ice_xo, ice_t) for _ in x] )
    flat_x = np.concatenate(ice_xo)
    flat_y = np.concatenate(ice_yo)
    
    points = np.vstack(  (flat_x, flat_y) ).T
    
    thing = griddata(points, flat_t-ice_t[0], (xgr,ygr), method='linear')
    
    mask, mask_out, mask_in, (x_mask,y_mask) = mask_ice_pos(xgr, ygr, ice_xo, ice_yo,  hol_c=True, min_y=2) 
    
    print( ice_t[-1] - ice_t[0], dt )
    
    hol_xo, hol_yo = order_holder(hol_x, hol_y)
    hol_xo, hol_yo = hol_xo-cx, hol_yo-cy
    
    thing0 = np.copy(thing)
    thing0[ ~mask_out ] = -5.
    thing0[ mask_in ] = (ice_t[-1] - ice_t[0]) + 3

    gthing = nangauss(thing0, 9 )

    # thing0[ np.isnan(thing0) ] = 0.
    # gthing = nangauss(thing0, 7 )
    
    t2 = time()
    print(t2-t1, end='\n\n')
    
    ice_ts.append( ice_t )
    gthings.append( gthing )
    hol_xos.append( hol_xo )
    hol_yos.append( hol_yo )
    

#%%

fig, axs = plt.subplots(2,2, figsize=(12, 10), layout='constrained')
axs= axs.flatten()

make_circle_plot(axs)
    
for i,f in enumerate(files_use):
    
    gthing = gthings[i]
    
    if i%2 == 0:
        time_cont = np.arange(0,210,10)
    else:
        time_cont = np.arange(0,100,5)
        
    levels = time_cont #ice_t-ice_t[0]
    x_conts, y_conts, th_conts = get_contours(xgr, ygr, gthing, levels)
    
    for j in range(len(levels)):
        mask = (x_conts[j] > -2.5) * (x_conts[j] < 2.5) * (y_conts[j] > 0.)
        x_conts[j][mask], y_conts[j][mask] = np.nan, np.nan

    norm = plt.Normalize()
    colors = plt.cm.viridis( norm(time_cont) )
    
    # for j,n in enumerate(range(1, len(ice_xo), 3)):    
    for j,n in enumerate(time_cont):    
        color = colors[j]
        # axs[i].plot( ice_xo[n], ice_yo[n], '-', markersize=1, color=color, alpha=0.3 )

        axs[i].plot( x_conts[j], y_conts[j], '-', markersize=1, color=color )

    # mthing = np.copy(gthing)
    # mthing[~mask] = np.nan 
    # im = axs[i].imshow( mthing , extent=(-30,30,-30,30), cmap='viridis', vmin=0, )

    axs[i].axis('equal')


axs[0].set_title(r"$U_\infty = 0.1$ m/s", pad=15) # fontsize=14,
axs[1].set_title(r"$U_\infty = 0.4$ m/s", pad=15) # fontsize=14,

axs[0].text(-0.2, 0.5, r"Opaque", va='center', ha='center', transform=axs[0].transAxes) # fontsize=14,
axs[2].text(-0.2, 0.5, r"Clear", va='center', ha='center', transform=axs[2].transAxes) # fontsize=14,
    
filename = './Documents/Sphere figures/clear_opaque_compa.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

fig.show()

#%%
V0, A0 = 4/3 * np.pi * 30**3, 4 * np.pi * 30**2

fig, axs = plt.subplots(1,2, figsize=(9, 5), layout='constrained')
axs = axs.flatten()

for i,f in enumerate(files_use):
# for i,f in enumerate(files_use[:2]):
    
    gthing, ice_t = gthings[i], ice_ts[i]
    ice_xo, ice_yo = ice_xos[i], ice_yos[i]
    hol_xo, hol_yo = hol_xos[i], hol_yos[i]


    if i%2 == 0:
        time_cont = np.arange(0,210,10)
    else:
        time_cont = np.arange(0,100,5)

    # levels = time_cont 
    levels = ice_t - ice_t[0]
    
    x_conts, y_conts, th_conts = get_contours(xgr, ygr, gthing, levels)
    
    vols, ars = [],[]
    for j in range(len(ice_xo)):
        va,aa = area_vol(ice_xo[j], ice_yo[j])
        vols.append( va ); ars.append( aa )
    vols, ars = np.array(vols) , np.array(ars) 
        
    # svo, sar, hov, har = calculate_areas_volumes_im2(  xgr, ygr, gthing, levels )
    svo, sar, hov, har = calculate_areas_volumes_im2( xgr, ygr, gthing, hol_xo, hol_yo, levels, order_grad=2 )
    
    # axs[0].plot( levels, vols-hov, '.-' )
    # axs[0].plot( levels, svo-hov, '.-' )
    axs[0].plot( levels, (svo-hov)/(svo-hov)[0], '.-' )
    
    # axs[1].plot( levels, ars-har, '.-' )
    # axs[1].plot( levels, sar-har, '.-' )
    axs[1].plot( levels, (sar-har)/(sar-har)[0], '.-' )
    
    # axs[0].plot( levels, vols / vols[0], '.-' )
    # axs[0].plot( levels, svo / svo[0] , '.-' )
    # axs[0].plot( levels, svo  , '.-' )
    
    # axs[1].plot( levels, ars / ars[0], '.-' )
    # axs[1].plot( levels, sar / sar[0], '.-' )
    # axs[1].plot( levels, sar , '.-' )

axs[0].grid()    
axs[1].grid()    

fig.show()
    
    

#%%

plt.figure()
plt.plot( ice_xo[0], ice_yo[0], '-' )
plt.plot( ice_xo[0][0], ice_yo[0][0], 'r.' )
plt.show()


#%%
# =============================================================================
# Nu vs Re (calcs of A and V)
# =============================================================================

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




plt.figure()

plt.imshow( thing )
# plt.plot( thing[:,400] )


plt.show()

thing0 = np.copy(thing)
thing0[ np.isnan(thing0) ] = 0.
gthing = nangauss(thing0, 7 )

gthing[~mask] = np.nan

plt.figure()

plt.imshow( gthing )
# plt.plot( gthing[:,400], '.-' )

plt.show()


thing0 = np.copy(thing)
gthing = nangauss(thing0, 7 )
gthing[~mask] = np.nan

plt.figure()
plt.imshow( gthing )
# plt.plot( gthing[:,400], '.-' )
plt.show()

thing0 = np.copy(thing)
thing0[~mask] = np.nan
gthing = nangauss_smooth_boundary(thing0, 7, transition_width=4 )

gthing[~mask] = np.nan

plt.figure()
plt.imshow( gthing )

# plt.plot( gthing[:,400], '.-' )

plt.show()


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

axs[0].set_title('Time (s)')
axs[1].set_title('Bins')
axs[2].set_title('Curvature (1/mm)')
axs[3].set_title('Mean Curvature (1/mm)')

for label in axs[3].get_yticklabels():
    label.set_horizontalalignment('center')

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




from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.signal import convolve2d, savgol_filter

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra, breadth_first_order, connected_components


def buscar_bordes_rap(tramo, k=8):
    """
    Find the two endpoints of an open curve given as an (N, 2) float array.

    Returns (x_list, y_list, ind), where ind are the row indices of the
    endpoints in `tramo`.
    """
    tramo = np.asarray(tramo, dtype=float)
    n = len(tramo)
    if n <= 2:
        return [tramo[0, 0]], [tramo[0, 1]], [0]

    # k-nearest-neighbour graph (edge weight = Euclidean distance)
    k = min(k, n - 1)
    dist, idx = cKDTree(tramo).query(tramo, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    w = np.maximum(dist[:, 1:].ravel(), 1e-12)   # zero weights would be dropped by scipy
    ok = rows != cols                             # guard against self-loops from duplicate points
    G = csr_matrix((w[ok], (rows[ok], cols[ok])), shape=(n, n))

    # the MST removes shortcuts, leaving a tree that follows the curve
    mst = minimum_spanning_tree(G)

    # "double sweep": farthest point from anywhere is one end,
    # farthest point from that end is the other end
    d0 = dijkstra(mst, directed=False, indices=0)
    d0[~np.isfinite(d0)] = -1
    a = int(np.argmax(d0))

    da = dijkstra(mst, directed=False, indices=a)
    da[~np.isfinite(da)] = -1
    b = int(np.argmax(da))

    ind = [a, b]
    return list(tramo[ind, 0]), list(tramo[ind, 1]), ind


def ordenar_tramo_rap(tramo, k=8):

    tramo = np.asarray(tramo, dtype=float)
    n = len(tramo)
    if n <= 2:
        return tramo.copy()

    # k-nearest-neighbour graph, increasing k until it is connected
    tree = cKDTree(tramo)
    k = min(k, n - 1)
    while True:
        dist, idx = tree.query(tramo, k=k + 1)
        rows = np.repeat(np.arange(n), k)
        cols = idx[:, 1:].ravel()
        w = np.maximum(dist[:, 1:].ravel(), 1e-12)   # scipy drops zero weights
        ok = rows != cols                             # ignore self-loops from duplicate points
        G = csr_matrix((w[ok], (rows[ok], cols[ok])), shape=(n, n))
        if connected_components(G, directed=False)[0] == 1 or k >= n - 1:
            break
        k = min(2 * k, n - 1)

    # the MST removes shortcuts and follows the curve
    mst = minimum_spanning_tree(G)

    # double sweep: find the two borders (ends of the longest path)
    d0 = dijkstra(mst, directed=False, indices=0)
    d0[~np.isfinite(d0)] = -1
    a = int(np.argmax(d0))                            # first border

    da = dijkstra(mst, directed=False, indices=a)     # distance along the curve from a
    da[~np.isfinite(da)] = np.inf                     # unreachable points (if any) go last
    order = np.argsort(da, kind="stable")             # starts at a, ends at the other border

    return tramo[order]

def ordernar_contorno( x, y ):
    tramo = np.vstack((x, y)).T
    tra_ord = ordenar_tramo_rap(tramo)
    return tra_ord 

def ordenar_contornos( ice_xo, ice_yo):
    x_ord, y_ord = [],[]
    for i in range(len(ice_xo)):
        # tramo = np.vstack((ice_xo[i], ice_yo[i])).T
        # tra_ord = ordenar_tramo_rap(tramo)
        tra_ord = ordernar_contorno( ice_xo[i], ice_yo[i] )

        x, y = tra_ord[:,0], tra_ord[:,1]
        ccx, ccy = np.mean(x), np.mean(y)
        angle = np.arctan2(x - ccx, -(y - ccy) )
        imin, imax = np.argmin(angle), np.argmax(angle)
        sidemin = imin < imax

        if sidemin:
            ice_ox, ice_oy = x[imin:imax], y[imin:imax]
        else:
            ice_ox, ice_oy = x[imin:imax:-1], y[imin:imax:-1]

        x_ord.append(ice_ox); y_ord.append(ice_oy)
    return x_ord, y_ord


def _dist_to_polygon(poly, pts, chunk=2000):
    """Distance from each point in pts (M, 2) to the closed polyline poly (N, 2)."""
    a = poly
    b = np.roll(poly, -1, axis=0)                  # closing segment included
    ab = b - a
    ab2 = np.einsum("ij,ij->i", ab, ab)
    ab2[ab2 == 0] = 1.0                            # degenerate (repeated) vertices

    out = np.empty(len(pts))
    for s in range(0, len(pts), chunk):            # chunked to limit memory
        p = pts[s:s + chunk]
        ap = p[:, None, :] - a[None, :, :]
        t = np.clip(np.einsum("cmj,mj->cm", ap, ab) / ab2, 0.0, 1.0)
        proj = a[None, :, :] + t[..., None] * ab[None, :, :]
        d = np.linalg.norm(p[:, None, :] - proj, axis=2)
        out[s:s + chunk] = d.min(axis=1)
    return out


def inside_mask(outer_x, outer_y, px, py, tol=0.0):
    """
    Boolean mask of points (px, py) that are inside the polygon (outer_x, outer_y)
    or outside it by no more than `tol` (same units as the coordinates).
    """
    poly = np.column_stack((np.asarray(outer_x, float), np.asarray(outer_y, float)))
    pts = np.column_stack((np.asarray(px, float), np.asarray(py, float)))
    if len(poly) < 3 or len(pts) == 0:
        return np.zeros(len(pts), dtype=bool)

    mask = Path(poly).contains_points(pts)
    if tol > 0:
        out_idx = np.where(~mask)[0]
        if out_idx.size:
            d = _dist_to_polygon(poly, pts[out_idx])
            mask[out_idx] = d <= tol
    return mask


def keep_inside_previous(ice_x, ice_y, lag=1, tol=0.0, use_trimmed=True, clip_early=False):
    """
    Keep contour i inside contour i - lag, allowing points up to `tol` outside it.

    Parameters
    ----------
    ice_x, ice_y : list of 1D float arrays (one entry per contour, outermost first)
    lag : int >= 1
        How many contours back to compare against.
    tol : float >= 0
        Points outside the reference contour but within this distance of it
        are kept. tol=0 gives the strict behaviour.
    use_trimmed : bool
        True  -> compare against the already-trimmed contour i-lag.
        False -> compare against the original contour i-lag.
    clip_early : bool
        False -> the first `lag` contours are left unchanged.
        True  -> they are compared against contour 0 instead (for i >= 1).
    """
    if lag < 1 or int(lag) != lag:
        raise ValueError("lag must be an integer >= 1.")
    if tol < 0:
        raise ValueError("tol must be >= 0.")
    lag = int(lag)

    n = len(ice_x)
    if n != len(ice_y):
        raise ValueError("ice_x and ice_y must have the same number of contours.")

    orig_x = [np.asarray(a, dtype=float) for a in ice_x]
    orig_y = [np.asarray(a, dtype=float) for a in ice_y]
    out_x, out_y = [], []

    for i in range(n):
        ref = i - lag
        if ref < 0:
            ref = 0 if (clip_early and i > 0) else None
        if ref is None:
            out_x.append(orig_x[i])
            out_y.append(orig_y[i])
            continue

        ref_x = out_x[ref] if use_trimmed else orig_x[ref]
        ref_y = out_y[ref] if use_trimmed else orig_y[ref]

        mask = inside_mask(ref_x, ref_y, orig_x[i], orig_y[i], tol=tol)
        out_x.append(orig_x[i][mask])
        out_y.append(orig_y[i][mask])

    return out_x, out_y

#%%
# =============================================================================
# Tries
# =============================================================================

path = '/Volumes/Ice blocks/Sphere channel/Results/'
files = glob.glob( path + '*.hdf5' )


t1 = time()

# cha_t, cha_s, cha_v, cha_Tb, cha_Tt, hol_x, hol_y, ice_t, ice_x, ice_y = get_data(files[1])
cha_t, cha_s, cha_v, cha_Tb, cha_Tt, hol_x, hol_y, ice_t, ice_x, ice_y = get_data(files[4])
# cha_t, cha_s, cha_v, cha_Tb, cha_Tt, hol_x, hol_y, ice_t, ice_x, ice_y = get_data(files[7])
# cha_t, cha_s, cha_v, cha_Tb, cha_Tt, hol_x, hol_y, ice_t, ice_x, ice_y = get_data(files[-1])

t2 = time()
print(t2-t1)
#%%


fourier_smooth = False

dt = np.diff(ice_t)[0]

t1 = time()

ice_xo, ice_yo, cx, cy = order_data(ice_x, ice_y)
ice_xo, ice_yo = ordenar_contornos(ice_xo, ice_yo)
# ice_xs, ice_ys, cx, cy = order_data(ice_x, ice_y, fourier_smooth=True , modes=15)

ice_xc, ice_yc = keep_inside_previous(ice_xo, ice_yo, lag=1, tol=.2) 

# ice_xs, ice_ys = keep_min_radius( ice_x, ice_y, np.deg2rad(0.01) )

t2 = time()
print(t2-t1)

#%%

fig, ax = plt.subplots(1,2,figsize=(10,5), layout='constrained')

# for n in range(2,len(ice_xo), 3):
for n in range(len(ice_xo)-10, len(ice_xo), 1):

# for n in [1,2,  56, 97]:
        
    ax[0].plot( ice_xo[n], ice_yo[n], '.', markersize=1, color=(n/len(ice_xo),0,1-n/len(ice_xo)) )
        
    ax[1].plot( ice_xc[n], ice_yc[n], '-', markersize=1, color=(n/len(ice_xo),0,1-n/len(ice_xo)) )
    # ax[1].plot( ice_x[n], ice_y[n], '.', markersize=1, color=(n/len(ice_xo),0,1-n/len(ice_xo)) )
    

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

# flat_t = np.array( [t for x, t in zip(ice_xo, ice_t) for _ in x] )
# flat_x = np.concatenate(ice_xo)
# flat_y = np.concatenate(ice_yo)
# points = np.vstack(  (flat_x, flat_y) ).T

# thing1 = griddata(points, flat_t-ice_t[0], (xgr,ygr), method='linear')

flat_t = np.array( [t for x, t in zip(ice_xc, ice_t) for _ in x] )
flat_x = np.concatenate(ice_xc)
flat_y = np.concatenate(ice_yc)
points = np.vstack(  (flat_x, flat_y) ).T

thing2 = griddata(points, flat_t-ice_t[0], (xgr,ygr), method='linear')



# thing2 = interpolate_gaussian(flat_x, flat_y, flat_t-ice_t[0], xgr, ygr, dist_threshold=10, sigmas=[0.6,0.6] )
# thing2 = interpolate_idw(flat_x, flat_y, flat_t-ice_t[0], xgr, ygr, dist_threshold=100, power=3)

mask, mask_in, (x_mask,y_mask) = mask_ice_pos(xgr, ygr, ice_xc, ice_yc,  hol_c=True) 
# mask = ( (~np.isnan(thing2))*1. - ((xgr > -2.5) * (xgr < 2.5) * (ygr>0)) )>0

hol_xo, hol_yo = order_holder(hol_x, hol_y)
hol_xo, hol_yo = hol_xo-cx, hol_yo-cy

t2 = time()
print(t2-t1)

#%%
t2 = time()

# thing0 = np.copy(thing1)
# thing0[ np.isnan(thing0) ] = 0.

# gthing1 = nangauss(thing0, 7 )
# # gthing1[~mask] = np.nan


thing0 = np.copy(thing2)
thing0[ np.isnan(thing0) ] = -5.
thing0[ mask_in ] = (ice_t[-1] - ice_t[0]) + 4

gthing1 = nangauss(thing0, 7 )

thing0 = np.copy(thing2)
thing0[ np.isnan(thing0) ] = -3.
thing0[ mask_in ] = (ice_t[-1] - ice_t[0]) + 2

gthing2 = nangauss(thing0, 3 )


levels = ice_t-ice_t[0]
# # levels[0] += 0.1
# time_cont = np.arange(5,100,5)        
# levels = time_cont #ice_t-ice_t[0]


x2_conts, y2_conts, th_conts = get_contours(xgr, ygr, gthing2, levels)
x1_conts, y1_conts, th_conts = get_contours(xgr, ygr, gthing1, levels)

gthing2[~mask] = np.nan
# gthing1[~mask] = np.nan

t3 = time()

t3 - t2

#%%
extent = (-30,30,-30,30)

# fig, axs = plt.subplots(1,2, layout='constrained', figsize=(8,4))
 
# axs[0].imshow( thing2, extent=extent )
# axs[1].imshow( thing0, extent=extent )
# # axs[1].imshow( thing1 - thing2, extent=extent )

# fig.show()


# fig, axs = plt.subplots(1,2, layout='constrained', figsize=(8,4))

# axs[0].imshow( gthing1, extent=extent )
# axs[1].imshow( gthing2, extent=extent )
# # axs[1].imshow( gthing1-gthing2, extent=extent )

# fig.show()


fig, axs = plt.subplots(1,2, layout='constrained', figsize=(8,4))

# for i in range(0, len(levels), 5):
for i in [0, 10, len(levels)-4, len(levels)-2, len(levels)-1]:
# for i in [0, 10, len(levels)-1]:

    axs[0].plot( x1_conts[i], y1_conts[i], '-', color=(i/len(levels),0,1-i/len(levels))  )    
    axs[0].plot( x2_conts[i], y2_conts[i], '--', color=(i/len(levels),0,1-i/len(levels))  )

    axs[1].plot( x1_conts[i], y1_conts[i], '-', color=(i/len(levels),0,1-i/len(levels))  )
    # axs[1].plot( x2_conts[i], y2_conts[i], '-', color=(i/len(levels),0,1-i/len(levels))  )
    axs[1].plot( ice_xc[i], ice_yc[i], '--', color=(i/len(levels),0,1-i/len(levels)) , alpha=0.5 )
    

axs[0].axis('equal')
axs[1].axis('equal')
axs[0].grid()
axs[1].grid()


fig.show()

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
    