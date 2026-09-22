#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:52:45 2026

@author: tomas
"""

import numpy as np
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt    

from scipy.integrate import cumulative_trapezoid, solve_bvp
from scipy.sparse import lil_matrix
from scipy.linalg import solve_banded
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

def solve_ode_ini(eta, Pr, Lambda, K, funcs0, eta_max=None,
              tol=1e-7, max_nodes=50000, sol_guess=None):
    """
    Solve the coupled nonlinear BVP

        f''' + f*f'' * H(0) - f'^2 + V(0) + Lambda* F(0) *(Theta - 1) = 0
        Theta'' + H(0) Pr*f*Theta' = 0

    Boundary conditions:
        f'(0)                       = 0
        f'(infty) - I(0)            = 0
        Theta(0)                    = 0
        Theta(infty) - 1            = 0
        H(0) * f(0) - K * Theta'(0) = 0

    Parameters
    ----------
    eta : array_like
        Points at which the solution is requested.
    Pr : float
        Prandtl number.
    Lambda : float
        Buoyancy/mixed-convection parameter.
    K : float
        Constant from the Stefan condition. Use K < 0.15.
    funcs0: [H(0), V(0), F(0), I(0)].
        list of floats with the values of those functions
    eta_max : float, optional
        Truncation of infinity. If None, uses max(eta) if sufficiently
        large, otherwise 20.
    tol : float, optional
        solve_bvp tolerance.
    max_nodes : int, optional
        Maximum number of mesh nodes.

    Returns
    -------
    f : ndarray
        Streamfunction similarity solution.
    fp : ndarray
        First derivative f'.
    theta : ndarray
        Temperature similarity solution.
    """

    H0, V0, F0, I0 = funcs0
    eta = np.asarray(eta, dtype=float)

    if np.any(eta < 0):
        raise ValueError("eta must be non-negative.")

    if Pr <= 0:
        raise ValueError("Pr must be positive.")

    # ---------------------------------------------------------
    # Choose computational domain
    # ---------------------------------------------------------

    # Low Pr requires a larger domain.
    if eta_max is None:
        eta_max = max( 20.0, 10.0 / np.sqrt(Pr) )

    # Make sure requested points lie inside the domain
    eta_max = max(eta_max, np.max(eta))
    x = np.linspace(0.0, eta_max, 400)

    # ---------------------------------------------------------
    # Initial guess
    # ---------------------------------------------------------

    if sol_guess is None:
        fp = 1.0 - np.exp(-x)
        f = x - 1.0 + np.exp(-x)
    
        # Thermal boundary layer gets thicker as Pr decreases
        thermal_scale = max(1.0, 1.0 / np.sqrt(Pr))
        theta = 1.0 - np.exp(-x / thermal_scale)
    
        fpp = np.gradient(fp, x)
        thetap = np.gradient(theta, x)
    
        y_guess = np.vstack([ f, fp, fpp, theta, thetap ])
        
    else:
        x = sol_guess.x
        y_guess = sol_guess.y        

    # ---------------------------------------------------------
    # ODE system
    # ---------------------------------------------------------

    def ode(x, y):

        f, fp, fpp = y[0], y[1], y[2]
        theta, thetap = y[3], y[4]

        # Momentum:
        # f''' + f*f'' * H(0) - f'^2 + V(0) + Lambda * F(0) *(1 - Theta) = 0

        fppp = ( -f * fpp * H0 + fp**2 - V0 - Lambda * F0 * (theta - 1.0) )

        # Energy:
        # Theta'' + H(0) Pr*f*Theta' = 0

        thetapp = -H0 * Pr * f * thetap

        return np.vstack([ fp, fpp, fppp, thetap, thetapp ])

    # ---------------------------------------------------------
    # Boundary conditions
    # ---------------------------------------------------------

    # f'(0) = 0
    # f'(infty) = I(0)
    # Theta(0) = 0
    # Theta(infty) = 1
    # H(0) * f(0) = K * Theta'(0) 

    def bc(ya, yb):
        return np.array([ ya[1], yb[1] - I0, ya[3], yb[3] - 1.0, H0 * ya[0] - K * ya[4] ])

    # ---------------------------------------------------------
    # Solve
    # ---------------------------------------------------------

    sol = solve_bvp( ode, bc, x, y_guess, tol=tol, max_nodes=max_nodes )

    if not sol.success:
        raise RuntimeError( f"BVP solver failed: {sol.message}" )

    # ---------------------------------------------------------
    # Evaluate at requested eta
    # ---------------------------------------------------------

    y = sol.sol(eta)

    f, fp, fpp = y[0], y[1], y[2]
    theta, thetap = y[3], y[4]
    
    dy = sol.sol.derivative(1)(eta)
    fppp, thpp = dy[2], dy[4]
    
    eq1 = fppp + H0*f*fpp - fp**2 + V0 + Lambda*F0*(theta - 1)
    eq2 = thpp + H0*Pr*f*thetap

    return sol, (eq1,eq2) #, (fppp, thpp)


def build_Y0_from_ini(eta,sol):
    """
    Get Y from sol of solve_ode_ini
    """
    y = sol.sol(eta)
    f, p, q, th, r = y[0], y[1], y[2], y[3], y[4]

    Y0 = np.column_stack([f, p, q, th, r])
    return Y0


def H(x):
    return 1 + np.cos(x) / np.sinc(x / np.pi)
def V(x):
    return 9/4 * np.sinc(x / np.pi) * np.cos(x)
def F(x):
    return np.sinc(x / np.pi) 
def I(x):
    return 3/2 * np.sinc(x / np.pi)

def sparse_to_banded(M):
    """
    Convert a sparse (or dense) matrix M into the banded storage format
    expected by scipy.linalg.solve_banded, auto-detecting the bandwidth.
    """
    if issparse(M):
        Mc = M.tocoo()
        rows, cols = Mc.row, Mc.col
    else:
        rows, cols = np.nonzero(M)

    lu = int(np.max(cols - rows)) if len(rows) else 0   # superdiagonal count
    ld = int(np.max(rows - cols)) if len(rows) else 0   # subdiagonal count

    n = M.shape[0]
    ab = np.zeros((lu + ld + 1, n))
    for offset in range(-ld, lu + 1):
        diag = M.diagonal(offset)
        row = lu - offset
        if offset >= 0:
            ab[row, offset:offset + len(diag)] = diag
        else:
            ab[row, 0:len(diag)] = diag

    return ab, lu, ld


def compute_all_Aj_Bj_Rj_vectorized(Y, Y_prev, xm, dx, dy, Lambda, Pr, Hm, Fm, Vm):
    """
    Vectorized version of compute_Aj_Bj_Rj: computes Aj, Bj, Rj for
    ALL interior intervals (j-1, j) at once, j = 1..N (i.e. pairs
    (0,1), (1,2), ..., (N-1,N)), using array ops instead of a
    Python loop.

    Returns
    -------
    As, Bs : ndarray, shape (N, 5, 5)
    Rs : ndarray, shape (N, 5)
    """
    fj, pj, qj, thj, rj = Y[:-1].T          # each shape (N,)
    fjp, pjp, qjp, thjp, rjp = Y[1:].T
    fjn, pjn, qjn, thjn, rjn = Y_prev[:-1].T
    fjpn, pjpn, qjpn, thjpn, rjpn = Y_prev[1:].T

    N = fj.shape[0]
    As = np.zeros((N, 5, 5))
    Bs = np.zeros((N, 5, 5))
    Rs = np.zeros((N, 5))

    # Row 1
    As[:, 0, 0] = -1
    As[:, 0, 1] = -dy / 2
    Bs[:, 0, 0] = 1
    Bs[:, 0, 1] = -dy / 2

    # Row 2
    As[:, 1, 1] = -1
    As[:, 1, 2] = -dy / 2
    Bs[:, 1, 1] = 1
    Bs[:, 1, 2] = -dy / 2

    # Row 3
    qsum = qj + qjn + qjp + qjpn
    As[:, 2, 0] = dy * (qsum * xm / (8 * dx) + qsum * Hm / 16)
    Bs[:, 2, 0] = As[:, 2, 0]

    As[:, 2, 1] = dy * (
        (1 / 8) * (-pj - pjn - pjp - pjpn)
        - ((pj - pjn + pjp - pjpn) * xm) / (8 * dx)
        - ((pj + pjn + pjp + pjpn) * xm) / (8 * dx)
    )
    Bs[:, 2, 1] = As[:, 2, 1]

    fdiff = fj - fjn + fjp - fjpn
    fsum = fj + fjn + fjp + fjpn
    common22 = dy * (fdiff * xm / (8 * dx) + fsum * Hm / 16)
    As[:, 2, 2] = -1 + common22
    Bs[:, 2, 2] = 1 + common22

    As[:, 2, 3] = (1 / 4) * Lambda * dy * Fm
    Bs[:, 2, 3] = As[:, 2, 3]

    # Row 4
    As[:, 3, 3] = -1
    As[:, 3, 4] = -dy / 2
    Bs[:, 3, 3] = 1
    Bs[:, 3, 4] = -dy / 2

    # Row 5
    rsum = rj + rjn + rjp + rjpn
    As[:, 4, 0] = dy * (rsum * xm / (8 * dx) + rsum * Hm / 16)
    Bs[:, 4, 0] = As[:, 4, 0]

    thdiff = thj - thjn + thjp - thjpn
    As[:, 4, 1] = -(thdiff * xm * dy) / (8 * dx)
    Bs[:, 4, 1] = As[:, 4, 1]

    psum = pj + pjn + pjp + pjpn
    As[:, 4, 3] = -(psum * xm * dy) / (8 * dx)
    Bs[:, 4, 3] = As[:, 4, 3]

    As[:, 4, 4] = -(1 / Pr) + common22
    Bs[:, 4, 4] = 1 / Pr + common22

    # R
    Rs[:, 0] = fj - fjp + 0.5 * (pj + pjp) * dy
    Rs[:, 1] = pj - pjp + 0.5 * (qj + qjp) * dy
    Rs[:, 2] = qj - qjp - dy * (
        -(1 / 16) * psum ** 2
        - ((pj - pjn + pjp - pjpn) * psum * xm) / (8 * dx)
        + fdiff * qsum * xm / (8 * dx)
        + Lambda * (-1 + 0.25 * (thj + thjn + thjp + thjpn)) * Fm
        + (1 / 16) * fsum * qsum * Hm
        + Vm
    )
    Rs[:, 3] = thj - thjp + 0.5 * (rj + rjp) * dy
    Rs[:, 4] = -((-rj + rjp) / Pr) - dy * (
        fdiff * rsum * xm / (8 * dx)
        - psum * thdiff * xm / (8 * dx)
        + (1 / 16) * fsum * rsum * Hm
    )

    return As, Bs, Rs


def build_wall_block(Y, Y_prev, xm, dx, dy, Hm, Fm, Vm, K, Pr, Lambda):
    """
    Build the reduced wall-boundary blocks A0 (5x2, coefficients of
    d(q0,r0)), B0 (5x5, coefficients of full Y1), and R0 (5,), for the
    formulation where p0=0, th0=0 are enforced directly and f0 is
    replaced by f0_sub(r0) via the Stefan condition.

    Returns
    -------
    A0 : ndarray, shape (5, 2)
    B0 : ndarray, shape (5, 5)
    R0 : ndarray, shape (5,)
    """
    # current Newton iterate at points 0, 1 (p0, th0 enforced == 0)
    qj, rj = Y[0, 2], Y[0, 4]
    fjp, pjp, qjp, thjp, rjp = Y[1]

    # previous x-station values at points 0, 1
    fjn, pjn, qjn, thjn, rjn = Y_prev[0]
    fjpn, pjpn, qjpn, thjpn, rjpn = Y_prev[1]

    denom = xm / dx - Hm / 2
    fj_sub = (-(K / 2) * (rj + rjn) + fjn * (xm / dx + Hm / 2)) / denom

    term1 = (xm * (-fjn + fjp - fjpn + fj_sub)) / (8 * dx)
    term2 = (1 / 16) * (fjn + fjp + fjpn + fj_sub) * Hm

    # --- A0 (5x2): columns = (dq0, dr0) ---
    A0 = np.zeros((5, 2))
    A0[0, 1] = K / (2 * denom)
    A0[1, 0] = -dy / 2
    
    A0[2, 0] = -1 + dy * (term1 + term2)
    A0[2, 1] = dy * ( - (K * (qj + qjn + qjp + qjpn) * xm) / (16 * dx * denom) 
                      - (K * (qj + qjn + qjp + qjpn) * Hm) / (32 * denom) )
    
    A0[3, 1] = -dy / 2

    A0[4, 1] = -(1 / Pr) + dy * ( term1 
                                 - (K * (rj + rjn + rjp + rjpn) * xm) / (16 * dx * denom) 
                                 + term2 
                                 - (K * (rj + rjn + rjp + rjpn) * Hm) / (32 * denom) )

    # --- B0 (5x5): columns = (df1, dp1, dq1, dth1, dr1) ---
    B0 = np.zeros((5, 5))

    B0[0, 0] = 1
    B0[0, 1] = -dy / 2

    B0[1, 1] = 1
    B0[1, 2] = -dy / 2

    B0[2, 0] = dy * ( ((qj + qjn + qjp + qjpn) * xm) / (8 * dx) 
                     + (1 / 16) * (qj + qjn + qjp + qjpn) * Hm )
    B0[2, 1] = dy * ( (1 / 8) * (-pjn - pjp - pjpn) 
                     - ((-pjn + pjp - pjpn) * xm) / (8 * dx) 
                     - ((pjn + pjp + pjpn) * xm) / (8 * dx) )
    B0[2, 2] = 1 + dy * (term1 + term2)
    B0[2, 3] = (1 / 4) * Lambda * dy * Fm

    B0[3, 3] = 1
    B0[3, 4] = -dy / 2

    B0[4, 0] = dy * ( ((rj + rjn + rjp + rjpn) * xm) / (8 * dx) 
                     + (1 / 16) * (rj + rjn + rjp + rjpn) * Hm )
    B0[4, 1] = -( ((-thjn + thjp - thjpn) * xm * dy) / (8 * dx) )
    B0[4, 3] = -(((pjn + pjp + pjpn) * xm * dy) / (8 * dx))
    B0[4, 4] = 1 / Pr + dy * (term1 + term2)

    # --- R0 (5,) ---
    R0 = np.zeros(5)
    
    R0[0] = -fjp + (pjp * dy) / 2 + fj_sub
    R0[1] = -pjp + 0.5 * (qj + qjp) * dy
    R0[2] = qj - qjp - dy * ( -(1 / 16) * (pjn + pjp + pjpn)**2
                              - ((-pjn + pjp - pjpn) * (pjn + pjp + pjpn) * xm) / (8 * dx)
                              + Lambda * (-1 + 0.25 * (thjn + thjp + thjpn)) * Fm
                              + ((qj + qjn + qjp + qjpn) * xm * (-fjn + fjp - fjpn + fj_sub)) / (8 * dx)
                              + (1 / 16) * (qj + qjn + qjp + qjpn) * (fjn + fjp + fjpn + fj_sub) * Hm
                              + Vm )
    R0[3] = -thjp + 0.5 * (rj + rjp) * dy
    R0[4] = -((-rj + rjp) / Pr) - dy * ( -((pjn + pjp + pjpn) * (-thjn + thjp - thjpn) * xm) / (8 * dx)
                                         + ((rj + rjn + rjp + rjpn) * xm * (-fjn + fjp - fjpn + fj_sub)) / (8 * dx)
                                         + (1 / 16) * (rj + rjn + rjp + rjpn) * (fjn + fjp + fjpn + fj_sub) * Hm )

    return A0, B0, R0


def build_far_field_block(Y, Y_prev, xm, dx, dy, Hm, Fm, Vm, I_val, Pr, Lambda):
    """
    Build the reduced far-field blocks AN (5x5, coefficients of the
    full state at point N-1), BN (5x3, coefficients of the reduced
    unknowns (f_N, q_N, r_N)), and RN (5,), for the formulation where
    p_N = I(x_{n+1}) and th_N = 1 are enforced directly.

    Returns
    -------
    AN : ndarray, shape (5, 5)
    BN : ndarray, shape (5, 3)
    RN : ndarray, shape (5,)
    """
    N = Y.shape[0] - 1

    fj, pj, qj, thj, rj = Y[N - 1]
    fjp, _, qjp, _, rjp = Y[N]  # pjp, thjp fixed by BC, not needed here

    fjn, pjn, qjn, thjn, rjn = Y_prev[N - 1]
    fjpn, pjpn, qjpn, thjpn, rjpn = Y_prev[N]

    Ifun = I_val

    # --- AN (5x5): columns = (dfj, dpj, dqj, dthj, drj) at point N-1 ---
    AN = np.zeros((5, 5))
    AN[0, 0] = -1
    AN[0, 1] = -dy / 2

    AN[1, 1] = -1
    AN[1, 2] = -dy / 2

    AN[2, 0] = dy * ( ((qj + qjn + qjp + qjpn) * xm) / (8 * dx) 
                     + (1 / 16) * (qj + qjn + qjp + qjpn) * Hm )
    AN[2, 1] = dy * ( (1 / 8) * (-pj - pjn - pjpn - Ifun) 
                     - (xm * (pj - pjn - pjpn + Ifun)) / (8 * dx) 
                     - (xm * (pj + pjn + pjpn + Ifun)) / (8 * dx) )
    AN[2, 2] = -1 + dy * ( ((fj - fjn + fjp - fjpn) * xm) / (8 * dx) 
                          + (1 / 16) * (fj + fjn + fjp + fjpn) * Hm )
    AN[2, 3] = (1 / 4) * Lambda * dy * Fm
    AN[2, 4] = 0

    AN[3, 3] = -1
    AN[3, 4] = -dy / 2

    AN[4, 0] = dy * ( ((rj + rjn + rjp + rjpn) * xm) / (8 * dx) 
                     + (1 / 16) * (rj + rjn + rjp + rjpn) * Hm )
    AN[4, 1] = -( ((1 + thj - thjn - thjpn) * xm * dy) / (8 * dx) )
    AN[4, 2] = 0
    AN[4, 3] = -((xm * dy * (pj + pjn + pjpn + Ifun)) / (8 * dx))
    AN[4, 4] = -(1 / Pr) + dy * ( ((fj - fjn + fjp - fjpn) * xm) / (8 * dx) 
                                 + (1 / 16) * (fj + fjn + fjp + fjpn) * Hm )

    # --- BN (5x3): columns = (dfjp, dqjp, drjp) at point N ---
    BN = np.zeros((5, 3))
    BN[0, 0] = 1

    BN[1, 1] = -dy / 2

    BN[2, 0] = dy * ( ((qj + qjn + qjp + qjpn) * xm) / (8 * dx) 
                     + (1 / 16) * (qj + qjn + qjp + qjpn) * Hm )
    BN[2, 1] = 1 + dy * ( ((fj - fjn + fjp - fjpn) * xm) / (8 * dx) 
                         + (1 / 16) * (fj + fjn + fjp + fjpn) * Hm )

    BN[3, 2] = -dy / 2

    BN[4, 0] = dy * ( ((rj + rjn + rjp + rjpn) * xm) / (8 * dx) 
                     + (1 / 16) * (rj + rjn + rjp + rjpn) * Hm )
    BN[4, 2] = 1 / Pr + dy * ( ((fj - fjn + fjp - fjpn) * xm) / (8 * dx) 
                              + (1 / 16) * (fj + fjn + fjp + fjpn) * Hm )

    # --- RN (5,) ---
    RN = np.zeros(5)
    RN[0] = fj - fjp + 0.5 * dy * (pj + Ifun)
    RN[1] = pj + 0.5 * (qj + qjp) * dy - Ifun
    RN[2] = qj - qjp - dy * ( ((fj - fjn + fjp - fjpn) * (qj + qjn + qjp + qjpn) * xm) / (8 * dx)
                              + Lambda * (-1 + 0.25 * (1 + thj + thjn + thjpn)) * Fm
                              + (1 / 16) * (fj + fjn + fjp + fjpn) * (qj + qjn + qjp + qjpn) * Hm
                              - (xm * (pj - pjn - pjpn + Ifun) * (pj + pjn + pjpn + Ifun)) / (8 * dx)
                              - (1 / 16) * (pj + pjn + pjpn + Ifun)**2
                              + Vm )
    RN[3] = -1 + thj + 0.5 * (rj + rjp) * dy
    RN[4] = -((-rj + rjp) / Pr) - dy * ( ((fj - fjn + fjp - fjpn) * (rj + rjn + rjp + rjpn) * xm) / (8 * dx)
                                         + (1 / 16) * (fj + fjn + fjp + fjpn) * (rj + rjn + rjp + rjpn) * Hm
                                         - ((1 + thj - thjn - thjpn) * xm * (pj + pjn + pjpn + Ifun)) / (8 * dx) )

    return AN, BN, RN


def _blocks_to_coo(blocks, row_starts, col_starts):
    """blocks: (K, nr, nc) stacked dense blocks -> flat (rows, cols, data) for COO."""
    K, nr, nc = blocks.shape
    row_idx = row_starts[:, None, None] + np.arange(nr)[None, :, None]
    col_idx = col_starts[:, None, None] + np.arange(nc)[None, None, :]
    row_idx = np.broadcast_to(row_idx, (K, nr, nc))
    col_idx = np.broadcast_to(col_idx, (K, nr, nc))
    return row_idx.ravel(), col_idx.ravel(), blocks.ravel()


def assemble_block_tridiagonal_vectorized(Y, Y_prev, xm, dx, dy, Lambda, Pr, K, Hm, Fm, Vm, I_val, sparse=True):
    """
    Same as assemble_block_tridiagonal, but interior blocks are computed
    and assembled without a Python loop.
    """
    N = Y.shape[0] - 1
    n_eq = 5 * N

    def col_offset(m):
        return 2 + 5 * (m - 1)

    A0, B0, R0 = build_wall_block(Y, Y_prev, xm, dx, dy, Hm, Fm, Vm, K, Pr, Lambda)
    AN, BN, RN = build_far_field_block(Y, Y_prev, xm, dx, dy, Hm, Fm, Vm, I_val, Pr, Lambda)

    # all pairs (Y[k], Y[k+1]), k = 0..N-1  -> keep only k = 1..N-2 (interior)
    As, Bs, Rs = compute_all_Aj_Bj_Rj_vectorized(Y, Y_prev, xm, dx, dy, Lambda, Pr, Hm, Fm, Vm)
    As_int = As[1:N - 1]
    Bs_int = Bs[1:N - 1]
    Rs_int = Rs[1:N - 1]

    n_int = N - 2
    row_starts = 5 + 5 * np.arange(n_int)
    colA_starts = col_offset(1) + 5 * np.arange(n_int)      # point j-1, j=2..N-1
    colB_starts = col_offset(2) + 5 * np.arange(n_int)      # point j

    rows, cols, data = [], [], []

    # wall
    r0, c0 = np.meshgrid(np.arange(5), np.arange(2), indexing='ij')
    rows.append(r0.ravel()); cols.append(c0.ravel()); data.append(A0.ravel())
    r0, c0 = np.meshgrid(np.arange(5), col_offset(1) + np.arange(5), indexing='ij')
    rows.append(r0.ravel()); cols.append(c0.ravel()); data.append(B0.ravel())

    # interior
    r, c, d = _blocks_to_coo(As_int, row_starts, colA_starts)
    rows.append(r); cols.append(c); data.append(d)
    r, c, d = _blocks_to_coo(Bs_int, row_starts, colB_starts)
    rows.append(r); cols.append(c); data.append(d)

    # far field
    row_far = 5 * (N - 1)
    r0, c0 = np.meshgrid(row_far + np.arange(5), col_offset(N - 1) + np.arange(5), indexing='ij')
    rows.append(r0.ravel()); cols.append(c0.ravel()); data.append(AN.ravel())
    r0, c0 = np.meshgrid(row_far + np.arange(5), n_eq - 3 + np.arange(3), indexing='ij')
    rows.append(r0.ravel()); cols.append(c0.ravel()); data.append(BN.ravel())

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    M = coo_matrix((data, (rows, cols)), shape=(n_eq, n_eq)).tocsr()

    R = np.empty(n_eq)
    R[0:5] = R0
    R[5:5 + 5 * n_int] = Rs_int.ravel()
    R[row_far:row_far + 5] = RN

    if not sparse:
        M = M.toarray()

    return M, R


def newton_step(Y_new, Y_old, H, V, F, I, xmid, dx, dy, Pr, Lambda, K, max_iter=20, tol=1e-8):
    """
    Perform the full Newton iteration marching the similarity solution
    from x_n (Y_old, converged) to x_{n+1}, returning the converged
    Y_new. Uses the reduced block-tridiagonal Keller-box system with
    wall (p0=0, th0=0, f0=f0_sub(r0)) and far-field (p_N=I, th_N=1)
    boundary conditions enforced directly rather than as free unknowns.

    Parameters
    ----------
    Y_new : ndarray, shape (N+1, 5)
        Initial guess for Y at x_{n+1} (e.g. Y_old, or an extrapolation).
    Y_old : ndarray, shape (N+1, 5)
        Converged solution at x_n.
    H, V, F, I : callable
        Functions of x.
    xmid : float
        x_n + dx/2 (the Keller-box midpoint for this step).
    dx, dy : float
        CapitalDelta x and CapitalDelta y.
    Pr, Lambda, K : float
        Physical parameters.
    max_iter : int, optional
        Maximum Newton iterations.
    tol : float, optional
        Convergence tolerance on the max-norm of the correction dY.

    Returns
    -------
    Y_new : ndarray, shape (N+1, 5)
        Converged solution at x_{n+1}.
    """
    N = Y_old.shape[0] - 1

    x_np1 = xmid + dx / 2          # = x_n + dx, i.e. x_{n+1}
    Hm = H(xmid)
    Vm = V(xmid)
    Fm = F(xmid)
    I_val = I(x_np1)

    denom = xmid / dx - Hm / 2
    fjn, rjn = Y_old[0, 0], Y_old[0, 4]

    Y = Y_new.copy()

    # Enforce exact BCs up front, in case the initial guess doesn't satisfy them
    Y[0, 1] = 0.0
    Y[0, 3] = 0.0
    Y[N, 1] = I_val
    Y[N, 3] = 1.0
    Y[0, 0] = (-(K / 2) * (Y[0, 4] + rjn) + fjn * (xmid / dx + Hm / 2)) / denom

    converged = False
    for it in range(max_iter):
        # M, R = assemble_block_tridiagonal( Y, Y_old, xmid, dx, dy, Lambda, Pr, K, Hm, Fm, Vm, I_val, sparse=True )
        # dY = spsolve(M, R)
        
        M, R = assemble_block_tridiagonal_vectorized(Y, Y_old, xmid, dx, dy, Lambda, Pr, K, Hm, Fm, Vm, I_val, sparse=True)
        ab, lu, ld = sparse_to_banded(M)
        dY = solve_banded((ld, lu), ab, R)

        # --- unpack reduced correction back into the full state ---
        dq0, dr0 = dY[0], dY[1]
        Y[0, 2] += dq0
        Y[0, 4] += dr0
        # f0 recomputed exactly from the updated r0 (Stefan condition)
        Y[0, 0] = (-(K / 2) * (Y[0, 4] + rjn) + fjn * (xmid / dx + Hm / 2)) / denom

        for j in range(1, N):
            off = 2 + 5 * (j - 1)
            Y[j, 0] += dY[off]
            Y[j, 1] += dY[off + 1]
            Y[j, 2] += dY[off + 2]
            Y[j, 3] += dY[off + 3]
            Y[j, 4] += dY[off + 4]

        dfN, dqN, drN = dY[-3], dY[-2], dY[-1]
        Y[N, 0] += dfN
        Y[N, 2] += dqN
        Y[N, 4] += drN
        # Y[N,1]=I_val, Y[N,3]=1 stay fixed (already set)

        max_update = np.max(np.abs(dY))
        if max_update < tol:
            converged = True
            break

    if not converged:
        import warnings
        warnings.warn( f"newton_step: did not converge in {max_iter} iterations "
                      f"(last max|dY| = {max_update:.3e})" )

    return Y

def sol_boundary_layer(nx, ny, xmax, ymax, Pr, Lambda, K, max_iter=20, tol_newton=1e-8, eta_max=None, tol_bvp=1e-7, max_nodes=50000 ):

    y,dy = np.linspace(0, ymax, ny, retstep=True)
    x,dx = np.linspace(0, xmax, nx, retstep=True)

    Y_all = np.zeros((nx, ny, 5))

    #step 0
    funcs0 = [H(0), V(0), F(0), I(0)]
    sol, eqs = solve_ode_ini(y, Pr, Lambda, K, funcs0, eta_max=eta_max, tol=tol_bvp, max_nodes=max_nodes)
    # f, fp, fpp, th, thp = sol.sol(y)
    Y0 = build_Y0_from_ini(y,sol) 
    Y_all[0] = Y0

    #step 1
    Y_ini = Y_all[0]
    xm = x[0]+dx/2

    Y = newton_step(Y_ini, Y_all[0], H, V, F, I, xm, dx, dy, Pr, Lambda, K, max_iter=max_iter, tol=tol_newton)
    Y_all[1] = Y
    
    #steps until finished:
    for i in tqdm(range(2, nx)):
        Y_ini = 2 * Y_all[i-1] - Y_all[i-2]
        xm = x[i-1]+dx/2

        Y = newton_step(Y_ini, Y_all[i-1], H, V, F, I, xm, dx, dy, Pr, Lambda, K, max_iter=max_iter, tol=tol_newton)
        Y_all[i] = Y

    return x, y, Y_all        

def get_adim_vals( x, y, Y_all, Ste, Pr, dif_rho=1.09 ):
    xx,yy = np.meshgrid(x,y)
    f, fy, fyy, th, thy = Y_all.T
    
    _, fx = np.gradient(f, y, x, edge_order=2)
    
    u = xx * fy
    v = - ( H(xx) * f + xx * fx )
    
    v_ret = dif_rho * Ste / Pr * thy[0,:]
    
    return u, v, th, v_ret


#%%

# =============================================================================
# Check ini condition solver
# =============================================================================
eta = np.linspace(0, 4, 100)

fig, ax = plt.subplots(1,3, layout='constrained', figsize=(13,4))

# Pr = 7
coso = ['-','--']
for n,Pr in enumerate([7, 0.7]):

    K = 0
    Lambda = 3.6e-3
    
    funcs0 = [ 2., 9/4, 1, 3/2]
    
    sol, eqs = solve_ode_ini(eta, Pr, Lambda, K, funcs0 ) #, eta_max=None, tol=1e-7, max_nodes=50000):
    f, fp, fpp, theta, thetap =  sol.sol(eta)
    
    ax[0].plot(eta, fp, coso[n], label=r"$f'$")
    ax[0].plot(eta, f, coso[n], label=r"$f$")
    
    ax[0].plot(eta, theta, coso[n], label=r"$\Theta$")

ax[0].set_xlabel(r"$\eta$")
ax[0].legend()
ax[0].grid()
# plt.show()

 
for n,Lambda in enumerate([3.6e-3, 4]):

    K = 0
    Pr = 7
    
    funcs0 = [ 2., 9/4, 1, 3/2]
    
    sol, eqs = solve_ode_ini(eta, Pr, Lambda, K, funcs0 ) #, eta_max=None, tol=1e-7, max_nodes=50000):
    f, fp, fpp, theta, thetap =  sol.sol(eta)
    
    ax[1].plot(eta, fp, coso[n], label=r"$f'$")
    ax[1].plot(eta, f, coso[n], label=r"$f$")
    
    ax[1].plot(eta, theta, coso[n], label=r"$\Theta$")

ax[1].set_xlabel(r"$\eta$")
ax[1].legend()
ax[1].grid()

for n,K in enumerate([0, 0.1]):

    Pr = 7
    Lambda = 3.6e-3
    
    funcs0 = [ 2., 9/4, 1, 3/2]
    
    sol, eqs = solve_ode_ini(eta, Pr, Lambda, K, funcs0 ) #, eta_max=None, tol=1e-7, max_nodes=50000):
    f, fp, fpp, theta, thetap =  sol.sol(eta)
    
    ax[2].plot(eta, fp, coso[n], label=r"$f'$")
    ax[2].plot(eta, f, coso[n], label=r"$f$")
    
    ax[2].plot(eta, theta, coso[n], label=r"$\Theta$")

ax[2].set_xlabel(r"$\eta$")
ax[2].legend()
ax[2].grid()

plt.show()


#%%
eta = np.linspace(0, 4, 10)

Pr = 7
K = 0.0
Lambda = 3.6e-3

funcs0 = [ 2., 9/4, 1, 3/2]
H0, V0, F0, I0 = funcs0

t1 = time()
sol, eqs = solve_ode_ini(eta, Pr, Lambda, K, funcs0, max_nodes=1e5, tol=1e-7 )    
f, fp, fpp, th, thp =  sol.sol(eta)
eq1, eq2 = eqs 
t2 = time()
print( t2-t1 )

sol_th = cumulative_trapezoid( np.exp( -H0*Pr * cumulative_trapezoid( f, eta, initial=0 )), eta, initial=0 ) / \
        np.trapezoid( np.exp( -H0*Pr * cumulative_trapezoid( f, eta, initial=0 )), eta )



plt.figure()

# plt.plot(eta, fp)
# plt.plot(eta, th)

plt.plot(eta, eq1)
plt.plot(eta, eq2)

# plt.plot( eta, th - sol_th )
# plt.plot( eta, sol_th)

plt.show()



#%%

# =============================================================================
# Check solver
# =============================================================================

Pr = 7
K = 0.1
Lambda = 3.6e-3
    
nx, ny = 501, 1001
xmax, ymax = np.pi/2, 4

x, y, Y_all = sol_boundary_layer(nx, ny, xmax, ymax, Pr, Lambda, K, max_iter=40, tol_newton=1e-8, eta_max=None, tol_bvp=1e-7, max_nodes=50000 )


#%%
plt.figure()

# labs = [r'$f$',r'$p$',r'$q$',r'$\Theta$',r'$r$']
labs = [r'$f$',r'$f_y$',r'$f_{yy}$',r'$\Theta$',r'$\Theta_y$']
ns = [0,150,300]

# for i in range(5):
for i in [3]:
    plt.plot(y, (Y_all[ns[0]].T)[i], '-', label=f'{labs[i]}, x={x[ns[0]]:.4f}' )
    plt.plot(y, (Y_all[ns[1]].T)[i], '--', label=f'{labs[i]}, x={x[ns[1]]:.4f}' )
    plt.plot(y, (Y_all[ns[2]].T)[i], '--', label=f'{labs[i]}, x={x[ns[2]]:.4f}' )


plt.legend()
plt.grid()
plt.show()


fig, ax = plt.subplots(1,5, layout='constrained', sharey=True, figsize=(12,4))

for v in range(5):
    im = ax[v].imshow( Y_all[:,:,v].T, extent=(x[0],x[-1],y[0],y[-1]), origin='lower', aspect='auto' )
    # im = ax[v].imshow( Y_all[:,:,v].T, origin='lower' )

    fig.colorbar(im, ax=ax[v], shrink=0.9, pad=0.03)
    ax[v].set_title(labs[v])
    ax[v].set_xlabel(r'$x$')
    
ax[0].set_ylabel(r'$y$')

plt.show()


#%%
# =============================================================================
# Convergence test. I'll probably use 512x2048 and ymax=3
# =============================================================================

def calc_eq_residuals( x, y, Y_all ):
    xx,yy = np.meshgrid(x,y)
    f, fy, fyy, th, thy = Y_all.T
    
    fy_n, fx = np.gradient(f, y, x, edge_order=2)
    fyy_n, fyx = np.gradient(fy, y, x, edge_order=2)
    fyyy, fyyx = np.gradient(fyy, y, x, edge_order=2)
    
    thy_n, thx = np.gradient(th, y, x, edge_order=2)
    thyy, thyx = np.gradient(thy, y, x, edge_order=2)
    
    eq1 = fyyy + H(xx) * f * fyy - fy**2 + V(xx) + Lambda * F(xx) * (th - 1) - xx * (fy * fyx - fx * fyy)
    eq2 = thyy / Pr + H(xx) * f * thy - xx * (fy * thx - fx * thy)

    return np.max(np.abs(eq1)), np.max(np.abs(eq2)), (eq1, eq2)

xmax, ymax = np.pi/2, 3

r1s, r2s, nys = [],[],[]
xs, thy0 = [], []
levels = 5

nxs = [256,512,1024,2048]
# for level in range(levels):
    # nx, ny = 2**(6+1+level),  2**(6+1+level)

for nxx in nxs:
    nx, ny = nxx, 2048

    x, y, Y_all = sol_boundary_layer(nx, ny, xmax, ymax, Pr, Lambda, K, max_iter=40, tol_newton=1e-8, eta_max=None, tol_bvp=1e-7, max_nodes=50000 )
    f, fy, fyy, th, thy = Y_all.T
    r1, r2, (eq1,eq2) = calc_eq_residuals( x, y, Y_all )
    r1s.append(r1); r2s.append(r2); nys.append(ny)
    xs.append(x)
    thy0.append( thy[0,:] )

nys = np.array(nys)

fig, ax = plt.subplots(1,2, figsize=(9,4))

# ax[0].plot( nys, r1s, '.-' )
# ax[0].plot( nys, r2s, '.-' )

ax[0].plot( nxs, r1s, '.-' )
ax[0].plot( nxs, r2s, '.-' )

# plt.xscale('log')
# plt.yscale('log')

ax[0].set_ylim(0,.05)

# for i in range(levels):
#     ax[1].plot(xs[i], thy0[i], '-', label=nys[i])
for i in range(len(nxs)):
    ax[1].plot(xs[i], thy0[i], '-', label=nxs[i])
ax[1].legend()
plt.show()

# print( r1s, r2s)
#%%

r1s, r2s, mys = [],[],[2,3,4,6,8]
xs, ys, thy0 = [], [], []
th1 = []
nx, ny = 512*2, 1024

for my in mys:
    
    ymax = my
    x, y, Y_all = sol_boundary_layer(nx, ny, xmax, ymax, Pr, Lambda, K, max_iter=40, tol_newton=1e-8, eta_max=None, tol_bvp=1e-7, max_nodes=50000 )
    f, fy, fyy, th, thy = Y_all.T
    r1, r2, (eq1,eq2) = calc_eq_residuals( x, y, Y_all )
    r1s.append(r1); r2s.append(r2);
    xs.append(x); ys.append(y)
    thy0.append( thy[0,:] )
    th1.append( th[:,-20] )


fig, ax = plt.subplots(1,3, figsize=(13,4))

ax[0].plot( mys, r1s, '.-' )
ax[0].plot( mys, r2s, '.-' )

# plt.xscale('log')
# plt.yscale('log')

# ax[0].set_ylim(0,.05)

for i in range(len(mys)):
    ax[1].plot(xs[i], thy0[i], '-', label=mys[i])
ax[1].legend()

for i in range(len(mys)):
    ax[2].plot(ys[i], th1[i], '-', label=mys[i])
ax[2].legend()

plt.show()

print( r1s, r2s)



#%%
# =============================================================================
# With actual values
# =============================================================================
a = 0.017 # m
# Uinf = 0.4 #m/s
# a = 27.10672204743023 / 1000 # m
Uinf = 0.39976853 #m/s

nu = 1e-6 # m^2/s
g = 9.81 #m/s^2
Tinf = 20 #°C
Tm = 0 #°C
alpha = 0.14e-6 # m^2/s

beta = 2.07e-4 # 1/K

rho_i = 916.8 # kg / m^3

# rho_w = 998.19 # kg / m^3
rho_w = 999.89 # kg / m^3


latent = 334e3 # m^2 / s^2 o J/kg
cp = 4184 # J/(kg°C)


Re = a * Uinf / nu
Pr = nu / alpha
Ste = cp * (Tinf - Tm) / latent
Gr = g * beta * (Tinf-Tm) * a**3 / nu**2

K = (rho_w/rho_i - 1) * Ste/Pr * 0
Lambda = Gr / Re**2

print( Re, Gr, Pr, Ste )
print(K, Lambda)

nx, ny = 512, 2048
xmax, ymax = np.pi/2, 3

xad, yad, Y_all = sol_boundary_layer(nx, ny, xmax, ymax, Pr, Lambda, K, max_iter=40, tol_newton=1e-8, eta_max=None, tol_bvp=1e-7, max_nodes=50000 )

uad, vad, thad, v_retad = get_adim_vals( xad, yad, Y_all, Ste, Pr, dif_rho= rho_w / rho_i  )

x, y = xad * a, yad * a / np.sqrt(Re)
u, v = uad * Uinf, vad * Uinf / np.sqrt(Re)
T = thad * (Tinf-Tm) + Tm
v_ret = v_retad * Uinf / np.sqrt(Re)

#%%


plt.figure()

plt.imshow( T, extent=(x[0]/a * 180 /np.pi, x[-1]/a * 180 /np.pi,y[0],y[-1]), origin='lower', aspect='auto' )
plt.colorbar()

plt.show()

plt.figure()
# plt.plot( x/a * 180 /np.pi, v_ret * 1000, '-'  )

plt.plot( x/a * 180 /np.pi, v_ret * a / nu, '-'  )
plt.xlabel( r'$\theta$ (°)' )
plt.ylabel( r'$\dot{R} R / \nu$ ' )

plt.show()


#%%









#%%








y,dy = np.linspace(0, 4, ny, retstep=True)
x,dx = np.linspace(0, 1.5, nx, retstep=True)
# x,dx = np.linspace(0, np.pi/2, 101, retstep=True)

Y_all = np.zeros((nx, ny, 5))

funcs0 = [H(0), V(0), F(0), I(0)]
    
sol, eqs = solve_ode_ini(y, Pr, Lambda, K, funcs0 ) #, eta_max=None, tol=1e-7, max_nodes=50000)
f, fp, fpp, th, thp = sol.sol(y)
Y0 = build_Y0_from_ini(y,sol) # Y = f, p, q, Theta, r
Y_all[0] = Y0

t1 = time()

Y_ini = Y_all[0] + 0.0
xm, Hm, Vm, Fm, I_val = x[0]+dx/2, H( x[0]+dx/2 ), V( x[0]+dx/2 ), F( x[0]+dx/2 ), I( x[1] )
Y = newton_step(Y_ini, Y_all[0], H, V, F, I, xm, dx, dy, Pr, Lambda, K, max_iter=20, tol=1e-8)
Y_all[1] = Y

t2 = time()

Y_ini = 2 * Y_all[1] - Y_all[0]
xm, Hm, Vm, Fm, I_val = x[1]+dx/2, H( x[1]+dx/2 ), V( x[1]+dx/2 ), F( x[1]+dx/2 ), I( x[2] )
Y = newton_step(Y_ini, Y_all[1], H, V, F, I, xm, dx, dy, Pr, Lambda, K, max_iter=20, tol=1e-8)
Y_all[2] = Y

t3 = time()


print(t2-t1, t3-t2)


plt.figure()
# plt.imshow( mdiff )
# plt.colorbar()

labs = [r'$f$',r'$p$',r'$q$',r'$\Theta$',r'$r$']
# for i in range(5):
for i in [1]:
    plt.plot(y, (Y_all[0].T)[i], '-', label=labs[i] )
    plt.plot(y, (Y_all[1].T)[i], '--', label=labs[i] )
    plt.plot(y, (Y_all[2].T)[i], '--', label=labs[i] )

plt.legend()
plt.grid()
plt.show()






















#%%







#%%









#%%














#%%

















#%%















#%%







