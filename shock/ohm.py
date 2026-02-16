#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D Generalized Ohm's Law Solver.

This module implements a finite-difference solver for the generalized Ohm's law
in 2D (∂/∂z = 0) with collocated grid, following the discretization in
`docs/wavetool.md`.

Generalized Ohm's law (Lorentz-Heaviside units):
    (Λ + c^2 ∇×∇×) E = - (Γ/c) × B + ∇·Π

References:
    - wavetool.md, equations (1)-(3), finite difference approximations
    - Amano, T. (2018). J. Comput. Phys. 366, 366-385.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import gmres, spsolve

try:
    import pyamg

    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False


def idx(i, j, Nx, Ny=None):
    """
    Flatten 2D index to 1D.

    Uses column-major ordering: k = i + Nx * j.
    This matches Fortran order and the meshgrid convention where
    the first index varies fastest.

    Parameters
    ----------
    i : int
        x-index (0 <= i < Nx)
    j : int
        y-index (0 <= j < Ny)
    Nx : int
        Number of grid points in x
    Ny : int, optional
        Number of grid points in y (if not given, Nx is used as stride)

    Returns
    -------
    int
        Flattened index k = i + Nx * j
    """
    if Ny is None:
        Ny = Nx
    return i + Nx * j


def assemble_ex_ey_matrix(Nx, Ny, Lambda, c2_dx2, c2_dx4):
    """
    Assemble sparse matrix for coupled (Ex, Ey) system.

    Builds a 2N × 2N block matrix where N = Nx * Ny:
    - Axx: Ex y-second-difference + Λ
    - Ayy: Ey x-second-difference + Λ
    - Axy, Ayx: mixed derivative terms (diagonal neighbors)

    The discretization follows wavetool.md equations (2a)-(2b).
    Uses periodic boundary conditions.

    Parameters
    ----------
    Nx : int
        Number of grid points in x
    Ny : int
        Number of grid points in y
    Lambda : ndarray
        Lambda array of shape (Nx, Ny), may be spatially varying
    c2_dx2 : float
        c²/Δ²
    c2_dx4 : float
        c²/(4Δ²)

    Returns
    -------
    sparse.csr_matrix
        Sparse matrix of shape (2*N, 2*N)
    """
    N = Nx * Ny
    nnz_estimate = 20 * N
    data = np.zeros(nnz_estimate)
    row = np.zeros(nnz_estimate, dtype=np.int32)
    col = np.zeros(nnz_estimate, dtype=np.int32)
    nnz = 0

    Lambda_flat = Lambda.flatten(order="F")

    for j in range(Ny):
        for i in range(Nx):
            k = idx(i, j, Nx)

            k_ym = idx(i, (j - 1) % Ny, Nx)
            k_yp = idx(i, (j + 1) % Ny, Nx)
            k_xm = idx((i - 1) % Nx, j, Nx)
            k_xp = idx((i + 1) % Nx, j, Nx)

            lambda_val = Lambda_flat[k]

            row[nnz] = k
            col[nnz] = k
            data[nnz] = lambda_val + 2 * c2_dx2
            nnz += 1

            row[nnz] = k
            col[nnz] = k_ym
            data[nnz] = -c2_dx2
            nnz += 1

            row[nnz] = k
            col[nnz] = k_yp
            data[nnz] = -c2_dx2
            nnz += 1

            k_xm_diag = k_xm
            k_xp_diag = k_xp
            k_xm_diag_ym = idx((i - 1) % Nx, (j - 1) % Ny, Nx)
            k_xm_diag_yp = idx((i - 1) % Nx, (j + 1) % Ny, Nx)

            row[nnz] = k
            col[nnz] = N + idx((i + 1) % Nx, (j + 1) % Ny, Nx)
            data[nnz] = c2_dx4
            nnz += 1

            row[nnz] = k
            col[nnz] = N + idx((i + 1) % Nx, (j - 1) % Ny, Nx)
            data[nnz] = -c2_dx4
            nnz += 1

            row[nnz] = k
            col[nnz] = N + idx((i - 1) % Nx, (j + 1) % Ny, Nx)
            data[nnz] = -c2_dx4
            nnz += 1

            row[nnz] = k
            col[nnz] = N + idx((i - 1) % Nx, (j - 1) % Ny, Nx)
            data[nnz] = c2_dx4
            nnz += 1

    for j in range(Ny):
        for i in range(Nx):
            k = idx(i, j, Nx)
            k_ym = idx(i, (j - 1) % Ny, Nx)
            k_yp = idx(i, (j + 1) % Ny, Nx)
            k_xm = idx((i - 1) % Nx, j, Nx)
            k_xp = idx((i + 1) % Nx, j, Nx)

            lambda_val = Lambda_flat[k]

            row[nnz] = N + k
            col[nnz] = N + k
            data[nnz] = lambda_val + 2 * c2_dx2
            nnz += 1

            row[nnz] = N + k
            col[nnz] = N + k_xm
            data[nnz] = -c2_dx2
            nnz += 1

            row[nnz] = N + k
            col[nnz] = N + k_xp
            data[nnz] = -c2_dx2
            nnz += 1

            row[nnz] = N + k
            col[nnz] = idx((i + 1) % Nx, (j + 1) % Ny, Nx)
            data[nnz] = c2_dx4
            nnz += 1

            row[nnz] = N + k
            col[nnz] = idx((i + 1) % Nx, (j - 1) % Ny, Nx)
            data[nnz] = -c2_dx4
            nnz += 1

            row[nnz] = N + k
            col[nnz] = idx((i - 1) % Nx, (j + 1) % Ny, Nx)
            data[nnz] = -c2_dx4
            nnz += 1

            row[nnz] = N + k
            col[nnz] = idx((i - 1) % Nx, (j - 1) % Ny, Nx)
            data[nnz] = c2_dx4
            nnz += 1

    data = data[:nnz]
    row = row[:nnz]
    col = col[:nnz]

    A = sparse.csr_matrix((data, (row, col)), shape=(2 * N, 2 * N))
    A.eliminate_zeros()

    return A


def assemble_ez_matrix(Nx, Ny, Lambda, c2_dx2):
    """
    Assemble sparse matrix for Ez (independent from Ex, Ey).

    From wavetool.md, equation (2c):
    (∇×∇×E)_z ≈ -(c²/Δ²)(Ez_{i+1,j} - 2Ez_{i,j} + Ez_{i-1,j})
                 -(c²/Δ²)(Ez_{i,j+1} - 2Ez_{i,j} + Ez_{i,j-1})

    Uses periodic boundary conditions.

    Parameters
    ----------
    Nx : int
        Number of grid points in x
    Ny : int
        Number of grid points in y
    Lambda : ndarray
        Lambda array of shape (Nx, Ny)
    c2_dx2 : float
        c²/Δ²

    Returns
    -------
    sparse.csr_matrix
        Sparse matrix of shape (N, N)
    """
    N = Nx * Ny
    nnz_estimate = 7 * N
    data = np.zeros(nnz_estimate)
    row = np.zeros(nnz_estimate, dtype=np.int32)
    col = np.zeros(nnz_estimate, dtype=np.int32)
    nnz = 0

    Lambda_flat = Lambda.flatten(order="F")

    for j in range(Ny):
        for i in range(Nx):
            k = idx(i, j, Nx)

            k_ym = idx(i, (j - 1) % Ny, Nx)
            k_yp = idx(i, (j + 1) % Ny, Nx)
            k_xm = idx((i - 1) % Nx, j, Nx)
            k_xp = idx((i + 1) % Nx, j, Nx)

            lambda_val = Lambda_flat[k]

            row[nnz] = k
            col[nnz] = k
            data[nnz] = lambda_val + 4 * c2_dx2
            nnz += 1

            row[nnz] = k
            col[nnz] = k_xm
            data[nnz] = -c2_dx2
            nnz += 1

            row[nnz] = k
            col[nnz] = k_xp
            data[nnz] = -c2_dx2
            nnz += 1

            row[nnz] = k
            col[nnz] = k_ym
            data[nnz] = -c2_dx2
            nnz += 1

            row[nnz] = k
            col[nnz] = k_yp
            data[nnz] = -c2_dx2
            nnz += 1

    data = data[:nnz]
    row = row[:nnz]
    col = col[:nnz]

    A = sparse.csr_matrix((data, (row, col)), shape=(N, N))
    A.eliminate_zeros()

    return A


def solve_ohm_2d(Lambda, S, c, delta, solver_opts=None):
    """
    Solve the 2D generalized Ohm's law.

    Solves: (Λ + c²∇×∇×)E = S
    Uses periodic boundary conditions.

    Parameters
    ----------
    Lambda : ndarray
        Lambda array of shape (Nx, Ny)
    S : ndarray
        Source term of shape (3, Nx, Ny)
    c : float
        Speed of light
    delta : float
        Grid spacing Δ (assumes dx = dy = Δ)
    solver_opts : dict, optional
        Solver options:
        - 'method': 'direct' (default) or 'iterative'
        - 'tol': tolerance for iterative solver (default 1e-10)
        - 'maxiter': max iterations for GMRES (default 1000)
        - 'use_amg': use AMG preconditioner if available (default True)

    Returns
    -------
    ndarray
        Electric field E of shape (3, Nx, Ny)

    Notes
    -----
    The discretization follows wavetool.md:
    - Equations (2a)-(2c) for (∇×∇×E) finite-difference approximation
    """
    if solver_opts is None:
        solver_opts = {}

    method = solver_opts.get("method", "direct")
    tol = solver_opts.get("tol", 1e-10)
    maxiter = solver_opts.get("maxiter", 1000)
    use_amg = solver_opts.get("use_amg", True)

    Nx, Ny = Lambda.shape
    c2 = c * c
    c2_dx2 = c2 / (delta * delta)
    c2_dx4 = c2 / (4.0 * delta * delta)

    A_ex_ey = assemble_ex_ey_matrix(Nx, Ny, Lambda, c2_dx2, c2_dx4)
    A_ez = assemble_ez_matrix(Nx, Ny, Lambda, c2_dx2)

    S_ex_ey = np.concatenate([S[0].flatten(order="F"), S[1].flatten(order="F")])
    S_ez = S[2].flatten(order="F")

    if method == "direct":
        E_ex_ey = spsolve(A_ex_ey, S_ex_ey)
        E_ez = spsolve(A_ez, S_ez)
    else:
        if use_amg and HAS_PYAMG:
            M_ex_ey = pyamg.ruge_stuben(A_ex_ey)
            M_ez = pyamg.ruge_stuben(A_ez)

            def preconditioner_ex_ey(r):
                return pyamg.gmres(M_ex_ey, r, maxiter=1)[0]

            def preconditioner_ez(r):
                return pyamg.gmres(M_ez, r, maxiter=1)[0]
        else:
            from scipy.sparse.linalg import spilu

            M_ex_ey = spilu(A_ex_ey, drop_tol=1e-4)
            M_ez = spilu(A_ez, drop_tol=1e-4)

            def preconditioner_ex_ey(r):
                return M_ex_ey.solve(r)

            def preconditioner_ez(r):
                return M_ez.solve(r)

        E_ex_ey, info_ex_ey = gmres(
            A_ex_ey,
            S_ex_ey,
            M=preconditioner_ex_ey,
            rtol=tol,
            maxiter=maxiter,
            restart=min(2 * Nx * Ny, 100),
        )
        E_ez, info_ez = gmres(
            A_ez, S_ez, M=preconditioner_ez, rtol=tol, maxiter=maxiter, restart=min(Nx * Ny, 100)
        )

        if info_ex_ey != 0:
            import warnings

            warnings.warn(f"GMRES for Ex,Ey did not converge, info={info_ex_ey}")
        if info_ez != 0:
            import warnings

            warnings.warn(f"GMRES for Ez did not converge, info={info_ez}")

    E = np.zeros((3, Nx, Ny))
    E[0] = E_ex_ey[: Nx * Ny].reshape((Nx, Ny), order="F")
    E[1] = E_ex_ey[Nx * Ny :].reshape((Nx, Ny), order="F")
    E[2] = E_ez.reshape((Nx, Ny), order="F")

    return E
