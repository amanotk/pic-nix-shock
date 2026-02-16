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


def apply_bc_periodic(arr, Nx, Ny, axis0_size=None):
    """
    Apply periodic boundary conditions to a field array.

    For periodic BC in both x and y:
        A[-1, j] -> A[Nx-1, j]
        A[Nx, j] -> A[0, j]
        A[i, -1] -> A[i, Ny-1]
        A[i, Ny] -> A[i, 0]

    Parameters
    ----------
    arr : ndarray
        Input array of shape (Nx, Ny), (3, Nx, Ny), or (3, 3, Nx, Ny)
    Nx : int
        Number of points in x
    Ny : int
        Number of points in y
    axis0_size : int, optional
        Size of first axis (1 for scalar, 3 for vector, 9 for tensor)
        If None, inferred from arr.shape[0]

    Returns
    -------
    ndarray
        Array with periodic BC applied (same shape as input)
    """
    if arr.ndim == 2:
        result = np.empty((Nx + 2, Ny + 2), dtype=arr.dtype)
        result[1:-1, 1:-1] = arr
        result[0, 1:-1] = arr[Nx - 1, :]
        result[Nx + 1, 1:-1] = arr[0, :]
        result[1:-1, 0] = arr[:, Ny - 1]
        result[1:-1, Ny + 1] = arr[:, 0]
        result[0, 0] = arr[Nx - 1, Ny - 1]
        result[0, Ny + 1] = arr[Nx - 1, 0]
        result[Nx + 1, 0] = arr[0, Ny - 1]
        result[Nx + 1, Ny + 1] = arr[0, 0]
    elif arr.ndim == 3:
        result = np.empty((arr.shape[0], Nx + 2, Ny + 2), dtype=arr.dtype)
        result[:, 1:-1, 1:-1] = arr
        result[:, 0, 1:-1] = arr[:, Nx - 1, :]
        result[:, Nx + 1, 1:-1] = arr[:, 0, :]
        result[:, 1:-1, 0] = arr[:, :, Ny - 1]
        result[:, 1:-1, Ny + 1] = arr[:, :, 0]
        result[:, 0, 0] = arr[:, Nx - 1, Ny - 1]
        result[:, 0, Ny + 1] = arr[:, Nx - 1, 0]
        result[:, Nx + 1, 0] = arr[:, 0, Ny - 1]
        result[:, Nx + 1, Ny + 1] = arr[:, 0, 0]
    else:
        result = np.empty((arr.shape[0], arr.shape[1], Nx + 2, Ny + 2), dtype=arr.dtype)
        result[:, :, 1:-1, 1:-1] = arr
        result[:, :, 0, 1:-1] = arr[:, :, Nx - 1, :]
        result[:, :, Nx + 1, 1:-1] = arr[:, :, 0, :]
        result[:, :, 1:-1, 0] = arr[:, :, :, Ny - 1]
        result[:, :, 1:-1, Ny + 1] = arr[:, :, :, 0]
        result[:, :, 0, 0] = arr[:, :, Nx - 1, Ny - 1]
        result[:, :, 0, Ny + 1] = arr[:, :, Nx - 1, 0]
        result[:, :, Nx + 1, 0] = arr[:, :, 0, Ny - 1]
        result[:, :, Nx + 1, Ny + 1] = arr[:, :, 0, 0]
    return result


def apply_bc_neumann_x_periodic_y(arr, Nx, Ny, axis0_size=None):
    """
    Apply Neumann BC in x (∂/∂x = 0 at boundaries) and periodic in y.

    Uses even reflection for ghost cells:
        A[-1, j] = A[0, j]
        A[Nx, j] = A[Nx-1, j]
    And periodic in y:
        A[i, -1] = A[i, Ny-1]
        A[i, Ny] = A[i, 0]

    Parameters
    ----------
    arr : ndarray
        Input array of shape (Nx, Ny), (3, Nx, Ny), or (3, 3, Nx, Ny)
    Nx : int
        Number of points in x
    Ny : int
        Number of points in y
    axis0_size : int, optional
        Size of first axis (1 for scalar, 3 for vector, 9 for tensor)

    Returns
    -------
    ndarray
        Array with BC applied (same shape as input)
    """
    if arr.ndim == 2:
        result = np.empty((Nx + 2, Ny + 2), dtype=arr.dtype)
        result[1:-1, 1:-1] = arr
        result[0, 1:-1] = arr[0, :]
        result[Nx + 1, 1:-1] = arr[Nx - 1, :]
        result[1:-1, 0] = arr[:, Ny - 1]
        result[1:-1, Ny + 1] = arr[:, 0]
        result[0, 0] = arr[0, Ny - 1]
        result[0, Ny + 1] = arr[0, 0]
        result[Nx + 1, 0] = arr[Nx - 1, Ny - 1]
        result[Nx + 1, Ny + 1] = arr[Nx - 1, 0]
    elif arr.ndim == 3:
        result = np.empty((arr.shape[0], Nx + 2, Ny + 2), dtype=arr.dtype)
        result[:, 1:-1, 1:-1] = arr
        result[:, 0, 1:-1] = arr[:, 0, :]
        result[:, Nx + 1, 1:-1] = arr[:, Nx - 1, :]
        result[:, 1:-1, 0] = arr[:, :, Ny - 1]
        result[:, 1:-1, Ny + 1] = arr[:, :, 0]
        result[:, 0, 0] = arr[:, 0, Ny - 1]
        result[:, 0, Ny + 1] = arr[:, 0, 0]
        result[:, Nx + 1, 0] = arr[:, Nx - 1, Ny - 1]
        result[:, Nx + 1, Ny + 1] = arr[:, Nx - 1, 0]
    else:
        result = np.empty((arr.shape[0], arr.shape[1], Nx + 2, Ny + 2), dtype=arr.dtype)
        result[:, :, 1:-1, 1:-1] = arr
        result[:, :, 0, 1:-1] = arr[:, :, 0, :]
        result[:, :, Nx + 1, 1:-1] = arr[:, :, Nx - 1, :]
        result[:, :, 1:-1, 0] = arr[:, :, :, Ny - 1]
        result[:, :, 1:-1, Ny + 1] = arr[:, :, :, 0]
        result[:, :, 0, 0] = arr[:, :, 0, Ny - 1]
        result[:, :, 0, Ny + 1] = arr[:, :, 0, 0]
        result[:, :, Nx + 1, 0] = arr[:, :, Nx - 1, Ny - 1]
        result[:, :, Nx + 1, Ny + 1] = arr[:, :, Nx - 1, 0]
    return result


def assemble_ex_ey_matrix(Nx, Ny, Lambda, c2_dx2, c2_dx4, bc_type):
    """
    Assemble sparse matrix for coupled (Ex, Ey) system.

    Builds a 2N × 2N block matrix where N = Nx * Ny:
    - Axx: Ex y-second-difference + Λ
    - Ayy: Ey x-second-difference + Λ
    - Axy, Ayx: mixed derivative terms (diagonal neighbors)

    The discretization follows wavetool.md equations (2a)-(2b).

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
    bc_type : str
        'periodic' or 'neumann_x_periodic_y'

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

            k_ym = idx(i, (j - 1) % Ny, Nx) if bc_type == "periodic" else idx(i, max(j - 1, 0), Nx)
            k_yp = (
                idx(i, (j + 1) % Ny, Nx)
                if bc_type == "periodic"
                else idx(i, min(j + 1, Ny - 1), Nx)
            )
            k_xm = idx((i - 1) % Nx, j, Nx) if bc_type == "periodic" else idx(max(i - 1, 0), j, Nx)
            k_xp = (
                idx((i + 1) % Nx, j, Nx)
                if bc_type == "periodic"
                else idx(min(i + 1, Nx - 1), j, Nx)
            )

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

            if bc_type == "neumann_x_periodic_y":
                if i == 0:
                    k_xm_diag = idx(0, j, Nx)
                    k_xp_diag = idx(1, j, Nx)
                    k_xm_diag_ym = idx(0, max(j - 1, 0), Nx)
                    k_xm_diag_yp = idx(0, min(j + 1, Ny - 1), Nx)
                elif i == Nx - 1:
                    k_xm_diag = idx(Nx - 2, j, Nx)
                    k_xp_diag = idx(Nx - 1, j, Nx)
                    k_xm_diag_ym = idx(Nx - 1, max(j - 1, 0), Nx)
                    k_xm_diag_yp = idx(Nx - 1, min(j + 1, Ny - 1), Nx)
                else:
                    k_xm_diag = k_xm
                    k_xp_diag = k_xp
                    k_xm_diag_ym = idx((i - 1) % Nx, max(j - 1, 0), Nx) if j > 0 else k_xm
                    k_xm_diag_yp = idx((i - 1) % Nx, min(j + 1, Ny - 1), Nx)
            else:
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
            k_ym = idx(i, (j - 1) % Ny, Nx) if bc_type == "periodic" else idx(i, max(j - 1, 0), Nx)
            k_yp = (
                idx(i, (j + 1) % Ny, Nx)
                if bc_type == "periodic"
                else idx(i, min(j + 1, Ny - 1), Nx)
            )
            k_xm = idx((i - 1) % Nx, j, Nx) if bc_type == "periodic" else idx(max(i - 1, 0), j, Nx)
            k_xp = (
                idx((i + 1) % Nx, j, Nx)
                if bc_type == "periodic"
                else idx(min(i + 1, Nx - 1), j, Nx)
            )

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


def assemble_ez_matrix(Nx, Ny, Lambda, c2_dx2, bc_type):
    """
    Assemble sparse matrix for Ez (independent from Ex, Ey).

    From wavetool.md, equation (2c):
    (∇×∇×E)_z ≈ -(c²/Δ²)(Ez_{i+1,j} - 2Ez_{i,j} + Ez_{i-1,j})
                 -(c²/Δ²)(Ez_{i,j+1} - 2Ez_{i,j} + Ez_{i,j-1})

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
    bc_type : str
        'periodic' or 'neumann_x_periodic_y'

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

            k_ym = idx(i, (j - 1) % Ny, Nx) if bc_type == "periodic" else idx(i, max(j - 1, 0), Nx)
            k_yp = (
                idx(i, (j + 1) % Ny, Nx)
                if bc_type == "periodic"
                else idx(i, min(j + 1, Ny - 1), Nx)
            )
            k_xm = idx((i - 1) % Nx, j, Nx) if bc_type == "periodic" else idx(max(i - 1, 0), j, Nx)
            k_xp = (
                idx((i + 1) % Nx, j, Nx)
                if bc_type == "periodic"
                else idx(min(i + 1, Nx - 1), j, Nx)
            )

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


def compute_source(Gamma, Pi, B, c, delta, bc_type):
    """
    Compute source term S = -(Γ/c)×B + ∇·Π.

    From wavetool.md, equations (3a)-(3c):
    S^x ≈ -(1/c)(Γ^y B^z - Γ^z B^y)
          + (1/(2Δ))[(Π^{xx}_{i+1,j}-Π^{xx}_{i-1,j}) + (Π^{xy}_{i,j+1}-Π^{xy}_{i,j-1})]

    S^y ≈ -(1/c)(Γ^z B^x - Γ^x B^z)
          + (1/(2Δ))[(Π^{yx}_{i+1,j}-Π^{yx}_{i-1,j}) + (Π^{yy}_{i,j+1}-Π^{yy}_{i,j-1})]

    S^z ≈ -(1/c)(Γ^x B^y - Γ^y B^x)
          + (1/(2Δ))[(Π^{zx}_{i+1,j}-Π^{zx}_{i-1,j}) + (Π^{zy}_{i,j+1}-Π^{zy}_{i,j-1})]

    Parameters
    ----------
    Gamma : ndarray
        Gamma array of shape (3, Nx, Ny)
    Pi : ndarray
        Pi tensor of shape (3, 3, Nx, Ny)
    B : ndarray
        B field of shape (3, Nx, Ny)
    c : float
        Speed of light
    delta : float
        Grid spacing Δ
    bc_type : str
        'periodic' or 'neumann_x_periodic_y'

    Returns
    -------
    ndarray
        Source term S of shape (3, Nx, Ny)
    """
    Nx, Ny = Gamma.shape[1], Gamma.shape[2]

    if bc_type == "periodic":
        Gamma_ext = apply_bc_periodic(Gamma, Nx, Ny)
        Pi_ext = apply_bc_periodic(Pi, Nx, Ny)
        B_ext = apply_bc_periodic(B, Nx, Ny)
    else:
        Gamma_ext = apply_bc_neumann_x_periodic_y(Gamma, Nx, Ny)
        Pi_ext = apply_bc_neumann_x_periodic_y(Pi, Nx, Ny)
        B_ext = apply_bc_neumann_x_periodic_y(B, Nx, Ny)

    S = np.zeros((3, Nx, Ny))
    inv_2d = 1.0 / (2.0 * delta)

    for j in range(1, Ny + 1):
        for i in range(1, Nx + 1):
            Gx = Gamma_ext[0, i, j]
            Gy = Gamma_ext[1, i, j]
            Gz = Gamma_ext[2, i, j]

            Bx = B_ext[0, i, j]
            By = B_ext[1, i, j]
            Bz = B_ext[2, i, j]

            S[0, i - 1, j - 1] = -(1.0 / c) * (Gy * Bz - Gz * By) + inv_2d * (
                Pi_ext[0, 0, i + 1, j]
                - Pi_ext[0, 0, i - 1, j]
                + Pi_ext[0, 1, i, j + 1]
                - Pi_ext[0, 1, i, j - 1]
            )

            S[1, i - 1, j - 1] = -(1.0 / c) * (Gz * Bx - Gx * Bz) + inv_2d * (
                Pi_ext[1, 0, i + 1, j]
                - Pi_ext[1, 0, i - 1, j]
                + Pi_ext[1, 1, i, j + 1]
                - Pi_ext[1, 1, i, j - 1]
            )

            S[2, i - 1, j - 1] = -(1.0 / c) * (Gx * By - Gy * Bx) + inv_2d * (
                Pi_ext[2, 0, i + 1, j]
                - Pi_ext[2, 0, i - 1, j]
                + Pi_ext[2, 1, i, j + 1]
                - Pi_ext[2, 1, i, j - 1]
            )

    return S


def solve_ohm_2d(Lambda, Gamma, Pi, B, c, delta, bc="periodic", solver_opts=None, S=None):
    """
    Solve the 2D generalized Ohm's law.

    Solves: (Λ + c²∇×∇×)E = -(Γ/c)×B + ∇·Π

    Parameters
    ----------
    Lambda : ndarray
        Lambda array of shape (Nx, Ny)
    Gamma : ndarray
        Gamma vector of shape (3, Nx, Ny)
    Pi : ndarray
        Pi tensor of shape (3, 3, Nx, Ny)
    B : ndarray
        B field of shape (3, Nx, Ny)
    c : float
        Speed of light
    delta : float
        Grid spacing Δ (assumes dx = dy = Δ)
    bc : str, optional
        Boundary condition: 'periodic' or 'neumann_x_periodic_y'
    solver_opts : dict, optional
        Solver options:
        - 'method': 'direct' (default) or 'iterative'
        - 'tol': tolerance for iterative solver (default 1e-10)
        - 'maxiter': max iterations for GMRES (default 1000)
        - 'use_amg': use AMG preconditioner if available (default True)
    S : ndarray, optional
        Pre-computed source term of shape (3, Nx, Ny).
        If provided, overrides the internal computation from Gamma, Pi, B.

    Returns
    -------
    ndarray
        Electric field E of shape (3, Nx, Ny)

    Notes
    -----
    The discretization follows wavetool.md:
    - Equations (2a)-(2c) for (∇×∇×E) finite-difference approximation
    - Equations (3a)-(3c) for source term S = -(Γ/c)×B + ∇·Π
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

    if S is None:
        S = compute_source(Gamma, Pi, B, c, delta, bc)

    A_ex_ey = assemble_ex_ey_matrix(Nx, Ny, Lambda, c2_dx2, c2_dx4, bc)
    A_ez = assemble_ez_matrix(Nx, Ny, Lambda, c2_dx2, bc)

    S_ex_ey = np.concatenate([S[0].flatten(order="F"), S[1].flatten(order="F")])
    S_ez = S[2].flatten(order="F")

    if method == "direct":
        E_ex_ey = spsolve(A_ex_ey, S_ex_ey)
        E_ez = spsolve(A_ez, S_ez)
    else:
        if use_amg and HAS_PYAMG:
            from scipy.sparse.linalg import LinearOperator

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


def apply_ohm_operator(E, Lambda, c, delta, bc_type):
    """
    Apply the Ohm operator (Λ + c²∇×∇×) to a field E.

    Useful for testing and verification.

    Parameters
    ----------
    E : ndarray
        Electric field of shape (3, Nx, Ny)
    Lambda : ndarray
        Lambda array of shape (Nx, Ny)
    c : float
        Speed of light
    delta : float
        Grid spacing
    bc_type : str
        'periodic' or 'neumann_x_periodic_y'

    Returns
    -------
    ndarray
        (Λ + c²∇×∇×)E of shape (3, Nx, Ny)
    """
    Nx, Ny = E.shape[1], E.shape[2]
    c2_dx2 = (c * c) / (delta * delta)
    c2_dx4 = (c * c) / (4.0 * delta * delta)

    if bc_type == "periodic":
        E_ext = apply_bc_periodic(E, Nx, Ny, 3)
    else:
        E_ext = apply_bc_neumann_x_periodic_y(E, Nx, Ny, 3)

    result = np.zeros((3, Nx, Ny))

    for j in range(1, Ny + 1):
        for i in range(1, Nx + 1):
            result[0, i - 1, j - 1] = (
                Lambda[i - 1, j - 1] * E[0, i - 1, j - 1]
                - c2_dx2 * (E_ext[0, i, j + 1] - 2 * E_ext[0, i, j] + E_ext[0, i, j - 1])
                + c2_dx4
                * (
                    E_ext[1, i + 1, j + 1]
                    - E_ext[1, i + 1, j - 1]
                    - E_ext[1, i - 1, j + 1]
                    + E_ext[1, i - 1, j - 1]
                )
            )

    for j in range(1, Ny + 1):
        for i in range(1, Nx + 1):
            result[1, i - 1, j - 1] = (
                Lambda[i - 1, j - 1] * E[1, i - 1, j - 1]
                - c2_dx2 * (E_ext[1, i + 1, j] - 2 * E_ext[1, i, j] + E_ext[1, i - 1, j])
                + c2_dx4
                * (
                    E_ext[0, i + 1, j + 1]
                    - E_ext[0, i + 1, j - 1]
                    - E_ext[0, i - 1, j + 1]
                    + E_ext[0, i - 1, j - 1]
                )
            )

    for j in range(1, Ny + 1):
        for i in range(1, Nx + 1):
            result[2, i - 1, j - 1] = (
                Lambda[i - 1, j - 1] * E[2, i - 1, j - 1]
                - c2_dx2 * (E_ext[2, i + 1, j] - 2 * E_ext[2, i, j] + E_ext[2, i - 1, j])
                - c2_dx2 * (E_ext[2, i, j + 1] - 2 * E_ext[2, i, j] + E_ext[2, i, j - 1])
            )

    return result
