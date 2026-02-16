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
from scipy.sparse.linalg import gmres


def idx(i, j, Nx, Ny=None):
    """
    Flatten 2D index to 1D.

    Uses x-fastest flattening: k = i + Nx * j.
    For arrays shaped (Ny, Nx), this matches C-order flattening.

    Parameters
    ----------
    i : int
        x-index (0 <= i < Nx)
    j : int
        y-index (0 <= j < Ny)
    Nx : int
        Number of grid points in x
    Ny : int, optional
        Number of grid points in y (unused, kept for compatibility)

    Returns
    -------
    int
        Flattened index k = i + Nx * j
    """
    if Ny is None:
        Ny = Nx
    return i + Nx * j


def assemble_matrix_1(Nx, Ny, Lambda, c2_dx2, c2_dx4):
    """
    Assemble sparse matrix for system 1 (Ex, Ey coupled).

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
        Lambda array of shape (Ny, Nx), may be spatially varying
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

    Lambda_flat = Lambda.flatten(order="C")

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


def assemble_matrix_2(Nx, Ny, Lambda, c2_dx2):
    """
    Assemble sparse matrix for system 2 (Ez decoupled).

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
        Lambda array of shape (Ny, Nx)
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

    Lambda_flat = Lambda.flatten(order="C")

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


def solve_ohm_2d(Lambda, S, c, delta):
    """
    Solve the 2D generalized Ohm's law.

    Solves: (Λ + c²∇×∇×)E = S
    Uses periodic boundary conditions.

    Parameters
    ----------
    Lambda : ndarray
        Lambda array of shape (Ny, Nx)
    S : ndarray
        Source term of shape (Ny, Nx, 3)
    c : float
        Speed of light
    delta : float
        Grid spacing Δ (assumes dx = dy = Δ)

    Returns
    -------
    ndarray
        Electric field E of shape (Ny, Nx, 3)

    Notes
    -----
    The discretization follows wavetool.md:
    - Equations (2a)-(2c) for (∇×∇×E) finite-difference approximation
    """
    Ny, Nx = Lambda.shape
    c2 = c * c
    c2_dx2 = c2 / (delta * delta)
    c2_dx4 = c2 / (4.0 * delta * delta)

    A_1 = assemble_matrix_1(Nx, Ny, Lambda, c2_dx2, c2_dx4)
    A_2 = assemble_matrix_2(Nx, Ny, Lambda, c2_dx2)

    S_1 = np.concatenate([S[..., 0].flatten(order="C"), S[..., 1].flatten(order="C")])
    S_2 = S[..., 2].flatten(order="C")

    N = Nx * Ny
    E_1, _ = gmres(A_1, S_1, restart=min(2 * N, 100), maxiter=1000)
    E_2, _ = gmres(A_2, S_2, restart=min(N, 100), maxiter=1000)

    E = np.zeros((Ny, Nx, 3))
    E[..., 0] = E_1[: Nx * Ny].reshape((Ny, Nx), order="C")
    E[..., 1] = E_1[Nx * Ny :].reshape((Ny, Nx), order="C")
    E[..., 2] = E_2.reshape((Ny, Nx), order="C")

    return E
