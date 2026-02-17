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


def _periodic_second_difference_matrix(n):
    """Return periodic 1D second-difference matrix (n x n)."""
    D = sparse.lil_matrix((n, n))
    D.setdiag(-2.0)
    D.setdiag(1.0, k=-1)
    D.setdiag(1.0, k=1)
    D[0, n - 1] = 1.0
    D[n - 1, 0] = 1.0
    return D.tocsr()


def _periodic_first_difference_matrix(n):
    """Return periodic 1D first-difference matrix (n x n), no 1/(2Δ) factor."""
    D = sparse.lil_matrix((n, n))
    D.setdiag(-1.0, k=-1)
    D.setdiag(1.0, k=1)
    D[0, n - 1] = -1.0
    D[n - 1, 0] = 1.0
    return D.tocsr()


def build_ohm_bases(Nx, Ny, c2_dx2, c2_dx4):
    """
    Build Lambda-independent base matrices for Ohm solver.

    Returns
    -------
    tuple[sparse.csr_matrix, sparse.csr_matrix]
        (A_1_base, A_2_base), where:
        - A_1 = A_1_base + block_diag(diag(L), diag(L))
        - A_2 = A_2_base + diag(L)
    """
    Ix = sparse.eye(Nx, format="csr")
    Iy = sparse.eye(Ny, format="csr")

    Dxx = _periodic_second_difference_matrix(Nx)
    Dyy = _periodic_second_difference_matrix(Ny)
    Dx1 = _periodic_first_difference_matrix(Nx)
    Dy1 = _periodic_first_difference_matrix(Ny)

    Lx = sparse.kron(Iy, Dxx, format="csr")
    Ly = sparse.kron(Dyy, Ix, format="csr")
    Cxy = sparse.kron(Dy1, Dx1, format="csr")

    Axx_base = (-c2_dx2) * Ly
    Ayy_base = (-c2_dx2) * Lx
    Axy_base = c2_dx4 * Cxy

    base1 = sparse.bmat([[Axx_base, Axy_base], [Axy_base, Ayy_base]], format="csr")
    base2 = (-c2_dx2) * (Lx + Ly)

    return base1, base2


def build_ohm_bases_from_grid(Nx, Ny, c, delta):
    """Build Lambda-independent base matrices from physical grid parameters."""
    c2 = c * c
    c2_dx2 = c2 / (delta * delta)
    c2_dx4 = c2 / (4.0 * delta * delta)
    return build_ohm_bases(Nx, Ny, c2_dx2, c2_dx4)


def assemble_matrix_1(Nx, Ny, L, c2_dx2, c2_dx4, base=None):
    """
    Assemble sparse matrix for system 1 (Ex, Ey coupled).

    Builds a 2N × 2N block matrix where N = Nx * Ny:
    - Axx: Ex y-second-difference + Λ
    - Ayy: Ey x-second-difference + Λ
    - Axy, Ayx: mixed derivative terms (diagonal neighbors)

    The discretization follows wavetool.md equations (2a)-(2b).
    Uses periodic boundary conditions in both x and y directions.

    Parameters
    ----------
    Nx : int
        Number of grid points in x
    Ny : int
        Number of grid points in y
    L : ndarray
        Lambda array of shape (Ny, Nx), may be spatially varying
    c2_dx2 : float
        c²/Δ²
    c2_dx4 : float
        c²/(4Δ²)
    base : sparse.csr_matrix, optional
        Precomputed Lambda-independent base matrix for system 1.

    Returns
    -------
    sparse.csr_matrix
        Sparse matrix of shape (2*N, 2*N)
    """
    if base is None:
        base, _ = build_ohm_bases(Nx, Ny, c2_dx2, c2_dx4)
    else:
        expected_shape = (2 * Nx * Ny, 2 * Nx * Ny)
        if base.shape != expected_shape:
            raise ValueError(
                f"base shape {base.shape} must be {expected_shape} for Nx={Nx}, Ny={Ny}"
            )
    L_flat = L.flatten(order="C")
    L_block = sparse.diags(np.concatenate((L_flat, L_flat)), format="csr")
    return base + L_block


def assemble_matrix_2(Nx, Ny, L, c2_dx2, base=None):
    """
    Assemble sparse matrix for system 2 (Ez decoupled).

    From wavetool.md, equation (2c):
    (∇×∇×E)_z ≈ -(c²/Δ²)(Ez_{i+1,j} - 2Ez_{i,j} + Ez_{i-1,j})
                 -(c²/Δ²)(Ez_{i,j+1} - 2Ez_{i,j} + Ez_{i,j-1})

    Uses periodic boundary conditions in both x and y directions.

    Parameters
    ----------
    Nx : int
        Number of grid points in x
    Ny : int
        Number of grid points in y
    L : ndarray
        Lambda array of shape (Ny, Nx)
    c2_dx2 : float
        c²/Δ²
    base : sparse.csr_matrix, optional
        Precomputed Lambda-independent base matrix for system 2.

    Returns
    -------
    sparse.csr_matrix
        Sparse matrix of shape (N, N)
    """
    if base is None:
        c2_dx4 = 0.25 * c2_dx2
        _, base = build_ohm_bases(Nx, Ny, c2_dx2, c2_dx4)
    else:
        expected_shape = (Nx * Ny, Nx * Ny)
        if base.shape != expected_shape:
            raise ValueError(
                f"base shape {base.shape} must be {expected_shape} for Nx={Nx}, Ny={Ny}"
            )
    L_flat = L.flatten(order="C")
    L_diag = sparse.diags(L_flat, format="csr")
    return base + L_diag


def solve_ohm_2d(L, S, c, delta, base1=None, base2=None):
    """
    Solve the 2D generalized Ohm's law.

    Solves: (Λ + c²∇×∇×)E = S
    Uses periodic boundary conditions in both x and y directions.

    Parameters
    ----------
    L : ndarray
        Lambda array of shape (Ny, Nx)
    S : ndarray
        Source term of shape (Ny, Nx, 3)
    c : float
        Speed of light
    delta : float
        Grid spacing Δ (assumes dx = dy = Δ)
    base1 : sparse.csr_matrix, optional
        Precomputed Lambda-independent base matrix for system 1.
    base2 : sparse.csr_matrix, optional
        Precomputed Lambda-independent base matrix for system 2.

    Returns
    -------
    ndarray
        Electric field E of shape (Ny, Nx, 3)

    Notes
    -----
    The discretization follows wavetool.md:
    - Equations (2a)-(2c) for (∇×∇×E) finite-difference approximation
    """
    if L.ndim != 2:
        raise ValueError(f"L must be 2D with shape (Ny, Nx), got {L.shape}")
    if S.ndim != 3 or S.shape[-1] != 3:
        raise ValueError(f"S must have shape (Ny, Nx, 3), got {S.shape}")

    Ny, Nx = L.shape
    if S.shape[:2] != (Ny, Nx):
        raise ValueError(f"S spatial shape {S.shape[:2]} must match L shape {(Ny, Nx)}")

    c2 = c * c
    c2_dx2 = c2 / (delta * delta)
    c2_dx4 = c2 / (4.0 * delta * delta)

    if base1 is None or base2 is None:
        built_base1, built_base2 = build_ohm_bases(Nx, Ny, c2_dx2, c2_dx4)
        if base1 is None:
            base1 = built_base1
        if base2 is None:
            base2 = built_base2

    expected_shape_1 = (2 * Nx * Ny, 2 * Nx * Ny)
    expected_shape_2 = (Nx * Ny, Nx * Ny)
    if base1.shape != expected_shape_1:
        raise ValueError(
            f"base1 shape {base1.shape} must be {expected_shape_1} for Nx={Nx}, Ny={Ny}"
        )
    if base2.shape != expected_shape_2:
        raise ValueError(
            f"base2 shape {base2.shape} must be {expected_shape_2} for Nx={Nx}, Ny={Ny}"
        )

    A_1 = assemble_matrix_1(Nx, Ny, L, c2_dx2, c2_dx4, base=base1)
    A_2 = assemble_matrix_2(Nx, Ny, L, c2_dx2, base=base2)

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
