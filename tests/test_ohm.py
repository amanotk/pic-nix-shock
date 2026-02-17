"""Tests for shock.ohm module - Step 1: Ez only (decoupled)."""

import numpy as np
import pyamg
import pytest
from scipy import sparse

from shock import ohm


def test_idx_flattening():
    """Test the idx() flattening function."""
    Nx, Ny = 8, 16
    assert ohm.idx(0, 0, Nx) == 0
    assert ohm.idx(1, 0, Nx) == 1
    assert ohm.idx(0, 1, Nx) == Nx
    assert ohm.idx(Nx - 1, Ny - 1, Nx) == Nx * Ny - 1


def test_solve_ohm_2d_shape_validation():
    """Test solve_ohm_2d rejects invalid input shapes."""
    L = np.ones((8, 8))
    c = 1.0
    delta = 1.0

    with pytest.raises(ValueError, match="L must be 2D"):
        ohm.solve_ohm_2d(np.ones((8, 8, 1)), np.zeros((8, 8, 3)), c, delta)

    with pytest.raises(ValueError, match=r"S must have shape \(Ny, Nx, 3\)"):
        ohm.solve_ohm_2d(L, np.zeros((3, 8, 8)), c, delta)

    with pytest.raises(ValueError, match="must match L shape"):
        ohm.solve_ohm_2d(L, np.zeros((7, 8, 3)), c, delta)


def test_solve_ohm_2d_bc_x_validation():
    """Test solve_ohm_2d rejects unsupported x-boundary names."""
    L = np.ones((8, 8))
    S = np.zeros((8, 8, 3))
    with pytest.raises(ValueError, match="bc_x must be one of"):
        ohm.solve_ohm_2d(L, S, c=1.0, delta=1.0, bc_x="invalid")


def test_solve_ohm_2d_with_precomputed_bases_matches_default():
    """Test optional base-matrix injection path matches default solver path."""
    Nx, Ny = 8, 8
    c = 1.0
    delta = 1.0

    x = np.arange(Nx)
    y = np.arange(Ny)
    xx = np.broadcast_to(x, (Ny, Nx))
    yy = np.broadcast_to(y[:, None], (Ny, Nx))
    L = 0.5 + 0.1 * np.cos(2 * np.pi * xx / Nx) + 0.05 * np.sin(2 * np.pi * yy / Ny)

    rng = np.random.default_rng(123)
    S = rng.normal(size=(Ny, Nx, 3))

    base1, base2 = ohm.build_ohm_bases_from_grid(Nx, Ny, c, delta)

    E_default = ohm.solve_ohm_2d(L, S, c, delta)
    E_with_bases = ohm.solve_ohm_2d(L, S, c, delta, base1=base1, base2=base2)

    np.testing.assert_allclose(E_with_bases, E_default, rtol=1e-11, atol=1e-12)


def test_solve_ohm_2d_base_shape_validation():
    """Test solve_ohm_2d rejects invalid precomputed base-matrix shapes."""
    Nx, Ny = 8, 8
    c = 1.0
    delta = 1.0
    N = Nx * Ny

    L = np.ones((Ny, Nx))
    S = np.zeros((Ny, Nx, 3))

    bad_base1 = sparse.eye(2 * N - 1, format="csr")
    with pytest.raises(ValueError, match="base1 shape"):
        ohm.solve_ohm_2d(L, S, c, delta, base1=bad_base1)

    bad_base2 = sparse.eye(N - 1, format="csr")
    with pytest.raises(ValueError, match="base2 shape"):
        ohm.solve_ohm_2d(L, S, c, delta, base2=bad_base2)


def test_solve_ohm_2d_with_external_preconditioners_matches_default():
    """Test externally built AMG preconditioners preserve solution."""
    Nx, Ny = 8, 8
    c = 1.0
    delta = 1.0

    x = np.arange(Nx)
    y = np.arange(Ny)
    xx = np.broadcast_to(x, (Ny, Nx))
    yy = np.broadcast_to(y[:, None], (Ny, Nx))
    L = 0.5 + 0.1 * np.cos(2 * np.pi * xx / Nx) + 0.05 * np.sin(2 * np.pi * yy / Ny)

    rng = np.random.default_rng(321)
    S = rng.normal(size=(Ny, Nx, 3))

    c2_dx2 = c * c / (delta * delta)
    c2_dx4 = c * c / (4.0 * delta * delta)
    base1, base2 = ohm.build_ohm_bases(Nx, Ny, c2_dx2, c2_dx4)
    A_1 = ohm.assemble_matrix_1(Nx, Ny, L, c2_dx2, c2_dx4, base=base1)
    A_2 = ohm.assemble_matrix_2(Nx, Ny, L, c2_dx2, base=base2)
    M1 = pyamg.smoothed_aggregation_solver(A_1).aspreconditioner(cycle="V")
    M2 = pyamg.smoothed_aggregation_solver(A_2).aspreconditioner(cycle="V")

    E_ref = ohm.solve_ohm_2d(L, S, c, delta)
    E_ext = ohm.solve_ohm_2d(L, S, c, delta, M1=M1, M2=M2)

    np.testing.assert_allclose(E_ext, E_ref, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("bc_x", ["periodic", "neumann"])
def test_assembled_matrices_are_symmetric(bc_x):
    """A1 and A2 should be symmetric for supported x-boundary modes."""
    Nx, Ny = 9, 7
    c = 1.0
    delta = 1.0
    c2_dx2 = c * c / (delta * delta)
    c2_dx4 = c * c / (4.0 * delta * delta)

    x = np.arange(Nx)
    y = np.arange(Ny)
    xx = np.broadcast_to(x, (Ny, Nx))
    yy = np.broadcast_to(y[:, None], (Ny, Nx))
    L = 1.0 + 0.1 * np.cos(2 * np.pi * xx / Nx) + 0.05 * np.sin(2 * np.pi * yy / Ny)

    base1, base2 = ohm.build_ohm_bases(Nx, Ny, c2_dx2, c2_dx4, bc_x=bc_x)
    A_1 = ohm.assemble_matrix_1(Nx, Ny, L, c2_dx2, c2_dx4, base=base1, bc_x=bc_x)
    A_2 = ohm.assemble_matrix_2(Nx, Ny, L, c2_dx2, base=base2, bc_x=bc_x)

    asym1 = (A_1 - A_1.T).tocoo()
    asym2 = (A_2 - A_2.T).tocoo()
    if asym1.nnz:
        assert np.max(np.abs(asym1.data)) < 1e-14
    if asym2.nnz:
        assert np.max(np.abs(asym2.data)) < 1e-14


@pytest.mark.parametrize("Nx,Ny", [(8, 8), (12, 16), (16, 12)])
@pytest.mark.parametrize("mx,my", [(1, 1), (2, 3), (3, 2)])
def test_ez_fourier_neumann_x_periodic_y(Nx, Ny, mx, my):
    """Test Ez solver against Neumann-x / periodic-y Fourier formula."""
    delta = 1.0
    c = 1.0
    c2 = c * c

    L_val = 0.5
    L = np.full((Ny, Nx), L_val)

    x = (np.arange(Nx) + 0.5) * delta
    y = np.arange(Ny) * delta
    xx = np.broadcast_to(x, (Ny, Nx))
    yy = np.broadcast_to(y[:, None], (Ny, Nx))

    kx = np.pi * mx / (Nx * delta)
    ky = 2.0 * np.pi * my / (Ny * delta)

    Ez0 = 0.8
    E_true = np.zeros((Ny, Nx, 3))
    E_true[..., 2] = Ez0 * np.cos(kx * xx) * np.cos(ky * yy)

    eigenvalue = L_val + 4.0 * c2 / (delta * delta) * (
        np.sin(np.pi * mx / (2.0 * Nx)) ** 2 + np.sin(np.pi * my / Ny) ** 2
    )

    S = np.zeros((Ny, Nx, 3))
    S[..., 2] = eigenvalue * E_true[..., 2]

    E_solved = ohm.solve_ohm_2d(L, S, c, delta, bc_x="neumann")

    rel_err_Ez = np.max(np.abs(E_solved[..., 2] - E_true[..., 2])) / np.max(np.abs(E_true[..., 2]))
    assert rel_err_Ez < 1e-10, f"Ez relative error {rel_err_Ez:.2e} exceeds tolerance"


class TestFourierVerification:
    """
    Test periodic boundary conditions using discrete Fourier verification for Ez only.

    Ez is decoupled from Ex, Ey. From wavetool.md:
    Λ + 4(c²/Δ²)(sin²(kxΔ/2) + sin²(kyΔ/2))

    Uses analytic source term, not apply_ohm_operator.
    """

    @pytest.mark.parametrize("Nx,Ny", [(8, 8), (12, 16), (16, 12)])
    @pytest.mark.parametrize("mx,my", [(1, 1), (2, 3), (3, 2)])
    def test_ez_fourier_periodic(self, Nx, Ny, mx, my):
        """Test Ez solver against Fourier verification formula."""
        delta = 1.0
        Lx = Nx * delta
        Ly = Ny * delta

        kx = 2 * np.pi * mx / Lx
        ky = 2 * np.pi * my / Ly

        L_val = 0.5
        L = np.full((Ny, Nx), L_val)
        c = 1.0
        c2 = c * c

        Ez0 = 0.8
        x = np.arange(Nx) * delta
        y = np.arange(Ny) * delta
        xx = np.broadcast_to(x, (Ny, Nx))
        yy = np.broadcast_to(y[:, None], (Ny, Nx))

        E_true = np.zeros((Ny, Nx, 3))
        E_true[..., 2] = Ez0 * np.cos(kx * xx + ky * yy)

        eigenvalue = L_val + 4 * c2 / (delta * delta) * (
            np.sin(kx * delta / 2) ** 2 + np.sin(ky * delta / 2) ** 2
        )

        S = np.zeros((Ny, Nx, 3))
        S[..., 2] = eigenvalue * E_true[..., 2]

        E_solved = ohm.solve_ohm_2d(L, S, c, delta, bc_x="periodic")

        rel_err_Ez = np.max(np.abs(E_solved[..., 2] - E_true[..., 2])) / np.max(
            np.abs(E_true[..., 2])
        )

        assert rel_err_Ez < 1e-10, f"Ez relative error {rel_err_Ez:.2e} exceeds tolerance"

    @pytest.mark.parametrize("Nx,Ny", [(8, 8), (12, 16), (16, 12)])
    @pytest.mark.parametrize("mx,my", [(1, 1), (2, 3), (3, 2)])
    def test_ex_ey_matrix_fourier_periodic(self, Nx, Ny, mx, my):
        """Test Ex-Ey matrix solver against Fourier verification formula."""
        delta = 1.0
        Lx = Nx * delta
        Ly = Ny * delta

        kx = 2 * np.pi * mx / Lx
        ky = 2 * np.pi * my / Ly

        L_val = 0.5
        L = np.full((Ny, Nx), L_val)
        c = 1.0
        c2 = c * c
        c2_dx2 = c2 / (delta * delta)
        c2_dx4 = c2 / (4 * delta * delta)

        Ex0, Ey0 = 1.0, 0.5
        x = np.arange(Nx) * delta
        y = np.arange(Ny) * delta
        xx = np.broadcast_to(x, (Ny, Nx))
        yy = np.broadcast_to(y[:, None], (Ny, Nx))

        E_true = np.zeros((Ny, Nx, 3))
        E_true[..., 0] = Ex0 * np.cos(kx * xx + ky * yy)
        E_true[..., 1] = Ey0 * np.cos(kx * xx + ky * yy)

        A_xx = L_val + 4 * c2_dx2 * np.sin(ky * delta / 2) ** 2
        A_yy = L_val + 4 * c2_dx2 * np.sin(kx * delta / 2) ** 2
        A_xy = -c2_dx2 * np.sin(kx * delta) * np.sin(ky * delta)

        S = np.zeros((Ny, Nx, 3))
        S[..., 0] = A_xx * E_true[..., 0] + A_xy * E_true[..., 1]
        S[..., 1] = A_xy * E_true[..., 0] + A_yy * E_true[..., 1]

        E_solved = ohm.solve_ohm_2d(L, S, c, delta, bc_x="periodic")

        rel_err_Ex = np.max(np.abs(E_solved[..., 0] - E_true[..., 0])) / np.max(
            np.abs(E_true[..., 0])
        )
        rel_err_Ey = np.max(np.abs(E_solved[..., 1] - E_true[..., 1])) / np.max(
            np.abs(E_true[..., 1])
        )

        assert rel_err_Ex < 1e-10, f"Ex relative error {rel_err_Ex:.2e} exceeds tolerance"
        assert rel_err_Ey < 1e-10, f"Ey relative error {rel_err_Ey:.2e} exceeds tolerance"

    @pytest.mark.parametrize("Nx,Ny", [(8, 8), (12, 16), (16, 12)])
    @pytest.mark.parametrize("mx,my", [(1, 1), (2, 3), (3, 2)])
    def test_ex_only_fourier_periodic(self, Nx, Ny, mx, my):
        """Test Ex solver with Ey=0."""
        delta = 1.0
        Lx = Nx * delta
        Ly = Ny * delta

        kx = 2 * np.pi * mx / Lx
        ky = 2 * np.pi * my / Ly

        L_val = 0.5
        L = np.full((Ny, Nx), L_val)
        c = 1.0

        Ex0 = 1.0
        x = np.arange(Nx) * delta
        y = np.arange(Ny) * delta
        xx = np.broadcast_to(x, (Ny, Nx))
        yy = np.broadcast_to(y[:, None], (Ny, Nx))

        E_true = np.zeros((Ny, Nx, 3))
        E_true[..., 0] = Ex0 * np.cos(kx * xx + ky * yy)
        E_true[..., 1] = 0.0

        c2_dx2 = c * c / (delta * delta)
        c2_dx4 = c * c / (4.0 * delta * delta)
        A_1 = ohm.assemble_matrix_1(Nx, Ny, L, c2_dx2, c2_dx4, bc_x="periodic")
        A_2 = ohm.assemble_matrix_2(Nx, Ny, L, c2_dx2, bc_x="periodic")

        E_flat = np.concatenate(
            [E_true[..., 0].flatten(order="C"), E_true[..., 1].flatten(order="C")]
        )
        S_1 = A_1 @ E_flat
        S_2 = A_2 @ E_true[..., 2].flatten(order="C")

        S = np.zeros((Ny, Nx, 3))
        S[..., 0] = S_1[: Nx * Ny].reshape((Ny, Nx), order="C")
        S[..., 1] = S_1[Nx * Ny :].reshape((Ny, Nx), order="C")
        S[..., 2] = S_2.reshape((Ny, Nx), order="C")

        E_solved = ohm.solve_ohm_2d(L, S, c, delta, bc_x="periodic")

        rel_err_Ex = np.max(np.abs(E_solved[..., 0] - E_true[..., 0])) / np.max(
            np.abs(E_true[..., 0])
        )
        rel_err_Ey = np.max(np.abs(E_solved[..., 1] - E_true[..., 1]))

        assert rel_err_Ex < 1e-10, f"Ex relative error {rel_err_Ex:.2e} exceeds tolerance"
        assert rel_err_Ey < 1e-10, f"Ey should be zero, got {rel_err_Ey:.2e}"

    @pytest.mark.parametrize("Nx,Ny", [(8, 8), (12, 16), (16, 12)])
    @pytest.mark.parametrize("mx,my", [(1, 1), (2, 3), (3, 2)])
    def test_ey_only_fourier_periodic(self, Nx, Ny, mx, my):
        """Test Ey solver with Ex=0."""
        delta = 1.0
        Lx = Nx * delta
        Ly = Ny * delta

        kx = 2 * np.pi * mx / Lx
        ky = 2 * np.pi * my / Ly

        L_val = 0.5
        L = np.full((Ny, Nx), L_val)
        c = 1.0

        Ey0 = 0.5
        x = np.arange(Nx) * delta
        y = np.arange(Ny) * delta
        xx = np.broadcast_to(x, (Ny, Nx))
        yy = np.broadcast_to(y[:, None], (Ny, Nx))

        E_true = np.zeros((Ny, Nx, 3))
        E_true[..., 0] = 0.0
        E_true[..., 1] = Ey0 * np.cos(kx * xx + ky * yy)

        c2_dx2 = c * c / (delta * delta)
        c2_dx4 = c * c / (4.0 * delta * delta)
        A_1 = ohm.assemble_matrix_1(Nx, Ny, L, c2_dx2, c2_dx4, bc_x="periodic")
        A_2 = ohm.assemble_matrix_2(Nx, Ny, L, c2_dx2, bc_x="periodic")

        E_flat = np.concatenate(
            [E_true[..., 0].flatten(order="C"), E_true[..., 1].flatten(order="C")]
        )
        S_1 = A_1 @ E_flat
        S_2 = A_2 @ E_true[..., 2].flatten(order="C")

        S = np.zeros((Ny, Nx, 3))
        S[..., 0] = S_1[: Nx * Ny].reshape((Ny, Nx), order="C")
        S[..., 1] = S_1[Nx * Ny :].reshape((Ny, Nx), order="C")
        S[..., 2] = S_2.reshape((Ny, Nx), order="C")

        E_solved = ohm.solve_ohm_2d(L, S, c, delta, bc_x="periodic")

        rel_err_Ex = np.max(np.abs(E_solved[..., 0] - E_true[..., 0]))
        rel_err_Ey = np.max(np.abs(E_solved[..., 1] - E_true[..., 1])) / np.max(
            np.abs(E_true[..., 1])
        )

        assert rel_err_Ex < 1e-10, f"Ex should be zero, got {rel_err_Ex:.2e}"
        assert rel_err_Ey < 1e-10, f"Ey relative error {rel_err_Ey:.2e} exceeds tolerance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
