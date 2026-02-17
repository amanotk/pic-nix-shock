"""Tests for shock.ohm module - Step 1: Ez only (decoupled)."""

import numpy as np
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
    Lambda = np.ones((8, 8))
    c = 1.0
    delta = 1.0

    with pytest.raises(ValueError, match="Lambda must be 2D"):
        ohm.solve_ohm_2d(np.ones((8, 8, 1)), np.zeros((8, 8, 3)), c, delta)

    with pytest.raises(ValueError, match="S must have shape \(Ny, Nx, 3\)"):
        ohm.solve_ohm_2d(Lambda, np.zeros((3, 8, 8)), c, delta)

    with pytest.raises(ValueError, match="must match Lambda shape"):
        ohm.solve_ohm_2d(Lambda, np.zeros((7, 8, 3)), c, delta)


def test_solve_ohm_2d_with_precomputed_bases_matches_default():
    """Test optional base-matrix injection path matches default solver path."""
    Nx, Ny = 8, 8
    c = 1.0
    delta = 1.0

    x = np.arange(Nx)
    y = np.arange(Ny)
    xx = np.broadcast_to(x, (Ny, Nx))
    yy = np.broadcast_to(y[:, None], (Ny, Nx))
    Lambda = 0.5 + 0.1 * np.cos(2 * np.pi * xx / Nx) + 0.05 * np.sin(2 * np.pi * yy / Ny)

    rng = np.random.default_rng(123)
    S = rng.normal(size=(Ny, Nx, 3))

    A_1_base, A_2_base = ohm.build_ohm_bases_from_grid(Nx, Ny, c, delta)

    E_default = ohm.solve_ohm_2d(Lambda, S, c, delta)
    E_with_bases = ohm.solve_ohm_2d(Lambda, S, c, delta, A_1_base=A_1_base, A_2_base=A_2_base)

    np.testing.assert_allclose(E_with_bases, E_default, rtol=1e-11, atol=1e-12)


def test_solve_ohm_2d_base_shape_validation():
    """Test solve_ohm_2d rejects invalid precomputed base-matrix shapes."""
    Nx, Ny = 8, 8
    c = 1.0
    delta = 1.0
    N = Nx * Ny

    Lambda = np.ones((Ny, Nx))
    S = np.zeros((Ny, Nx, 3))

    bad_A_1_base = sparse.eye(2 * N - 1, format="csr")
    with pytest.raises(ValueError, match="A_1_base shape"):
        ohm.solve_ohm_2d(Lambda, S, c, delta, A_1_base=bad_A_1_base)

    bad_A_2_base = sparse.eye(N - 1, format="csr")
    with pytest.raises(ValueError, match="A_2_base shape"):
        ohm.solve_ohm_2d(Lambda, S, c, delta, A_2_base=bad_A_2_base)


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

        Lambda_val = 0.5
        Lambda = np.full((Ny, Nx), Lambda_val)
        c = 1.0
        c2 = c * c

        Ez0 = 0.8
        x = np.arange(Nx) * delta
        y = np.arange(Ny) * delta
        xx = np.broadcast_to(x, (Ny, Nx))
        yy = np.broadcast_to(y[:, None], (Ny, Nx))

        E_true = np.zeros((Ny, Nx, 3))
        E_true[..., 2] = Ez0 * np.cos(kx * xx + ky * yy)

        eigenvalue = Lambda_val + 4 * c2 / (delta * delta) * (
            np.sin(kx * delta / 2) ** 2 + np.sin(ky * delta / 2) ** 2
        )

        S = np.zeros((Ny, Nx, 3))
        S[..., 2] = eigenvalue * E_true[..., 2]

        E_solved = ohm.solve_ohm_2d(Lambda, S, c, delta)

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

        Lambda_val = 0.5
        Lambda = np.full((Ny, Nx), Lambda_val)
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

        A_xx = Lambda_val + 4 * c2_dx2 * np.sin(ky * delta / 2) ** 2
        A_yy = Lambda_val + 4 * c2_dx2 * np.sin(kx * delta / 2) ** 2
        A_xy = -c2_dx2 * np.sin(kx * delta) * np.sin(ky * delta)

        S = np.zeros((Ny, Nx, 3))
        S[..., 0] = A_xx * E_true[..., 0] + A_xy * E_true[..., 1]
        S[..., 1] = A_xy * E_true[..., 0] + A_yy * E_true[..., 1]

        E_solved = ohm.solve_ohm_2d(Lambda, S, c, delta)

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

        Lambda_val = 0.5
        Lambda = np.full((Ny, Nx), Lambda_val)
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
        A_1 = ohm.assemble_matrix_1(Nx, Ny, Lambda, c2_dx2, c2_dx4)
        A_2 = ohm.assemble_matrix_2(Nx, Ny, Lambda, c2_dx2)

        E_flat = np.concatenate(
            [E_true[..., 0].flatten(order="C"), E_true[..., 1].flatten(order="C")]
        )
        S_1 = A_1 @ E_flat
        S_2 = A_2 @ E_true[..., 2].flatten(order="C")

        S = np.zeros((Ny, Nx, 3))
        S[..., 0] = S_1[: Nx * Ny].reshape((Ny, Nx), order="C")
        S[..., 1] = S_1[Nx * Ny :].reshape((Ny, Nx), order="C")
        S[..., 2] = S_2.reshape((Ny, Nx), order="C")

        E_solved = ohm.solve_ohm_2d(Lambda, S, c, delta)

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

        Lambda_val = 0.5
        Lambda = np.full((Ny, Nx), Lambda_val)
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
        A_1 = ohm.assemble_matrix_1(Nx, Ny, Lambda, c2_dx2, c2_dx4)
        A_2 = ohm.assemble_matrix_2(Nx, Ny, Lambda, c2_dx2)

        E_flat = np.concatenate(
            [E_true[..., 0].flatten(order="C"), E_true[..., 1].flatten(order="C")]
        )
        S_1 = A_1 @ E_flat
        S_2 = A_2 @ E_true[..., 2].flatten(order="C")

        S = np.zeros((Ny, Nx, 3))
        S[..., 0] = S_1[: Nx * Ny].reshape((Ny, Nx), order="C")
        S[..., 1] = S_1[Nx * Ny :].reshape((Ny, Nx), order="C")
        S[..., 2] = S_2.reshape((Ny, Nx), order="C")

        E_solved = ohm.solve_ohm_2d(Lambda, S, c, delta)

        rel_err_Ex = np.max(np.abs(E_solved[..., 0] - E_true[..., 0]))
        rel_err_Ey = np.max(np.abs(E_solved[..., 1] - E_true[..., 1])) / np.max(
            np.abs(E_true[..., 1])
        )

        assert rel_err_Ex < 1e-10, f"Ex should be zero, got {rel_err_Ex:.2e}"
        assert rel_err_Ey < 1e-10, f"Ey relative error {rel_err_Ey:.2e} exceeds tolerance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
