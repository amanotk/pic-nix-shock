"""Tests for shock.ohm module - Step 1: Ez only (decoupled)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from shock import ohm


def test_idx_flattening():
    """Test the idx() flattening function."""
    Nx, Ny = 8, 16
    assert ohm.idx(0, 0, Nx) == 0
    assert ohm.idx(1, 0, Nx) == 1
    assert ohm.idx(0, 1, Nx) == Nx
    assert ohm.idx(Nx - 1, Ny - 1, Nx) == Nx * Ny - 1


class TestEzFourierVerification:
    """
    Test periodic boundary conditions using discrete Fourier verification for Ez only.

    Ez is decoupled from Ex, Ey. From wavetool.md:
    Λ + 4(c²/Δ²)(sin²(kxΔ/2) + sin²(kyΔ/2))

    Uses ANALYTIC source term, not apply_ohm_operator.
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
        Lambda = np.full((Nx, Ny), Lambda_val)
        c = 1.0
        c2 = c * c

        Ez0 = 0.8
        x = np.arange(Nx) * delta
        y = np.arange(Ny) * delta
        xx, yy = np.meshgrid(x, y, indexing="ij")

        E_true = np.zeros((3, Nx, Ny))
        E_true[2] = Ez0 * np.cos(kx * xx + ky * yy)

        eigenvalue = Lambda_val + 4 * c2 / (delta * delta) * (
            np.sin(kx * delta / 2) ** 2 + np.sin(ky * delta / 2) ** 2
        )

        S = np.zeros((3, Nx, Ny))
        S[2] = eigenvalue * E_true[2]

        E_solved = ohm.solve_ohm_2d(Lambda, S, c, delta, solver_opts={"method": "direct"})

        rel_err_Ez = np.max(np.abs(E_solved[2] - E_true[2])) / np.max(np.abs(E_true[2]))

        assert rel_err_Ez < 1e-10, f"Ez relative error {rel_err_Ez:.2e} exceeds tolerance"


class TestEzMatrix:
    """Test Ez matrix assembly directly."""

    def test_ez_matrix_diagonal(self):
        """Test that Ez matrix has correct diagonal entries."""
        Nx, Ny = 4, 4
        Lambda = np.ones((Nx, Ny))
        c = 1.0
        delta = 1.0
        c2_dx2 = c * c / (delta * delta)

        A_ez = ohm.assemble_ez_matrix(Nx, Ny, Lambda, c2_dx2)

        diag = A_ez.diagonal()
        expected_diag = 1.0 + 4 * c2_dx2

        assert_allclose(diag, expected_diag)

    def test_ez_matrix_off_diagonal(self):
        """Test Ez matrix off-diagonal entries (Laplacian stencil)."""
        Nx, Ny = 4, 4
        Lambda = np.ones((Nx, Ny))
        c = 1.0
        delta = 1.0
        c2_dx2 = c * c / (delta * delta)

        A_ez = ohm.assemble_ez_matrix(Nx, Ny, Lambda, c2_dx2)

        assert A_ez[0, 1] == -c2_dx2
        assert A_ez[0, Nx] == -c2_dx2


class TestExEyMatrixFourierVerification:
    """
    Test assemble_ex_ey_matrix using analytic Fourier verification.

    From wavetool.md:
    [ Λ + 4(c²/Δ²) sin²(kyΔ/2),   -c²/Δ² sin(kxΔ) sin(kyΔ) ]
    [ -c²/Δ² sin(kxΔ) sin(kyΔ),   Λ + 4(c²/Δ²) sin²(kxΔ/2) ]
    """

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
        Lambda = np.full((Nx, Ny), Lambda_val)
        c = 1.0
        c2 = c * c
        c2_dx2 = c2 / (delta * delta)
        c2_dx4 = c2 / (4 * delta * delta)

        Ex0, Ey0 = 1.0, 0.5
        x = np.arange(Nx) * delta
        y = np.arange(Ny) * delta
        xx, yy = np.meshgrid(x, y, indexing="ij")

        E_true = np.zeros((3, Nx, Ny))
        E_true[0] = Ex0 * np.cos(kx * xx + ky * yy)
        E_true[1] = Ey0 * np.cos(kx * xx + ky * yy)

        A_xx = Lambda_val + 4 * c2_dx2 * np.sin(ky * delta / 2) ** 2
        A_yy = Lambda_val + 4 * c2_dx2 * np.sin(kx * delta / 2) ** 2
        A_xy = -c2_dx2 * np.sin(kx * delta) * np.sin(ky * delta)

        S = np.zeros((3, Nx, Ny))
        S[0] = A_xx * E_true[0] + A_xy * E_true[1]
        S[1] = A_xy * E_true[0] + A_yy * E_true[1]

        E_solved = ohm.solve_ohm_2d(Lambda, S, c, delta, solver_opts={"method": "direct"})

        rel_err_Ex = np.max(np.abs(E_solved[0] - E_true[0])) / np.max(np.abs(E_true[0]))
        rel_err_Ey = np.max(np.abs(E_solved[1] - E_true[1])) / np.max(np.abs(E_true[1]))

        assert rel_err_Ex < 1e-10, f"Ex relative error {rel_err_Ex:.2e} exceeds tolerance"
        assert rel_err_Ey < 1e-10, f"Ey relative error {rel_err_Ey:.2e} exceeds tolerance"


class TestExOnlyFourierVerification:
    """
    Test Ex component only (Ey=0).
    From wavetool.md: Ex equation has y-derivative + coupling to Ey
    """

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
        Lambda = np.full((Nx, Ny), Lambda_val)
        c = 1.0

        Ex0 = 1.0
        x = np.arange(Nx) * delta
        y = np.arange(Ny) * delta
        xx, yy = np.meshgrid(x, y, indexing="ij")

        E_true = np.zeros((3, Nx, Ny))
        E_true[0] = Ex0 * np.cos(kx * xx + ky * yy)
        E_true[1] = 0.0

        c2_dx2 = c * c / (delta * delta)
        c2_dx4 = c * c / (4.0 * delta * delta)
        A_ex_ey = ohm.assemble_ex_ey_matrix(Nx, Ny, Lambda, c2_dx2, c2_dx4)
        A_ez = ohm.assemble_ez_matrix(Nx, Ny, Lambda, c2_dx2)

        E_flat = np.concatenate([E_true[0].flatten(order="F"), E_true[1].flatten(order="F")])
        S_ex_ey = A_ex_ey @ E_flat
        S_ez = A_ez @ E_true[2].flatten(order="F")

        S = np.zeros((3, Nx, Ny))
        S[0] = S_ex_ey[: Nx * Ny].reshape((Nx, Ny), order="F")
        S[1] = S_ex_ey[Nx * Ny :].reshape((Nx, Ny), order="F")
        S[2] = S_ez.reshape((Nx, Ny), order="F")

        E_solved = ohm.solve_ohm_2d(Lambda, S, c, delta, solver_opts={"method": "direct"})

        rel_err_Ex = np.max(np.abs(E_solved[0] - E_true[0])) / np.max(np.abs(E_true[0]))
        rel_err_Ey = np.max(np.abs(E_solved[1] - E_true[1]))

        assert rel_err_Ex < 1e-10, f"Ex relative error {rel_err_Ex:.2e} exceeds tolerance"
        assert rel_err_Ey < 1e-10, f"Ey should be zero, got {rel_err_Ey:.2e}"


class TestEyOnlyFourierVerification:
    """
    Test Ey component only (Ex=0).
    From wavetool.md: Ey equation has x-derivative + coupling to Ex
    """

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
        Lambda = np.full((Nx, Ny), Lambda_val)
        c = 1.0

        Ey0 = 0.5
        x = np.arange(Nx) * delta
        y = np.arange(Ny) * delta
        xx, yy = np.meshgrid(x, y, indexing="ij")

        E_true = np.zeros((3, Nx, Ny))
        E_true[0] = 0.0
        E_true[1] = Ey0 * np.cos(kx * xx + ky * yy)

        c2_dx2 = c * c / (delta * delta)
        c2_dx4 = c * c / (4.0 * delta * delta)
        A_ex_ey = ohm.assemble_ex_ey_matrix(Nx, Ny, Lambda, c2_dx2, c2_dx4)
        A_ez = ohm.assemble_ez_matrix(Nx, Ny, Lambda, c2_dx2)

        E_flat = np.concatenate([E_true[0].flatten(order="F"), E_true[1].flatten(order="F")])
        S_ex_ey = A_ex_ey @ E_flat
        S_ez = A_ez @ E_true[2].flatten(order="F")

        S = np.zeros((3, Nx, Ny))
        S[0] = S_ex_ey[: Nx * Ny].reshape((Nx, Ny), order="F")
        S[1] = S_ex_ey[Nx * Ny :].reshape((Nx, Ny), order="F")
        S[2] = S_ez.reshape((Nx, Ny), order="F")

        E_solved = ohm.solve_ohm_2d(Lambda, S, c, delta, solver_opts={"method": "direct"})

        rel_err_Ex = np.max(np.abs(E_solved[0] - E_true[0]))
        rel_err_Ey = np.max(np.abs(E_solved[1] - E_true[1])) / np.max(np.abs(E_true[1]))

        assert rel_err_Ex < 1e-10, f"Ex should be zero, got {rel_err_Ex:.2e}"
        assert rel_err_Ey < 1e-10, f"Ey relative error {rel_err_Ey:.2e} exceeds tolerance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
