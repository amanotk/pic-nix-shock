# Document for `wavefit.py`

To identify the wave properties, one may fit the electromagnetic fluctuations with a model function.

## Model Function
### General Localized Wave Model
We here consider a monochromatic plane wave propagating in $x-y$ plane, with a window function $W(x, y; x_0, y_0)$ to represent the wave localized around the specific location centered around $(x_0, y_0)$. In general, the fluctuations may be expressed as:
```math
\begin{aligned}
F_i (x, y; x_0, y_0) = f_i \cos(k_x x + k_y y + \phi_i) W(x, y; x_0, y_0)
\end{aligned}
```
where $F_i = (E_1, E_2, E_3, B_1, B_2, B_3)$ are the electromagnetic field. The coordinate is defined such that
```math
\hat{\boldsymbol{e}}_1 = \hat{\boldsymbol{e}}_2 \times \hat{\boldsymbol{e}}_3, \quad
\hat{\boldsymbol{e}}_2 = \hat{\boldsymbol{e}}_z, \quad
\hat{\boldsymbol{e}}_3 = \boldsymbol{k} / |\boldsymbol{k}|.
```
A Gaussian window of spread $\sigma$:
```math
W(x, y; x_0, y_0) = \exp \left[ - \frac{(x - x_0)^2 + (y - y_0)^2}{2 \sigma^2} \right]
```
is used.

The electromagnetic field in the original Cartesian coordinate system may be obtained by:
```math
\begin{aligned}
\begin{pmatrix} E_x \\ E_y \end{pmatrix}
&=
\begin{pmatrix}
- \sin \theta & + \cos \theta \\
+ \cos \theta & + \sin \theta \\
\end{pmatrix}
\begin{pmatrix} E_1 \\ E_3 \end{pmatrix}
\end{aligned}
```
where $\theta = \tan^{-1} (k_y / k_x)$ is the angle between the wave vector and the $x$-axis. The same transformation applies to the magnetic field components as well.

### Circularly Polarized Electromagnetic Wave Model
The model may be further constrained by the following physical conditions:

1. Solenoidal Condition  
  The solenoidal condition for the magnetic field $\nabla \cdot \boldsymbol{B} = 0$ gives $B_3 = 0$.
1. Electromagnetic Fluctuations  
  If the fluctuation is electromagnetic, $\nabla \cdot \boldsymbol{E} = 0$, meaning that $E_3 = 0$.
1. Circular Polarization  
  For circularly polarized electromagnetic waves, in addition to the above two conditions, we have further constraints on the amplitudes and phases:
  ```math
    f_1 = f_2, \quad \phi_2 = \phi_1 - \pi/2
    \\
    f_4 = f_5, \quad \phi_5 = \phi_4 - \pi/2
  ```

If all the above conditions are applied, the model function reduces to:
```math
\begin{aligned}
& E_1 (x, y) = E_w \cos(k_x x + k_y y + \phi_E) W(x, y; x_0, y_0) \\
& E_2 (x, y) = E_w \sin(k_x x + k_y y + \phi_E) W(x, y; x_0, y_0) \\
& B_1 (x, y) = B_w \cos(k_x x + k_y y + \phi_B) W(x, y; x_0, y_0) \\
& B_2 (x, y) = B_w \sin(k_x x + k_y y + \phi_B) W(x, y; x_0, y_0) \\
& E_3(x, y) = B_3 (x, y) = 0
\end{aligned}
```
where we have introduced the notation: $E_w = f_1$ and $B_w = f_4$, $\phi_E = \phi_1$, and $\phi_B = \phi_4$.

### Helicity and Polarization Conventions

The model above defines the transverse electric field as
```math
E_1 = E_w \cos(k_x x + k_y y + \phi_E), \quad
E_2 = h \, E_w \sin(k_x x + k_y y + \phi_E),
```
where $h = \pm 1$ is the **helicity** parameter (stored as `helicity` in the fit result).

The transverse components are combined into a complex quantity
```math
E_\perp \equiv E_1 + i E_2,
```
so that:
- $h = +1$ (right-hand helicity): $E_\perp \propto \cos\phi + i\sin\phi = e^{+i\phi}$
- $h = -1$ (left-hand helicity):  $E_\perp \propto \cos\phi - i\sin\phi = e^{-i\phi}$

This is a **spatial** definition: the handedness of the transverse vector as phase increases along the propagation direction at a fixed time (a "snapshot"). This follows the convention of Terasawa et al. [1], Appendix B, where right-hand helicity corresponds to the spatial factor $e^{+ikX}$ and left-hand helicity to $e^{-ikX}$.

#### Relationship to Temporal Polarization

The phase convention used in the model is $\phi = k_x x + k_y y$ (no explicit time dependence). If a time dependence of the form $e^{-i\omega t}$ is assumed, then:

- **Right-hand helicity** ($h = +1$) with forward propagation ($k > 0$, $\omega > 0$) corresponds to **left-hand temporal polarization** at a fixed point.
- This follows from Terasawa et al. [1], who note that the right-hand helical component propagating in the $+X$ direction is observed as left-hand polarized in time (their $L^+$ component).

In other words, the sign of helicity and the sign of temporal polarization are opposite for forward-propagating waves when using the $\cos(kx-\omega t)$ phase convention.

#### Reference

[1] Terasawa, T., Hoshino, M., Sakai, J.-I., and Hada, T. (1986), "Decay instability of finite-amplitude circularly polarized Alfvén waves: A numerical simulation of stimulated Brillouin scattering," *J. Geophys. Res.*, 91(A4), 4171–4187, doi:10.1029/JA091iA04p04171. (See Appendix B for the helicity/spiral decomposition and the relation between spatial helicity and temporal polarization.)

### Frequency and Wavenumber Conventions

The wave frequency (`omega`) and wavenumber (`k`) returned by wavefit are defined such that:

- **R-mode polarization**: `omega > 0` (corresponds to `omega < 0` → L-mode)
- **Poynting flux sign**: `sign(omega / k)` corresponds to the sign of the Poynting flux

The formulas used are:
- `sign_k_dot_b = sign(kx*Bx + ky*By)`
- `sign_phi_diff = sign(phiE - phiB)` (phase difference normalized to [-π, π])
- `omega = - |k| * c * Ew/Bw * sign_k_dot_b * sign_phi_diff * helicity`
- `k = -sign_k_dot_b * sqrt(kx^2 + ky^2) * helicity`

where `c = 1` in normalized simulation units.

### Number of Free Parameters

If we assume that the window function is fixed (i.e., $x_0$, $y_0$, $\sigma$ are given), the free parameters are as follows.
- The general localized wave model (14 parameters): $k_x$, $k_y$, $f_i (i=1,\ldots,6)$, $\phi_i (i=1,\ldots,6)$.
- The circularly polarized wave model (6 parameters): $k_x$, $k_y$, $E_w$, $B_w$, $\phi_E$, $\phi_B$.

## Goodness of Fit Criteria

Current wavefit implementation evaluates goodness with two conditions:

1. Balanced error criterion:
   - `nrmse_balanced <= good_nrmse_bal_max` (default `0.4`; `0.7` is a common relaxed choice)
   - where
     ```math
     nrmse_{balanced} = \sqrt{\frac{1}{2}\left(nrmse_E^2 + nrmse_B^2\right)}
     ```
     and
     ```math
     nrmse_E = \frac{\mathrm{rms}(E_{data}-E_{model})}{\mathrm{rms}(E_{data})},
     \quad
     nrmse_B = \frac{\mathrm{rms}(B_{data}-B_{model})}{\mathrm{rms}(B_{data})}
     ```
2. Scale-separation criterion:
   - `lambda <= 4 * sigma`
   - with
     ```math
     k = \sqrt{k_x^2 + k_y^2},
     \quad
     \lambda = \frac{2\pi}{k}
     ```

The fit is marked as good only when both conditions are satisfied.

## Diagnostic Plot Function (Reusable)

`shock/wavefit/plot.py` provides:

- `save_quickcheck_plot_12panel(filename, fit_result, title=None, rms_normalize=True)`

This creates a 2x6 panel diagnostic plot for one fitted candidate:

- top row: data
- bottom row: model
- columns: `Ex, Ey, Ez, Bx, By, Bz`

When `rms_normalize=True` (recommended), electric and magnetic components are
normalized separately (`E/rmsE`, `B/rmsB`) to make visual comparison robust
when `|E| << |B|`.

### Minimal Usage

```python
from shock import wavefit

# fit_result returned by wavefit.fit_one_candidate(...)
wavefit.save_quickcheck_plot_12panel(
    "quickcheck-example.png",
    fit_result,
    title="candidate quickcheck",
    rms_normalize=True,
)
```

## Interactive Tuning

For iterative, human-in-the-loop fitting-quality sessions, use the generic
playbook in `docs/wavefit-interactive-tuning.md`.

## Envelope Map Plot Job

`python -m shock.wavefit -j plot <config.toml>` renders one envelope map PNG per
fitted snapshot from `fitfile.h5` + `wavefile.h5`.

- Input snapshots come from `fitfile` (`snapshots/<step>`).
- Overlay points include only candidates with `is_good=1`.
- Output filename pattern: `<fitfile>-envelope-<index>.png`.
- `--debug`, `--debug-count`, and `--debug-index` may be used to render a subset.

## MPI Execution Notes

- `analyze` automatically uses snapshot-level MPI parallelism when launched under
  multi-rank MPI (`mpiexec -n N ...`).
- In MPI mode, root uses `mpi4py.futures.MPICommExecutor`, workers fit snapshots,
  and root writes results to HDF5.
- `analyze --debug` remains serial by design.
- `plot` remains serial (root rank only when launched under MPI).

Preferred launcher wrapper:

```bash
scripts/mpi-wavefit.sh -n 4 -j analyze work/ma05-tbn80-run002/wavefit-config.toml
scripts/mpi-wavefit.sh -n 4 -j analyze,plot --snapshot-index 10 --snapshot-index 11 work/ma05-tbn80-run002/wavefit-config.toml
```
