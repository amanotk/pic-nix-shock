# Document for `wavetool.py`

## Ohm’s law

The generalized Ohm’s law may be written quite generally in the following form (see [1]):

```math
(\Lambda + c^2 \nabla \times \nabla \times) \mathbf{E} = -\frac{\Gamma}{c} \times \mathbf{B} + \nabla \cdot \Pi.
```

Note that use Lorentz-Heaviside units so that $4\pi$ factors are absorbed.

The quantities $\Lambda, \Gamma, \Pi$ appearing in the above equation are defined as follows:

```math
\begin{aligned}
\Lambda &= \sum_{s} \frac{q_s^2}{m_s} \int f_s d\mathbf{v}, \\
\Gamma &= \sum_{s} \frac{q_s^2}{m_s} \int \mathbf{v} f_s d\mathbf{v}, \\
\Pi &= \sum_{s} q_s \int \mathbf{v} \mathbf{v} f_s d\mathbf{v}.
\end{aligned}
```

In `PIC-NIX` (see `pic/engine/moment.hpp`), each species stores moments

```math
M_{s, i} = m_s \int w_i f_s d\mathbf{v}
```

with the following weights $w_i$:

- $w_0 = 1$
- $w_1 = u_x/\gamma = v_x$
- $w_2 = u_y/\gamma = v_y$
- $w_3 = u_z/\gamma = v_z$
- $w_4 = \gamma c \approx c$
- $w_5 = u_x \approx v_x$
- $w_6 = u_y \approx v_y$
- $w_7 = u_z \approx v_z$
- $w_8 = u_x^2/\gamma \approx v_x^2$
- $w_9 = u_y^2/\gamma \approx v_y^2$
- $w_{10} = u_z^2/\gamma \approx v_z^2$
- $w_{11} = u_x u_y/\gamma \approx v_x v_y$
- $w_{12} = u_y u_z/\gamma \approx v_y v_z$
- $w_{13} = u_z u_x/\gamma \approx v_z v_x$

where the approximation is valid in the non-relativistic limit.

In `wavetool`, we adopt the non-relativistic approximation and store the following transformed moments in the output:
```math
\begin{aligned}
\tilde{M}_i &= \sum_{s} \left( \frac{q_s}{m_s} \right)^2 M_{s, i} \quad (i = 0, \ldots, 3), \\
\tilde{M}_i &= \sum_{s} \left( \frac{q_s}{m_s} \right) M_{s, i + 4} \quad (i = 4, \ldots, 9).
\end{aligned}
```
Clearly, these transformed moments are related to $\Lambda, \Gamma, \Pi$ as follows:
```math
\begin{aligned}
\Lambda &= \tilde{M}_0, \\
\Gamma &= \begin{pmatrix} \tilde{M}_1 \\ \tilde{M}_2 \\ \tilde{M}_3 \end{pmatrix}, \\
\Pi &=
\begin{pmatrix}
\tilde{M}_4 & \tilde{M}_7 & \tilde{M}_9 \\
\tilde{M}_7 & \tilde{M}_5 & \tilde{M}_8 \\
\tilde{M}_9 & \tilde{M}_8 & \tilde{M}_6
\end{pmatrix}.
\end{aligned}
```
Substituting the above expressions into the generalized Ohm’s law, we can obtain the electric field.

## Numerical Implementation
### Finite Difference Approximation
The generalized Ohm's law is a second-order partial differential equation, with the spatial derivative term arises from the finite electron inertia effect. When the derivative is approximated by finite difference, the electric field can be obtained numerically by inverting the corresponding matrix.

Here we only consider the 2D case ($\partial/\partial z = 0$) for simplicity. Using the formula:
```math
\nabla \times \nabla \times \mathbf{E} = - \nabla^2 \mathbf{E} + \nabla \left( \nabla \cdot \mathbf{E} \right),
```
and using the second-order central finite difference assuming the equal grid size $\Delta$ both in $x$ and $y$ directions, we have the following approximations:
```math
\begin{aligned}
& \left( \nabla \times \nabla \times \mathbf{E} \right)_x \approx
-\frac{c^2}{\Delta^2} \left( E^x_{i,j+1} - 2 E^x_{i,j} + E^x_{i,j-1} \right)
+\frac{c^2}{4 \Delta^2} \left( E^y_{i+1,j+1} - E^y_{i+1,j-1} - E^y_{i-1,j+1} + E^y_{i-1,j-1} \right)
\\
& \left( \nabla \times \nabla \times \mathbf{E} \right)_y \approx
-\frac{c^2}{\Delta^2} \left( E^y_{i+1,j} - 2 E^y_{i,j} + E^y_{i-1,j} \right)
+\frac{c^2}{4 \Delta^2} \left( E^x_{i+1,j+1} - E^x_{i+1,j-1} - E^x_{i-1,j+1} + E^x_{i-1,j-1} \right)
\\
& \left( \nabla \times \nabla \times \mathbf{E} \right)_z \approx
-\frac{c^2}{\Delta^2} \left( E^z_{i+1,j} - 2 E^z_{i,j} + E^z_{i-1,j} \right)
-\frac{c^2}{\Delta^2} \left( E^z_{i,j+1} - 2 E^z_{i,j} + E^z_{i,j-1} \right).
\end{aligned}
```
We note that the equations for $E^x$ and $E^y$ are coupled, whereas $E^z$ is independent of the other components.

Similarly, the source term on the right-hand side of the generalized Ohm's law may be approximated as follows:
```math
\begin{aligned}
& S^x = \left( -\frac{\Gamma}{c} \times \mathbf{B} + \nabla \cdot \Pi \right)^x \approx
-\frac{1}{c} \left( \Gamma^y B^z - \Gamma^z B^y \right)
+ \frac{1}{2 \Delta} \left( \Pi^{xx}_{i+1,j} - \Pi^{xx}_{i-1,j} + \Pi^{xy}_{i,j+1} - \Pi^{xy}_{i,j-1} \right)
\\
& S^y = \left( -\frac{\Gamma}{c} \times \mathbf{B} + \nabla \cdot \Pi \right)^y \approx
-\frac{1}{c} \left( \Gamma^z B^x - \Gamma^x B^z \right)
+ \frac{1}{2 \Delta} \left( \Pi^{yx}_{i+1,j} - \Pi^{yx}_{i-1,j} + \Pi^{yy}_{i,j+1} - \Pi^{yy}_{i,j-1} \right)
\\
& S^z = \left( -\frac{\Gamma}{c} \times \mathbf{B} + \nabla \cdot \Pi \right)^z \approx
-\frac{1}{c} \left( \Gamma^x B^y - \Gamma^y B^x \right) + \frac{1}{2 \Delta} \left( \Pi^{zx}_{i+1,j} - \Pi^{zx}_{i-1,j} + \Pi^{zy}_{i,j+1} - \Pi^{zy}_{i,j-1} \right).
\end{aligned}
```

### Boundary Condition
In this solver, fields are defined at cell centers. In the $x$-direction, boundaries are
located at half-grid positions, and homogeneous Neumann boundary conditions are
imposed with ghost-cell copying:
```math
E^\alpha_{-1,j} = E^\alpha_{0,j}, \qquad
E^\alpha_{N_x,j} = E^\alpha_{N_x-1,j}, \qquad
\alpha \in \{x,y,z\}.
```
This corresponds to $\partial E^\alpha / \partial x = 0$ at both $x$ boundaries.

In the $y$-direction, periodic boundary conditions are used:
```math
E^\alpha_{i,-1} = E^\alpha_{i,N_y-1}, \qquad
E^\alpha_{i,N_y} = E^\alpha_{i,0}.
```

For the coupled $(E^x, E^y)$ system, mixed-derivative coupling blocks at the
$x$-boundary rows are assembled with transpose pairing,
$A_{yx} = A_{xy}^{\mathsf T}$, to preserve global matrix symmetry.
Therefore, one mixed block is interpreted as the explicitly imposed boundary
closure and the other is its adjoint-consistent counterpart. This keeps the
discrete operator self-adjoint and compatible with CG-based solvers.

### Verification
Two verification paths are used in this repository.

#### Periodic-$x$ mode (test-only)

When both $x$ and $y$ are periodic (`bc_x="periodic"`), assuming
$A(x,y) = \tilde{A} \exp \left[ i (k_x x + k_y y) \right]$, the finite-difference
operator yields the algebraic system:
```math
\begin{aligned}
\begin{pmatrix}
\Lambda + 4 \dfrac{c^2}{\Delta^2} \sin^2 \left( \dfrac{k_y \Delta}{2} \right) &
-\dfrac{c^2}{\Delta^2} \sin \left( k_x \Delta \right) \sin \left( k_y \Delta \right) &
0 \\
-\dfrac{c^2}{\Delta^2} \sin \left( k_x \Delta \right) \sin \left( k_y \Delta \right) &
\Lambda + 4 \dfrac{c^2}{\Delta^2} \sin^2 \left( \dfrac{k_x \Delta}{2} \right) &
0 \\
0 &
0 &
\Lambda + 4 \dfrac{c^2}{\Delta^2}
\left[
    \sin^2 \left( \dfrac{k_x \Delta}{2} \right) +
    \sin^2 \left( \dfrac{k_y \Delta}{2} \right)
\right]
\end{pmatrix}
\begin{pmatrix} \tilde{E}^x \\ \tilde{E}^y \\ \tilde{E}^z \end{pmatrix}
=
\begin{pmatrix} \tilde{S}^x \\ \tilde{S}^y \\ \tilde{S}^z \end{pmatrix},
\end{aligned}
```
with constant $\Lambda$ for simplicity.

#### Neumann-$x$ mode (default runtime)

With cell-centered unknowns and Neumann closure at half-grid boundaries in $x$,
the natural $x$ eigenmodes are cosine (DCT-like) modes:
```math
\phi_m(i) = \cos\!\left(\frac{\pi m (i+1/2)}{N_x}\right),
\qquad m = 0,\ldots,N_x-1.
```
Using periodic Fourier modes in $y$ with index $n$, an $E^z$ manufactured solution
```math
E^z_{i,j} = E_0\,\phi_m(i)\cos\!\left(\frac{2\pi n j}{N_y}\right)
```
is an eigenfunction of the discrete $E^z$ operator with eigenvalue
```math
\lambda_{m,n} = \Lambda + \frac{4c^2}{\Delta^2}
\left[
\sin^2\!\left(\frac{\pi m}{2N_x}\right) +
\sin^2\!\left(\frac{\pi n}{N_y}\right)
\right].
```
Hence $S^z_{i,j} = \lambda_{m,n} E^z_{i,j}$ provides a direct verification case for
the Neumann-$x$/periodic-$y$ discretization.

For the coupled $(E^x,E^y)$ block with symmetry-preserving mixed-boundary assembly,
verification is performed numerically (matrix symmetry checks and
manufactured-solution solves) rather than with a single closed-form periodic
Fourier matrix.

## Reference

[1] Amano, T. (2018). A generalized quasi-neutral fluid-particle hybrid plasma model and its application to energetic-particle-magnetohydrodynamics hybrid simulation. *Journal of Computational Physics, 366*, 366-385. https://doi.org/10.1016/j.jcp.2018.04.020
