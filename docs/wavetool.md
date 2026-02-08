# Document for `wavetool.py`

## Electric field from Ohm’s law

The generalized Ohm’s law may be written quite generally in the following form:

```math
(\Lambda + c^2 \nabla \times \nabla \times) \mathbf{E} = -\frac{\Gamma}{c} \times \mathbf{B} + \nabla \cdot \Pi.
```

The quantities $\Lambda, \Gamma, \Pi$ appearing in the above equation are defined as follows:

```math
\begin{aligned}
\Lambda &= \sum_{s} \frac{4\pi q_s^2}{m_s} \int f_s d\mathbf{v}, \\
\Gamma &= \sum_{s} \frac{4\pi q_s^2}{m_s} \int \mathbf{v} f_s d\mathbf{v}, \\
\Pi &= \sum_{s} 4\pi q_s \int \mathbf{v} \mathbf{v} f_s d\mathbf{v}.
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
from which the electric field may be obtained by using the generalized Ohm’s law.