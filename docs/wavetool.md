# Document for `wavetool.py`

## Electric field from Ohm’s law

The generalized Ohm’s law may be written quite generally in the following form:

$$
(\Lambda + c^2 \nabla \times \nabla \times) \mathbf{E} = -\frac{\Gamma}{c} \times \mathbf{B} + \nabla \cdot \Pi.
$$

The quantities $\Lambda, \Gamma, \Pi$ appearing in the above equation are defined as follows:

$$
\begin{aligned}
& \Lambda = \sum_{s} \frac{4\pi q_s^2}{m_s} \int f_s d\mathbf{v},
\\
& \Gamma = \sum_{s} \frac{4\pi q_s^2}{m_s} \int \mathbf{v} f_s d\mathbf{v},
\\
& \Pi = \sum_{s} 4\pi q_s \int \mathbf{v} \mathbf{v} f_s d\mathbf{v}.
\end{aligned}
$$

The moment quantities stored in `PIC-NIX` output are:
$$
\begin{aligned}
& M_{s, i} = m_s \int w_i f_s d\mathbf{v}
\end{aligned}
$$
where $w_i$ are the weights defined by:
$$
\begin{aligned}
& w_0 = 1,
\\
& w_1 = v_x, \quad w_2 = v_y, \quad w_3 = v_z,
\\
& w_8 = v_x^2, \quad w_9 = v_y^2, \quad w_{10} = v_z^2,
\\
& w_{11} = v_x v_y, \quad w_{12} = v_y v_z, \quad w_{13} = v_z v_x.
\end{aligned}
$$
Note that the output moments are relativistic four velocity and stress-energy tensor, but in the non-relativistic limit, they reduce to the above forms.

To calculate the electric field using Ohm's law, we store the following transformed moments in the output:
$$
\begin{aligned}
& \tilde{M}_i = \sum_{s} \left( \frac{q_s}{m_s} \right)^2 M_{s, i} \quad (i = 0, \ldots, 3),
\\
& \tilde{M}_i = \sum_{s} \left( \frac{q_s}{m_s} \right) M_{s, i + 4} \quad (i = 4, \ldots, 9).
\end{aligned}
$$
Clearly, these transformed moments are related to $\Lambda, \Gamma, \Pi$ as follows:
$$
\begin{aligned}
& \Lambda = \tilde{M}_0,
\\
& \Gamma = \begin{pmatrix} \tilde{M}_1 \\ \tilde{M}_2 \\ \tilde{M}_3 \end{pmatrix},
\\
& \Pi =
\begin{pmatrix}
\tilde{M}_4 & \tilde{M}_7 & \tilde{M}_9 \\
\tilde{M}_7 & \tilde{M}_5 & \tilde{M}_8 \\
\tilde{M}_9 & \tilde{M}_8 & \tilde{M}_6
\end{pmatrix}.
\end{aligned}
$$
