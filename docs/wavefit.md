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

### Number of Free Parameters

If we assume that the window function is fixed (i.e., $x_0$, $y_0$, $\sigma$ are given), the free parameters are as follows.
- The general localized wave model (14 parameters): $k_x$, $k_y$, $f_i (i=1,\ldots,6)$, $\phi_i (i=1,\ldots,6)$.
- The circularly polarized wave model (6 parameters): $k_x$, $k_y$, $E_w$, $B_w$, $\phi_E$, $\phi_B$.

