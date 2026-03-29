# Solving the Spitzer Plasma Heat-Conduction Equation

In this work, the author addresses the numerical solution of the Spitzer plasma heat-conduction equation. For this purpose, a fully implicit scheme with nonlinearity evaluated at the upper time layer is implemented. The scheme is tested on the model problem of Sobol–Samarskii–Zeldovich, as well as on the nonhomogeneous formulation of the original problem. The implementation of all algorithms can be found in the [GitHub repository](https://github.com/Claptar/spitzer-conductivity.git).

## Problem statement

The Cauchy problem for the two-component plasma heat-conduction equation is given by:

$$
\frac{\partial u_1}{\partial t} = \frac{\partial}{\partial x}\kappa_1 u_1^{\alpha_1}\frac{\partial u_1}{\partial x} - q_{ei},
$$

$$
\frac{\partial u_2}{\partial t} = \frac{\partial}{\partial x}\kappa_2 u_2^{\alpha_2}\frac{\partial u_2}{\partial x} + q_{ei}.
$$

The initial conditions represent a uniformly heated plasma layer of fixed length:

$$
u_1(x, 0) = u_2(x, 0) =
\begin{cases}
T_0, x \le 1,\
0, x > 1.
\end{cases}
$$

The boundary conditions correspond to a thermally insulated wall on the left and zero at infinity:

$$
\frac{\partial u_1}{\partial x} = \frac{\partial u_2}{\partial x} = 0, \
\lim_{x \rightarrow +\inf}u_1(x,t) = \lim_{x \rightarrow +\inf}u_2(x,t) = 0.
$$

Where $q_{ei} = \frac{u_1 - u_2}{u_1^2}$, $T_0 \ge 1$, $\kappa_1 = 0.2$, $\kappa_2 = 0.3$, $\alpha_1 = 2.5$, and $\alpha_2 = 1.5$.

## Choice of scheme

As is well known, the derivative of the solution $u(x, t)$ of a quasilinear heat-conduction equation with $\alpha > 1$ becomes infinite at the front. Accordingly, when using non-monotone schemes, numerical oscillations can easily arise. Therefore, for solving such problems it is convenient to use fully implicit schemes, which are **monotone** and **stable** for any time step. We will use a scheme with nonlinearity evaluated at the upper layer:

$$
\frac{y_{m}^{n+1} - y_{m}^{n}}{\tau} = \frac{1}{h}\left[k_{m + 1/2}^{n + 1}\frac{y_{m+1}^{n+1} - y_{m}^{n+1}}{h} - k_{m - 1/2}^{n+1}\frac{y_{m}^{n+1} - y_{m - 1}^{n+1}}{h}\right] + f_m^{n + 1}
$$

$$
k_{m + 1/2}^{n + 1} = \kappa\frac{(u_{m}^{n + 1})^{\alpha} - (u_{m + 1}^{n + 1})^{\alpha}}{2}
$$

The scheme has first-order accuracy in time, $O(\tau)$, and second-order accuracy in space, $O(h^2)$. The scheme is also monotone in the sense of Friedrichs.

### Scheme implementation

Let us write the finite-difference problem:

$$
\begin{cases}
\frac{u_{m}^{\alpha, n + 1} - u_{m}^{\alpha, n}}{\tau} = \frac{1}{h}\left[k_{m + 1/2}^{n + 1}\frac{u_{m+1}^{\alpha, n+1} - u_{m}^{\alpha, n+1}}{h} - k_{m - 1/2}^{n+1}\frac{u_{m}^{\alpha, n+1} - u_{m - 1}^{\alpha, n+1}}{h}\right] - \varphi_{{\alpha,}m}^{n + 1}, m=1..M - 2, n=1..N-1 \
\frac{u_0^{\alpha, n + 1} - u_0^{\alpha, n}}{\tau} = \frac{k_{1/2}^{n + 1}}{h^2}(u_1^{\alpha, n + 1} - u_0^{\alpha, n + 1}) - \varphi_{{\alpha,}0}^{n + 1}, n=1..N-1 \
u_{\alpha}(t^{n}, M) = 0, n=1..N-1 \
\begin{equation*}
u_{\alpha}(x_m, 0) =
\begin{cases}
T_0, x \le 1,\
0, x_m > 1.
\end{cases}
\end{equation*}
\end{cases}
$$

To implement the Thomas algorithm, we linearize the system:

$$
\delta \hat u_{n+1}^{\alpha}\left[k_{n + 1/2} + \frac{\partial k_{n + 1/2}}{\partial \hat u_{n+1}^{\alpha}}(\hat u_{n+1}^{\alpha} - \hat u_{n}^{\alpha})\right] - \delta \hat u_{n}^{\alpha}\left[\frac{h^2}{\tau} + k_{n + 1/2} + k_{n - 1/2} - \frac{\partial k_{n + 1/2}}{\partial \hat u_{n}^{\alpha}}(\hat u_{n+1}^{\alpha} - \hat u_{n}^{\alpha}) + \frac{\partial k_{n - 1/2}}{\partial \hat u_{n}^{\alpha}}(\hat u_{n}^{\alpha} - \hat u_{n - 1}^{\alpha}) - \sum_ih^2\frac{\partial \varphi^\alpha}{\partial \hat u_{n}^{i}}\right] +
$$

$$
+ \delta \hat u_{n-1}^{\alpha}\left[k_{n - 1/2} - \frac{\partial k_{n - 1/2}}{\partial \hat u_{n-1}^{\alpha}}(\hat u_{n}^{\alpha} - \hat u_{n-1}^{\alpha})\right] = \frac{h^2}{\tau}(\hat u_{n}^{\alpha} - u_{n}^{\alpha}) - k_{n + 1/2}(\hat u_{n+1}^{\alpha} - \hat u_{n}^{\alpha}) + k_{n - 1/2}(\hat u_{n}^{\alpha} - \hat u_{n - 1}^{\alpha}) - h^2\varphi^\alpha_n
$$

$$
\hat{u}_n^{\alpha, (s + 1)} = \hat{u}_n^{\alpha, (s)} + \delta\hat{u}_n^{\alpha, (s)}
$$

## Scheme verification

### Solving the Sobol–Samarskii–Zeldovich problem

We will test the scheme on the model Sobol–Samarskii–Zeldovich problem:

$$
\begin{cases}
\frac{\partial u}{\partial t} = \frac{\partial}{\partial x}\kappa u^{\alpha}\frac{\partial u}{\partial x} \
u(x, 0) = 0 \
u(0, t) = ct^{1/\alpha} \
\lim_{x \rightarrow +\inf}u(x,t) = 0
\end{cases}
$$

For this problem, the analytical solution is known:

$$
\begin{equation*}
u =
\begin{cases}
\left(\frac{\alpha v}{\kappa}(vt - x)\right)^{(1/\alpha)}, \text{ } x - vt \le 0\
0, \text{ } x - vt > 0.
\end{cases}
\end{equation*}
$$

```python
import numpy as np
from tqdm.notebook import tqdm
from numba import njit
from thomas import solve_equations, solve_blocks
from make_diagonal import zeldovich_nonlinear, make_block_diagonals
import matplotlib.pyplot as plt
from matplotlib.pyplot import axes
from celluloid import Camera
```

Implementation of Newton's method:

```python
@njit
def newton_solver(y, y_left, tau, h, alpha=2.5, kappa=0.2, iter=10):
    y_n, y_s = y, y
    y_s[0] = y_left
    for i in range(iter):
        a, b, c, d = zeldovich_nonlinear(y_s, tau, h, y_n, alpha=alpha, kappa=kappa)
        dy = solve_equations(a, b, c, d)
        y_s = y_s + dy
    return y_s
```

Set the parameters and initial values:

```python
a = 0
b = 3
t_0 = 0
T = 2
To = 2
c = 3
alpha = 2.5
kappa = 0.2

print(f'>>> Wave speed v = {np.sqrt(c ** (1 / alpha) * kappa / alpha): .4f}')
```

```
>>> Wave speed v =  0.3523
```

Set the grid parameters:

```python
# Number of nodes
N = 6000
M = 100

# Grid step sizes
h = (b - a) / (M - 1)
tau = (T - t_0) / (N - 1)

print(f'>>> {h=: .8f}, {tau=: .8f}')
print(f'>>> Hyperbolic analogue of the Courant number sigma ={kappa * tau / h ** 2 * 0.5: .4f}')
```

```
>>> h= 0.03030303, tau= 0.00033339
>>> Hyperbolic analogue of the Courant number sigma = 0.0363
```

```python
t = np.linspace(t_0, T, N)
x = np.linspace(a, b, M)
```

Initialize the grid, initial conditions, and boundary conditions:

```python
u = np.zeros((N, M), dtype=np.double)
u[:, 0] = c * t ** (1 / alpha)
```

When solving the system of equations, we will perform 10 Newton iterations:

```python
for n in tqdm(range(N - 1)):
    u_sol = newton_solver(u[n, :], u[n + 1, 0], tau, h, alpha=2.5, kappa=0.2, iter=10)
    u[n + 1, 1:] = u_sol[1:]
```

The analytical solution is shown in green, and the numerical solution in red. As can be seen, the wave front of the numerical solution lags slightly behind the analytical one.

<img alt="SegmentLocal" height="400" src="lab_gifs\zeldovich_true.gif" title="segment" width="600"/>

### Solving the homogeneous problem

Let us test the implementation of the scheme on the homogeneous formulation of the problem:

$$
\frac{\partial u_1}{\partial t} = \frac{\partial}{\partial x}\kappa_1 u_1^{\alpha_1}\frac{\partial u_1}{\partial x},
$$

$$
\frac{\partial u_2}{\partial t} = \frac{\partial}{\partial x}\kappa_2 u_2^{\alpha_2}\frac{\partial u_2}{\partial x}.
$$

Implementation of Newton's method:

```python
def newton_solver(u1, u2, tau, h, alpha=(2.5, 1.5), kappa=(0.2, 0.3), iter=10):
    u1_s, u2_s = u1, u2
    for i in range(iter):
        A, B, C, D = make_block_diagonals(u1_s, u2_s, tau, h, u1, u2, alpha=alpha, kappa=kappa, nonhomogen=True)
        du = solve_blocks(A, B, C, D)
        u1_s = u1_s + du[:, 0]
        u2_s = u2_s + du[:, 1]
    return u1_s, u2_s
```

Set the parameters and initial values:

```python
# Initial values
a = 0
b = 3
t_0 = 0
T = 4
To = 2
alpha = [2.5, 1.5]
kappa = [0.2, 0.3]
```

Set the grid parameters:

```python
# Number of nodes
N = 6000
M = 100

# Grid step sizes
h = (b - a) / (M - 1)
tau = (T - t_0) / (N - 1)

print(f'>>> {h=: .8f}, {tau=: .8f}')
print(f'>>> Hyperbolic analogue of the Courant number: \n sigma_1 ={kappa[0] * tau / h ** 2 * 0.5: .4f} \n sigma_2 ={kappa[1] * tau / h ** 2 * 0.5: .4f}')
```

```
>>> h= 0.03030303, tau= 0.00066678
>>> Hyperbolic analogue of the Courant number: 
 sigma_1 = 0.0726 
 sigma_2 = 0.1089
```

```python
t = np.linspace(t_0, T, N)
x = np.linspace(a, b, M)
```

Initialize the grid, initial conditions, and boundary conditions:

```python
def u_init(x, To):
    u = np.zeros(x.size)
    for i in range(len(x)):
        if x[i] <= 1:
            u[i] = To
    return u
```

```python
u1 = np.zeros((N, M), dtype=np.double)
u1[0, :] = u_init(x, 2)
u2 = np.zeros((N, M), dtype=np.double)
u2[0, :] = u_init(x, 2)
```

When solving the system of equations, we will perform 10 Newton iterations:

```python
for n in tqdm(range(N - 1)):
    u1_sol, u2_sol = newton_solver(u1[n, :], u2[n, :], tau, h, alpha=(2.5, 1.5), kappa=(0.2, 0.3), iter=10)
    u1[n + 1, :], u2[n + 1, :] = u1_sol, u2_sol
```

The electron temperature $T_e$ is shown in blue, and the ion temperature in red. It can be seen that the ion wave moves faster, which is generally consistent with its larger heat-conductivity coefficient $\kappa$.

<img alt="SegmentLocal" height="400" src="lab_gifs\spitz_no_f_block.gif" title="segment" width="600"/>

## Solving the original problem

### Formulation

Let us recall the formulation of the original problem.

The Cauchy problem for the two-component plasma heat-conduction equation is:

$$
\frac{\partial u_1}{\partial t} = \frac{\partial}{\partial x}\kappa_1 u_1^{\alpha_1}\frac{\partial u_1}{\partial x} - q_{ei},
$$

$$
\frac{\partial u_2}{\partial t} = \frac{\partial}{\partial x}\kappa_2 u_2^{\alpha_2}\frac{\partial u_2}{\partial x} + q_{ei}.
$$

The initial conditions represent a uniformly heated plasma layer of fixed length:

$$
u_1(x, 0) = u_2(x, 0) =
\begin{cases}
T_0, x \le 1,\
0, x > 1.
\end{cases}
$$

The boundary conditions correspond to a thermally insulated wall on the left and zero at infinity:

$$
\frac{\partial u_1}{\partial x} = \frac{\partial u_2}{\partial x} = 0, \
\lim_{x \rightarrow +\inf}u_1(x,t) = \lim_{x \rightarrow +\inf}u_2(x,t) = 0.
$$

Where $q_{ei} = \frac{u_1 - u_2}{u_1^2}$, $T_0 \ge 1$, $\kappa_1 = 0.2$, $\kappa_2 = 0.3$, $\alpha_1 = 2.5$, and $\alpha_2 = 1.5$.

### Numerical solution

Implementation of Newton's method:

```python
def newton_solver(u1, u2, tau, h, alpha=(2.5, 1.5), kappa=(0.2, 0.3), iter=10):
    u1_s, u2_s = u1, u2
    for i in range(iter):
        A, B, C, D = make_block_diagonals(u1_s, u2_s, tau, h, u1, u2, alpha=alpha, kappa=kappa, nonhomogen=False)
        du = solve_blocks(A, B, C, D)
        u1_s = u1_s + du[:, 0]
        u2_s = u2_s + du[:, 1]
    return u1_s, u2_s
```

Set the parameters and initial values:

```python
# Initial values
a = 0
b = 3
t_0 = 0
T = 4
To = 2
alpha = [2.5, 1.5]
kappa = [0.2, 0.3]
```

Set the grid parameters:

```python
# Number of nodes
N = 6000
M = 100

# Grid step sizes
h = (b - a) / (M - 1)
tau = (T - t_0) / (N - 1)

print(f'>>> {h=: .8f}, {tau=: .8f}')
print(f'>>> Hyperbolic analogue of the Courant number: \n sigma_1 ={kappa[0] * tau / h ** 2 * 0.5: .4f} \n sigma_2 ={kappa[1] * tau / h ** 2 * 0.5: .4f}')
```

```
>>> h= 0.03030303, tau= 0.00066678
>>> Hyperbolic analogue of the Courant number: 
 sigma_1 = 0.0726 
 sigma_2 = 0.1089
```

```python
t = np.linspace(t_0, T, N)
x = np.linspace(a, b, M)
```

Initialize the grid, initial conditions, and boundary conditions:

```python
def u_init(x, To):
    u = np.zeros(x.size)
    for i in range(len(x)):
        if x[i] <= 1:
            u[i] = To
    return u
```

```python
u1 = np.zeros((N, M), dtype=np.double)
u1[0, :] = u_init(x, 2)
u2 = np.zeros((N, M), dtype=np.double)
u2[0, :] = u_init(x, 2)
```

```python
np.seterr(divide='raise', invalid='raise')
```

```python
{'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
```

When solving the system of equations, we will perform 2 Newton iterations:

```python
for n in tqdm(range(N - 1)):
    u1_sol, u2_sol = newton_solver(u1[n, :], u2[n, :], tau, h, alpha=(2.5, 1.5), kappa=(0.2, 0.3), iter=2)
    u1[n + 1, :], u2[n + 1, :] = u1_sol, u2_sol
```

The electron temperature $T_e$ is shown in blue, and the ion temperature $T_i$ is shown in red. Unlike the homogeneous case without heat exchange, both waves propagate at approximately the same speed.

<img alt="SegmentLocal" height="400" src="lab_gifs\spitz_with_f.gif" title="segment" width="600"/>

If you want, I can also turn this into a polished GitHub-ready `README.md` with smoother English and cleaner technical phrasing.
