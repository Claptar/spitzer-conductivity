# Решение уравнения спитцеровской теплопроводности плазмы

В данной работе автор ставит задачу численного решения уравнения спитцеровской теплопроводности плазмы. Для этого реализуется чисто неявная схема с нелинейностью на верхнем слое. Схема проверяется на решении модельной задачи Соболя-Самарского-Зельдовича, а так же на неоднородной постановке исходной задачи. Реализацию всех алгоритмов можно найти в [github-репозитории](https://github.com/Claptar/spitzer-conductivity.git).

## Постановка задачи
Задача Коши для уравнения двухкомпонентной теплороводности плазмы:

$$
\frac{\partial u_1}{\partial t} = \frac{\partial}{\partial x}\kappa_1 u_1^{\alpha_1}\frac{\partial u_1}{\partial x} - q_{ei}, \\
\frac{\partial u_2}{\partial t} = \frac{\partial}{\partial x}\kappa_2 u_2^{\alpha_2}\frac{\partial u_2}{\partial x} + q_{ei}. \\
$$

Начальные условия представляют равномерно прогретый слой плазмы фиксированной длинны:

$$
u_1(x, 0) = u_2(x, 0) =
\begin{cases}
T_0, x \le 1,\\
0, x > 1.
\end{cases}
$$

Граничные условия представляют собой теплоизолированную стенку слева и ноль на бесконечности:

$$
\frac{\partial u_1}{\partial x} = \frac{\partial u_2}{\partial x} = 0, \\
\lim_{x \rightarrow +\inf}u_1(x,t) = \lim_{x \rightarrow +\inf}u_2(x,t) = 0.
$$

Где $q_{ei} = \frac{u_1 - u_2}{u_1^2}$, $T_0 \ge 1$, $\kappa_1 = 0.2$, $\kappa_2 = 0.3$, $\alpha_1 = 2.5$, $\alpha_2 = 1.5$.

## Выбор схемы

Как известно производная решения $u(x, t)$ квазилинейного уравнения теплопроводности с $\alpha > 1$ на фронте обращается в бесконечность. Соответсвенно при расчёте по не монотонным схемам легко возникает разболтка. Поэтому для решения подобных задач удобно использовать чисто неявные схемы, которые **монотонны** и **устойчивы** при любых шагах. Будем использовать схему с нелинейностью сверху:

$$
\frac{y_{m}^{n+1} - y_{m}^{n}}{\tau} = \frac{1}{h}\left[k_{m + 1/2}^{n + 1}\frac{y_{m+1}^{n+1} - y_{m}^{n+1}}{h} - k_{m - 1/2}^{n+1}\frac{y_{m}^{n+1} - y_{m - 1}^{n+1}}{h}\right] + f_m^{n + 1}
$$

$$
k_{m + 1/2}^{n + 1} = \kappa\frac{(u_{m}^{n + 1})^{\alpha} - (u_{m + 1}^{n + 1})^{\alpha}}{2}
$$

Схема имеет первый порядок апроксимации по времени $O(\tau)$ и второй порядок апроксимации по пространству $O(h^2)$. Так же схема является монотонной про Фридрихсу.

### Реализация схемы

Запишем разностную задачу:

$$
\begin{cases}
\frac{u^{\alpha, n + 1}_{m} - u^{\alpha, n}_{m}}{\tau} = \frac{1}{h}\left[k_{m + 1/2}^{n + 1}\frac{u^{\alpha, n+1}_{m+1} - u^{\alpha, n+1}_{m}}{h} - k_{m - 1/2}^{n+1}\frac{u^{\alpha, n+1}_{m} - u^{\alpha, n+1}_{m - 1}}{h}\right] - \varphi_{{\alpha,}m}^{n + 1}, m=1..M - 2, n=1..N-1 \\
\frac{u_0^{\alpha, n + 1} - u_0^{\alpha, n}}{\tau} = \frac{k_{1/2}^{n + 1}}{h^2}(u_1^{\alpha, n + 1} - u_0^{\alpha, n + 1}) - \varphi_{{\alpha,}0}^{n + 1}, n=1..N-1 \\
u_{\alpha}(t^{n}, M) = 0, n=1..N-1 \\
\begin{equation*}
u_{\alpha}(x_m, 0) =
\begin{cases}
T_0, x \le 1,\\
0, x_m > 1.
\end{cases}
\end{equation*}
\end{cases}
$$


Для реализации метода прогонки проведём линеаризацию:

$$
\delta \hat{u}^{\alpha}_{n+1}\left[k_{n + 1/2} + \frac{\partial k_{n + 1/2}}{\partial \hat{u}^{\alpha}_{n+1}}(\hat{u}^{\alpha}_{n+1} - \hat{u}^{\alpha}_{n})\right] - \delta \hat{u}^{\alpha}_{n}\left[\frac{h^2}{\tau} + k_{n + 1/2} + k_{n - 1/2} - \frac{\partial k_{n + 1/2}}{\partial \hat{u}^{\alpha}_{n}}(\hat{u}^{\alpha}_{n+1} - \hat{u}^{\alpha}_{n}) + \frac{\partial k_{n - 1/2}}{\partial \hat{u}^{\alpha}_{n}}(\hat{u}^{\alpha}_{n} - \hat{u}^{\alpha}_{n - 1}) - \sum_ih^2\frac{\partial \varphi^\alpha}{\partial \hat{u}^{i}_{n}}\right] +
$$

$$
\+ \delta \hat{u}^{\alpha}_{n-1}\left[k_{n - 1/2} - \frac{\partial k_{n - 1/2}}{\partial \hat{u}^{\alpha}_{n-1}}(\hat{u}^{\alpha}_{n} - \hat{u}^{\alpha}_{n-1})\right] = \frac{h^2}{\tau}(\hat{u}^{\alpha}_{n} - u^{\alpha}_{n}) - k_{n + 1/2}(\hat{u}^{\alpha}_{n+1} - \hat{u}^{\alpha}_{n}) + k_{n - 1/2}(\hat{u}^{\alpha}_{n} - \hat{u}^{\alpha}_{n - 1}) - h^2\varphi^\alpha_n
$$

$$
\hat{u}_n^{\alpha, (s + 1)} = \hat{u}_n^{\alpha, (s)} + \delta\hat{u}_n^{\alpha, (s)}
$$

## Проверка схемы

### Решение задачи Соболя-Самарского-Зельдовича

Будем проверять схему на модельной задаче Соболя-Самарского-Зельдовича:

$$
\begin{cases}
\frac{\partial u}{\partial t} = \frac{\partial}{\partial x}\kappa u^{\alpha}\frac{\partial u}{\partial x} \\
u(x, 0) = 0 \\
u(0, t) = ct^{1/\alpha} \\
\lim_{x \rightarrow +\inf}u(x,t) = 0
\end{cases}
$$

Для задачи известо аналитическое решение:

$$
\begin{equation*}
u =
\begin{cases}
\left(\frac{\alpha v}{\kappa}(vt - x)\right)^{(1/\alpha)}, \text{ } x - vt \le 0\\
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

Реализация метода Ньютона


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

Зададим параметры и начальные значения


```python
a = 0
b = 3
t_0 = 0
T = 2
To = 2
c = 3
alpha = 2.5
kappa = 0.2

print(f'>>> Скорость волны v = {np.sqrt(c ** (1 / alpha) * kappa / alpha): .4f}')
```

    >>> Скорость волны v =  0.3523
    

Зададим параметры сетки


```python
# Число узлов
N = 6000
M = 100

# Размер шага сетки
h = (b - a) / (M - 1)
tau = (T - t_0) / (N - 1)

print(f'>>> {h=: .8f}, {tau=: .8f}')
print(f'>>> Гиперболический аналог числа куранта sigma ={kappa * tau / h ** 2 * 0.5: .4f}')
```

    >>> h= 0.03030303, tau= 0.00033339
    >>> Гиперболический аналог числа куранта sigma = 0.0363
    


```python
t = np.linspace(t_0, T, N)
x = np.linspace(a, b, M)
```

Инициализируем сетку, начальные и граничные условия


```python
u = np.zeros((N, M), dtype=np.double)
u[:, 0] = c * t ** (1 / alpha)
```

При решении системы уравнений будем делать 10 итераций метода Ньютона


```python
for n in tqdm(range(N - 1)):
    u_sol = newton_solver(u[n, :], u[n + 1, 0], tau, h, alpha=2.5, kappa=0.2, iter=10)
    u[n + 1, 1:] = u_sol[1:]
```


      0%|          | 0/5999 [00:00<?, ?it/s]


Аналитическое решение изображено зелёным, численное -- красным. Как можно видеть, волновой фронт численного решения несколько отстаёт от аналитического.

<img alt="SegmentLocal" height="400" src="lab_gifs\zeldovich_true.gif" title="segment" width="600"/>

### Решение однородной задачи

Реализация метода Ньютона


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

Зададим параметры и начальные значения


```python
# Начальные значения
a = 0
b = 3
t_0 = 0
T = 4
To = 2
alpha = [2.5, 1.5]
kappa = [0.2, 0.3]
```

Зададим параметры сетки


```python
# Число узлов
N = 6000
M = 100

# Размер шага сетки
h = (b - a) / (M - 1)
tau = (T - t_0) / (N - 1)

print(f'>>> {h=: .8f}, {tau=: .8f}')
print(f'>>> Гиперболический аналог числа куранта: \n sigma_1 ={kappa[0] * tau / h ** 2 * 0.5: .4f} \n sigma_2 ={kappa[1] * tau / h ** 2 * 0.5: .4f}')
```

    >>> h= 0.03030303, tau= 0.00066678
    >>> Гиперболический аналог числа куранта: 
     sigma_1 = 0.0726 
     sigma_2 = 0.1089
    


```python
t = np.linspace(t_0, T, N)
x = np.linspace(a, b, M)
```

Инициализируем сетку, начальные и граничные условия


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

При решении системы уравнений будем делать 10 итераций метода Ньютона


```python
for n in tqdm(range(N - 1)):
    u1_sol, u2_sol = newton_solver(u1[n, :], u2[n, :], tau, h, alpha=(2.5, 1.5), kappa=(0.2, 0.3), iter=10)
    u1[n + 1, :], u2[n + 1, :] = u1_sol, u2_sol
```


      0%|          | 0/5999 [00:00<?, ?it/s]


Синим обозначена электронная температура $T_e$, красным обозначена ионная температура. Можно заметить, что ионная волна двигается быстрее (что в целом соотносится с тем, что её коэффициент теплопроводности $\kappa$ больше).

<img alt="SegmentLocal" height="400" src="lab_gifs\spitz_no_f_block.gif" title="segment" width="600"/>

## Решение исходной задачи

Реализация метода Ньютона


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

Зададим параметры и начальные значения


```python
# Начальные значения
a = 0
b = 3
t_0 = 0
T = 4
To = 2
alpha = [2.5, 1.5]
kappa = [0.2, 0.3]
```

Зададим параметры сетки


```python
# Число узлов
N = 6000
M = 100

# Размер шага сетки
h = (b - a) / (M - 1)
tau = (T - t_0) / (N - 1)

print(f'>>> {h=: .8f}, {tau=: .8f}')
print(f'>>> Гиперболический аналог числа куранта: \n sigma_1 ={kappa[0] * tau / h ** 2 * 0.5: .4f} \n sigma_2 ={kappa[1] * tau / h ** 2 * 0.5: .4f}')
```

    >>> h= 0.03030303, tau= 0.00066678
    >>> Гиперболический аналог числа куранта: 
     sigma_1 = 0.0726 
     sigma_2 = 0.1089
    


```python
t = np.linspace(t_0, T, N)
x = np.linspace(a, b, M)
```

Инициализируем сетку, начальные и граничные условия


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




    {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}



При решении системы уравнений будем делать 2 итераций метода Ньютона


```python
for n in tqdm(range(N - 1)):
    u1_sol, u2_sol = newton_solver(u1[n, :], u2[n, :], tau, h, alpha=(2.5, 1.5), kappa=(0.2, 0.3), iter=2)
    u1[n + 1, :], u2[n + 1, :] = u1_sol, u2_sol
```


      0%|          | 0/5999 [00:00<?, ?it/s]


Синим обозначена электронная температура $T_e$, красным обозначена ионная температура $T_i$. В отличии от однородного случая без теплообмена обе волны движутся с примерно одинаковой скоростью.

<img alt="SegmentLocal" height="400" src="lab_gifs\spitz_with_f.gif" title="segment" width="600"/>
