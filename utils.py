import numpy as np
from numba import njit


@njit
def k_12(y_m, y_m_1, alpha):
    return (y_m ** alpha + y_m_1 ** alpha) / 2


@njit
def qei(Te, Ti):
    if Te <= 1e-5:
        return 0.0
    else:
        return (Te - Ti) / Te ** 2


@njit
def make_diagonals(u1, u2, n, tau, h, M, mode='electrons'):
    # init variables
    a = np.zeros(M - 1)
    b = np.zeros(M)
    c = np.zeros(M - 1)
    d = np.zeros(M)
    u = u1[n, :] if mode == 'electrons' else u2[n, :]
    alpha = 2.5 if mode == 'electrons' else 1.5
    kappa = 0.2 if mode == 'electrons' else 0.3
    betta = -1 if mode == 'electrons' else 1
    sigma = kappa * tau / h ** 2
    # first row
    c[0] = k_12(u[0], u[1], alpha) * sigma
    b[0] = 1 + c[0]
    d[0] = - (u[0] + betta * qei(u1[n, 0], u2[n, 0]) * tau)
    # rows m = 2,...,M - 1
    for m in range(1, M - 1):
        a[m - 1] = k_12(u[m - 1], u[m], alpha) * sigma
        c[m] = k_12(u[m], u[m + 1], alpha) * sigma
        b[m] = 1 + a[m - 1] + c[m]
        d[m] = - (u[m] + betta * qei(u1[n, m], u2[n, m]) * tau)
    # last row
    m = M - 1
    a[m - 1] = k_12(u[m - 1], u[m], alpha) * sigma
    b[m] = 1 + u[m] ** alpha + a[m - 1]
    d[m] = - (u[m] + betta * qei(u1[n, m], u2[n, m]) * tau)
    return a, b, c, d


@njit
def thomas_solver(a, b, c, d):
    # init arrays
    M = len(b)
    u_sol = np.zeros(M)
    P = np.zeros(M)
    Q = np.zeros(M)
    # Forward first coefs
    P[0] = c[0] / b[0]
    Q[0] = - d[0] / b[0]
    # Forward coefs n = 1 ... M - 1
    for m in range(1, M - 1):
        P[m] = c[m] / (b[m] - a[m - 1] * P[m - 1])
        Q[m] = (a[m - 1] * Q[m - 1] - d[m]) / (b[m] - a[m - 1] * P[m - 1])
    #Forward last
    m = M - 1
    Q[m] = (a[m - 1] * Q[m - 1] - d[m]) / (b[m] - a[m - 1] * P[m - 1])

    # backward
    u_sol[-1] = Q[-1]
    for m in range(M - 1, -1, -1):
        sol = P[m - 1] * u_sol[m] + Q[m - 1]
        u_sol[m - 1] = sol if sol > 0 else 0
    return u_sol