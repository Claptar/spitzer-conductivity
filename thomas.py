import numpy as np
from numpy.linalg import inv
from numba import njit


@njit
def solve_equations(a, b, c, d):
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
        u_sol[m - 1] = P[m - 1] * u_sol[m] + Q[m - 1]

    return u_sol