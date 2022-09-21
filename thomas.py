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


def solve_blocks(A, B, C, D):
    M = len(D)
    u_sol = np.zeros_like(D)
    P = np.zeros_like(B)
    Q = np.zeros_like(D)
    # Forward first coefs
    P[0] = inv(B[0]) @ C[0]
    Q[0] = - inv(B[0]) @ D[0]
    # Forward coefs
    for m in range(1, M - 1):
        inv_denom = inv(B[m] - A[m - 1] @ P[m - 1])
        P[m] = inv_denom @ C[m]
        Q[m] = inv_denom @ (A[m - 1] @ Q[m - 1] - D[m])
    #Forward last
    m = M - 1
    Q[m] = inv(B[m] - A[m - 1] @ P[m - 1]) @ (A[m - 1] @ Q[m - 1] - D[m])

    # backward
    u_sol[-1] = Q[-1]
    for m in range(M - 1, -1, -1):
        u_sol[m - 1] = P[m - 1] @ u_sol[m] + Q[m - 1]

    return u_sol