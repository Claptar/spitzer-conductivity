import numpy as np
from numba import njit

@njit
def k_12(y_m, y_m_1, alpha, kappa=None):
    kappa = 1 if kappa is None else kappa
    return kappa * (y_m ** alpha + y_m_1 ** alpha) / 2


@njit
def spitz_homogeneous_nonlinear(y_s, tau, h, y_n, alpha=2.5, kappa=0.2):
    # init variables
    M = len(y_s)
    a = np.zeros(M - 1)
    b = np.zeros(M)
    c = np.zeros(M - 1)
    d = np.zeros(M)
    sigma = tau / h ** 2
    dkappa = lambda x: kappa * 0.5 * alpha * x ** (alpha - 1)
    # first row
    m = 0
    c[m] = k_12(y_s[m + 1], y_s[m], alpha, kappa) + dkappa(y_s[m + 1]) * (y_s[m + 1] - y_s[m])
    b[m] = 1 / sigma + k_12(y_s[m + 1], y_s[m], alpha, kappa) - dkappa(y_s[m]) * (y_s[m + 1] - y_s[m])
    d[m] = (y_s[m] - y_n[m]) / sigma - k_12(y_s[m + 1], y_s[m], alpha, kappa) * (y_s[m + 1] - y_s[m])
    # rows m = 2,...,M - 1
    for m in range(1, M - 1):
        a[m - 1] = k_12(y_s[m - 1], y_s[m], alpha, kappa) - dkappa(y_s[m - 1]) * (y_s[m] - y_s[m - 1])
        c[m] = k_12(y_s[m + 1], y_s[m], alpha, kappa) + dkappa(y_s[m + 1]) * (y_s[m + 1] - y_s[m])
        b[m] = 1 / sigma + k_12(y_s[m + 1], y_s[m], alpha, kappa) + k_12(y_s[m - 1], y_s[m], alpha, kappa) - dkappa(y_s[m]) * (y_s[m + 1] - y_s[m]) + dkappa(y_s[m]) * (y_s[m] - y_s[m - 1])
        d[m] = (y_s[m] - y_n[m]) / sigma - k_12(y_s[m + 1], y_s[m], alpha, kappa) * (y_s[m + 1] - y_s[m]) + k_12(y_s[m - 1], y_s[m], alpha, kappa) * (y_s[m] - y_s[m - 1])
    # last row
    m = M - 1
    b[m] = 1
    d[m] = 0
    return a, b, c, d