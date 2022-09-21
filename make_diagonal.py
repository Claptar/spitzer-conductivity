import numpy as np
from numba import njit
from numpy.linalg import norm

@njit
def k_12(y_m, y_m_1, alpha, kappa=1):
    return kappa * (y_m ** alpha + y_m_1 ** alpha) / 2


@njit
def qei(y):
    Te, Ti = y
    if Te == 0:
        return 0, 0, 0
    try:
        q = (Te - Ti) / Te ** 2
        dq_dTe = (2 * Te * Ti - Te ** 2) / Te ** 4
        dq_dTi = - 1 / Te ** 2
    except Exception:
        q, dq_dTe, dq_dTi = 0, 0, 0
    return q, dq_dTe, dq_dTi


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


@njit
def zeldovich_nonlinear(y_s, tau, h, y_n, alpha=2.5, kappa=0.2):
    # init variables
    M = len(y_s)
    a = np.zeros(M - 1)
    b = np.zeros(M)
    c = np.zeros(M - 1)
    d = np.zeros(M)
    sigma = tau / h ** 2
    dkappa = lambda x: kappa * 0.5 * alpha * x ** (alpha - 1)
    # first row
    b[0] = 1
    d[0] = 0
    # rows m = 2,...,M - 1
    for m in range(1, M - 1):
        a[m - 1] = k_12(y_s[m - 1], y_s[m], alpha, kappa) -  dkappa(y_s[m - 1]) * (y_s[m] - y_s[m - 1])
        c[m] = k_12(y_s[m + 1], y_s[m], alpha, kappa) +  dkappa(y_s[m + 1]) * (y_s[m + 1] - y_s[m])
        b[m] = 1 / sigma + k_12(y_s[m + 1], y_s[m], alpha, kappa) + k_12(y_s[m - 1], y_s[m], alpha, kappa) - dkappa(y_s[m]) * (y_s[m + 1] - y_s[m]) + dkappa(y_s[m]) * (y_s[m] - y_s[m - 1])
        d[m] = (y_s[m] - y_n[m]) / sigma - k_12(y_s[m + 1], y_s[m], alpha, kappa) * (y_s[m + 1] - y_s[m]) + k_12(y_s[m - 1], y_s[m], alpha, kappa) * (y_s[m] - y_s[m - 1])
    # last row
    m = M - 1
    b[m] = 1
    d[m] = 0
    return a, b, c, d


def make_block_nonhomogen(u, m, tau, h, u_n, alpha=(2.5, 1.5), kappa=(0.2, 0.3)):
    alpha = np.array(alpha)
    kappa = np.array(kappa)
    sigma = tau / h ** 2
    A, B, C, D = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    dkappa = lambda x: kappa * 0.5 * alpha * x ** (alpha - 1)
    if m == 0:
        C = np.diagflat(k_12(u[m], u[m + 1], alpha, kappa) + dkappa(u[m + 1]) * (u[m + 1] - u[m]))
        B = np.diagflat(1 / sigma + k_12(u[m], u[m + 1], alpha, kappa) - dkappa(u[m]) * (u[m + 1] - u[m]))
        D = (u[m] - u_n[m]) / sigma - k_12(u[m + 1], u[m], alpha, kappa) * (u[m + 1] - u[m])
        return C, B, D
    A = np.diagflat(k_12(u[m - 1], u[m], alpha, kappa) - dkappa(u[m - 1]) * (u[m] - u[m - 1]))
    C = np.diagflat(k_12(u[m], u[m + 1], alpha, kappa) + dkappa(u[m + 1]) * (u[m + 1] - u[m]))
    B = np.diagflat(1 / sigma + k_12(u[m], u[m + 1], alpha, kappa) + k_12(u[m - 1], u[m], alpha, kappa) - dkappa(u[m]) * (u[m + 1] - u[m]) + dkappa(u[m]) * (u[m] - u[m - 1]))
    D = (u[m] - u_n[m]) / sigma - k_12(u[m + 1], u[m], alpha, kappa) * (u[m + 1] - u[m]) + k_12(u[m], u[m - 1], alpha, kappa) * (u[m] - u[m - 1])
    return A, B, C, D


def make_block_homogen(u, m, tau, h, u_n, alpha=(2.5, 1.5), kappa=(0.2, 0.3)):
    alpha = np.array(alpha)
    kappa = np.array(kappa)
    sigma = tau / h ** 2
    A, B, C, D = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    dkappa = lambda x: kappa * 0.5 * alpha * x ** (alpha - 1)
    q, dq_dTe, dq_dTi = qei(u[m])
    dphi_dy = np.array([- dq_dTe, dq_dTi])
    phi = np.array([- q, q])
    if m == 0:
        C = np.diagflat(k_12(u[m], u[m + 1], alpha, kappa) + dkappa(u[m + 1]) * (u[m + 1] - u[m]))
        B = np.diagflat(1 / sigma + k_12(u[m], u[m + 1], alpha, kappa) - dkappa(u[m]) * (u[m + 1] - u[m]))
        B -= np.diagflat(h ** 2 * dphi_dy)
        B[0, 1], B[1, 0] = dphi_dy[1], dphi_dy[0]
        D = (u[m] - u_n[m]) / sigma - k_12(u[m + 1], u[m], alpha, kappa) * (u[m + 1] - u[m])
        D -= h ** 2 * phi
        return C, B, D
    A = np.diagflat(k_12(u[m - 1], u[m], alpha, kappa) - dkappa(u[m - 1]) * (u[m] - u[m - 1]))
    C = np.diagflat(k_12(u[m], u[m + 1], alpha, kappa) + dkappa(u[m + 1]) * (u[m + 1] - u[m]))
    B = np.diagflat(1 / sigma + k_12(u[m], u[m + 1], alpha, kappa) + k_12(u[m - 1], u[m], alpha, kappa) - dkappa(u[m]) * (u[m + 1] - u[m]) + dkappa(u[m]) * (u[m] - u[m - 1]))
    D = (u[m] - u_n[m]) / sigma - k_12(u[m + 1], u[m], alpha, kappa) * (u[m + 1] - u[m]) + k_12(u[m], u[m - 1], alpha, kappa) * (u[m] - u[m - 1])
    if norm(h ** 2 * dphi_dy) / norm(np.diag(B)) < 1e14:
        B -= np.diagflat(h ** 2 * dphi_dy)
        B[0, 1], B[1, 0] = h ** 2 * dphi_dy[1], h ** 2 * dphi_dy[0]
        D -= h ** 2 * phi
    return A, B, C, D


def make_block_diagonals(u1_s, u2_s, tau, h, u1_n, u2_n, alpha=(2.5, 1.5), kappa=(0.2, 0.3), nonhomogen=True):
    M = len(u1_s)
    A = np.zeros((M - 1, 2, 2))
    B = np.zeros((M, 2, 2))
    C = np.zeros((M - 1, 2, 2))
    D = np.zeros((M, 2))
    u_s = np.vstack([u1_s, u2_s]).T
    u_n = np.vstack([u1_n, u2_n]).T
    make_block = make_block_nonhomogen if nonhomogen else make_block_homogen
    # first row
    C[0], B[0], D[0] = make_block(u_s, 0, tau, h, u_n, alpha=alpha, kappa=kappa)
    # rows m = 1, ..., M - 2
    for m in range(1, M - 1):
        A[m - 1], B[m], C[m], D[m] = make_block(u_s, m, tau, h, u_n, alpha=alpha, kappa=kappa)
    # last row
    B[M - 1] = np.eye(2)
    return A, B, C, D