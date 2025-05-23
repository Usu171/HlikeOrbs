import numpy as np
import scipy.special as sp

hbar = 1


def C_ab(a, b):
    return np.sqrt((a + 0.5 + b) / (2 * a + 1))


def Y_ab(a, b, theta, phi):
    return (sp.sph_harm_y(int(a), int(b), theta, phi)
            if a >= 0 else sp.sph_harm_y(int(-a - 1), int(b), theta, phi))


def nk(n, k):
    return n - np.abs(k)


def gamma_k(k, Z, alpha):
    return np.sqrt(k**2 - (Z * alpha)**2)


def epsilon_nk(n_k, g_k, Z, alpha):
    return (1 + (Z * alpha / (n_k + g_k))**2)**-0.5


def lambda_nk(e_nk, c, me):
    return me * c * np.sqrt(1 - e_nk**2) / hbar


def rho_(l_nk, r):
    return 2 * l_nk * r


def Ank(k, n_k, g_k, l_nk, e_nk, Z, alpha):
    if n_k > 0:
        return np.sqrt((1 / (2 * k * (k - g_k))) * (l_nk / (n_k + g_k)) *
                       (sp.gamma(n_k) / sp.gamma(n_k + 1 + 2 * g_k)) * 0.5 *
                       ((k * e_nk / g_k)**2 + k * e_nk / g_k))
    if n_k == 0:
        return np.sqrt(
            (1 / (2 * k * (k - g_k))) * (l_nk / (g_k)) *
            (1 / sp.gamma(1 + 2 * g_k)) * 1 / (2 * g_k) * (Z * alpha / k)**2)
    else:
        print("n_k is negative")
        return 0


def L_ab(a, b, rho):
    if a < 0:
        return 0
    else:
        return sp.genlaguerre(a, b)(rho)


def r1(rho, n_k, g_k):
    return rho * L_ab(n_k - 1, 2 * g_k + 1, rho)


def r2(rho, n_k, g_k, k, e_nk):
    return (
        (g_k - k * e_nk) / np.sqrt(1 - e_nk**2)) * L_ab(n_k, 2 * g_k - 1, rho)


def rc(rho, g_k):
    return rho**g_k * np.exp(-rho / 2)


def rf(Z, alpha, k, g_k, r_1, r_2, r_c, g):
    return (r_c * (Z * alpha * r_1 + (g_k - k) * r_2) if g else r_c *
            (Z * alpha * r_2 + (g_k - k) * r_1))


def c1_(E, A_nk, r, t):
    if t == 0:
        return A_nk * (1 / r)
    else:
        return np.exp(-1j * E * t * (1 / hbar)) * A_nk * (1 / r)
