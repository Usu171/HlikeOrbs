import numpy as np
import scipy.special as sp
import copy
import grids as gr
from typing import Optional, Union, List
from dirac_utils import (
    C_ab,
    Y_ab,
    nk,
    gamma_k,
    epsilon_nk,
    lambda_nk,
    rho_,
    Ank,
    r1,
    r2,
    rc,
    rf,
    c1_,
)

hbar = 1


def xyz_to_spherical(x, y, z):
    """
    将笛卡尔坐标转换为球坐标

    Parameters
    ----------
    x : float, x坐标
    y : float, y坐标
    z : float, z坐标

    Returns
    -------
    r : float, 径向距离
    theta : float, 极角
    phi : float, 方位角
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.divide(z, r, where=(r != 0), out=np.full_like(z, np.pi / 2, dtype=np.float64)))
    phi = np.arctan2(y, x)
    return r, theta, phi


class BaseSpinor:
    """
    旋量的基类，存储和操作旋量波函数

    Parameters
    ----------
    psi : np.ndarray
        旋量波函数，4分量复数数组
    active_components : list, optional
        用于计算概率密度的分量索引列表，默认为 None（使用所有分量）
    """

    def __init__(self, psi, active_components=None):
        self.psi = np.asarray(psi, dtype=complex)
        # 如果未指定 active_components，则使用所有分量
        self.active_components = active_components if active_components is not None else list(range(self.psi.shape[0]))

    def __repr__(self):
        return f"{self.__class__.__name__}(psi={self.psi}, active_components={self.active_components})"

    def copy(self):
        return copy.deepcopy(self)

    def density(self):
        """
        计算旋量波函数的概率密度

        Returns
        -------
        array-like : 概率密度，沿指定分量的绝对值平方和
        """
        return np.sum(np.abs(self.psi[self.active_components]) ** 2, axis=0)

    def component_fractions(self):
        """
        计算并打印各分量占比

        Parameters
        ----------
        dV : float
            网格的体视素体积，用于积分

        Returns
        -------
        dict : 各分量的占比，键为分量索引，值为占比（浮点数）
        """
        component_squares = np.abs(self.psi) ** 2
        total_density = np.sum(component_squares, axis=0)
        total_density = np.sum(total_density)

        if total_density == 0:
            print("total density is zero")
            return {}
        # 计算各分量的积分和占比
        fractions = {i: np.sum(component_squares[i]) / total_density for i in range(self.psi.shape[0])}

        for i, fraction in fractions.items():
            print(f"Component {i + 1}: {fraction:.6f}")
        print(f"Sum of fractions: {sum(fractions.values()):.6f}")

        return fractions


class DiracSpinor(BaseSpinor):
    """
    Dirac旋量类，表示Dirac方程的解

    Parameters
    ----------
    psi : np.ndarray, Dirac旋量波函数，4分量复数数组
    """

    def current(self):
        psi = self.psi
        jx = (
            np.conj(psi[3]) * psi[0] + np.conj(psi[2]) * psi[1] + np.conj(psi[1]) * psi[2] + np.conj(psi[0]) * psi[3]
        ).real
        jy = (
            -np.conj(psi[3]) * psi[0] + np.conj(psi[2]) * psi[1] - np.conj(psi[1]) * psi[2] + np.conj(psi[0]) * psi[3]
        ).imag
        jz = (
            np.conj(psi[2]) * psi[0] - np.conj(psi[3]) * psi[1] + np.conj(psi[0]) * psi[2] - np.conj(psi[1]) * psi[3]
        ).real
        return np.array([jx, jy, jz])

    def to_wely(self):
        result = np.zeros_like(self.psi, dtype=complex)
        result[0] = (self.psi[0] - self.psi[2]) / np.sqrt(2)
        result[1] = (self.psi[1] - self.psi[3]) / np.sqrt(2)
        result[2] = (self.psi[0] + self.psi[2]) / np.sqrt(2)
        result[3] = (self.psi[1] + self.psi[3]) / np.sqrt(2)
        return WeylSpinor(result)


class WeylSpinor(BaseSpinor):
    def to_dirac(self):
        result = np.zeros_like(self.psi, dtype=complex)
        result[0] = (self.psi[0] + self.psi[2]) / np.sqrt(2)
        result[1] = (self.psi[1] + self.psi[3]) / np.sqrt(2)
        result[2] = (-self.psi[0] + self.psi[2]) / np.sqrt(2)
        result[3] = (-self.psi[1] + self.psi[3]) / np.sqrt(2)
        return DiracSpinor(result)


class DiracHydrogen:
    """
    Dirac氢原子模型，计算氢原子的Dirac波函数

    Parameters
    ----------
    n : int, 主量子数
    k : int, Dirac量子数
    m : float, 磁量子数
    Z : int, 原子序数
    alpha : float, 精细结构常数，默认为0.0072992701
    c : float, 光速，默认为137
    me : float, 电子质量，默认为1.0
    """

    def __init__(self, n, k, m, Z, alpha=0.0072992701, c=137, me=1.0):
        self.n = n
        self.k = k
        self.m = m
        self.Z = Z
        self.alpha = alpha
        self.c = c
        self.me = me

        self.n_k = nk(n, k)
        self.g_k = gamma_k(k, Z, alpha)
        self.e_nk = epsilon_nk(self.n_k, self.g_k, Z, alpha)
        self.l_nk = lambda_nk(self.e_nk, c, me)
        self.A_nk = Ank(k, self.n_k, self.g_k, self.l_nk, self.e_nk, Z, alpha)
        self.E = me * c**2 * self.e_nk

    def compute_psi(self, r, theta, phi, t=0):
        """
        计算Dirac氢原子在球坐标下的波函数

        Parameters
        ----------
        r : float, 径向距离
        theta : float, 极角（弧度）
        phi : float, 方位角（弧度）
        t : float, 时间，默认为0

        Returns
        -------
        DiracSpinor : Dirac自旋子波函数，包含4个分量
        """
        rho = rho_(self.l_nk, r)
        r_1 = r1(rho, self.n_k, self.g_k)
        r_2 = r2(rho, self.n_k, self.g_k, self.k, self.e_nk)
        r_c = rc(rho, self.g_k)
        g = rf(self.Z, self.alpha, self.k, self.g_k, r_1, r_2, r_c, 1)
        f = rf(self.Z, self.alpha, self.k, self.g_k, r_1, r_2, r_c, 0)
        c1 = c1_(self.E, self.A_nk, r, t)
        psi1 = g * C_ab(self.k, -self.m) * Y_ab(self.k, self.m - 0.5, theta, phi) * c1
        psi2 = -g * np.sign(self.k) * C_ab(self.k, self.m) * Y_ab(self.k, self.m + 0.5, theta, phi) * c1
        psi3 = 1j * f * C_ab(-self.k, -self.m) * Y_ab(-self.k, self.m - 0.5, theta, phi) * c1
        psi4 = 1j * f * np.sign(self.k) * C_ab(-self.k, self.m) * Y_ab(-self.k, self.m + 0.5, theta, phi) * c1
        return DiracSpinor([psi1, psi2, psi3, psi4])

    def compute_psi_xyz(self, x, y, z, t=0):
        """
        计算Dirac氢原子在笛卡尔坐标下的波函数

        Parameters
        ----------
        x : float, x坐标
        y : float, y坐标
        z : float, z坐标
        t : float, 时间，默认为0

        Returns
        -------
        DiracSpinor : Dirac自旋子波函数，包含4个分量
        """
        r, theta, phi = xyz_to_spherical(x, y, z)
        return self.compute_psi(r, theta, phi, t)


class SchrodingerHydrogen:
    """
    Schrödinger氢原子模型，计算氢原子的Schrödinger波函数

    Parameters
    ----------
    n : int, 主量子数
    l : int, 角量子数
    m : int, 磁量子数
    Z : int, 原子序数
    a_mu : float, me*a0/mu ,me为电子质量，a0为波尔半径，mu为约化质量
    mu : float, 约化质量，默认为1.0
    """

    def __init__(self, n, l, m, Z, a_mu=1, mu=1):
        self.n = n
        self.l = l
        self.m = m
        self.Z = Z
        self.a_mu = a_mu
        self.me = mu
        self.c1 = np.sqrt(((2 * Z) / (n * a_mu)) ** 3 * (sp.factorial(n - l - 1) / (2 * n * sp.factorial(n + l))))
        self.E = -(mu * Z**2) / (2 * n**2)

    def compute_psi(self, r, theta, phi, t=0, isreal=False):
        """
        计算Schrödinger氢原子在球坐标下的波函数

        Parameters
        ----------
        r : float, 径向距离
        theta : float, 极角（弧度）
        phi : float, 方位角（弧度）
        t : float, 时间，默认为0
        isreal : bool, 是否返回实数波函数（适用于非零磁量子数），默认为False

        Returns
        -------
        array-like : Schrödinger波函数，复数数组
        """
        rho = 2 * self.Z * r / (self.n * self.a_mu)
        c2 = rho**self.l * np.exp(-rho / 2) * np.exp(-1j * self.E * t)
        cR = self.c1 * c2 * sp.genlaguerre(self.n - self.l - 1, 2 * self.l + 1)(rho)
        if isreal and self.m != 0:
            m1 = np.abs(self.m)
            Y1 = sp.sph_harm_y(self.l, m1, theta, phi)
            Y2 = sp.sph_harm_y(self.l, -m1, theta, phi)
            c = cR * (1 / np.sqrt(2))
            Y = (-1) ** m1 * Y1 + Y2 if self.m < 0 else (-1) ** (m1 + 1) * 1j * Y1 + 1j * Y2
            psi = Y * c
        else:
            psi = cR * sp.sph_harm_y(self.l, self.m, theta, phi)
        return np.asarray(psi, dtype=complex)

    def compute_psi_xyz(self, x, y, z, t=0, isreal=False):
        """
        计算Schrödinger氢原子在笛卡尔坐标下的波函数

        Parameters
        ----------
        x : float, x坐标
        y : float, y坐标
        z : float, z坐标
        t : float, 时间，默认为0
        isreal : bool, 是否返回实数波函数（适用于非零磁量子数），默认为False

        Returns
        -------
        array-like : Schrödinger波函数，复数数组
        """
        r, theta, phi = xyz_to_spherical(x, y, z)
        return self.compute_psi(r, theta, phi, t, isreal)


def generate_hybrid_orbital(
    wavefunctions: List[Union[BaseSpinor, np.ndarray]],
    coefficients: List[Union[float, complex]],
    normalize: bool = False,
) -> Union[BaseSpinor, np.ndarray]:
    """
    生成杂化轨道，通过线性组合n个波函数。

    Parameters
    ----------
    wavefunctions : List[Union[BaseSpinor, np.ndarray]]
        n个波函数列表，每个波函数为BaseSpinor或np.ndarray
    coefficients : List[Union[float, complex]]
        n个组合系数
    normalize : bool, optional
        是否归一化系数，使sum(|c_i|^2) = 1，默认为False

    Returns
    -------
    Union[BaseSpinor, np.ndarray]
        杂化轨道波函数，与输入波函数类型一致

    Raises
    ------
    ValueError
        如果输入波函数类型不一致、形状不匹配或系数数量错误
    """
    # 验证输入
    if len(wavefunctions) != len(coefficients):
        raise ValueError("Number of wavefunctions must match number of coefficients.")
    if len(wavefunctions) == 0:
        raise ValueError("Wavefunctions list cannot be empty.")

    # 检查波函数类型一致性
    wavefunction_type = type(wavefunctions[0])
    if not all(isinstance(wf, wavefunction_type) for wf in wavefunctions):
        raise ValueError("All wavefunctions must be of the same type (BaseSpinor or np.ndarray).")

    # 检查波函数形状一致性
    if isinstance(wavefunctions[0], BaseSpinor):
        ref_shape = wavefunctions[0].psi.shape
        if not all(wf.psi.shape == ref_shape for wf in wavefunctions):
            raise ValueError("All BaseSpinor wavefunctions must have the same shape.")
    else:
        ref_shape = wavefunctions[0].shape
        if not all(wf.shape == ref_shape for wf in wavefunctions):
            raise ValueError("All np.ndarray wavefunctions must have the same shape.")

    # 系数归一化
    coeffs = np.array(coefficients, dtype=complex)
    if normalize:
        norm = np.sqrt(np.sum(np.abs(coeffs) ** 2))
        if norm == 0:
            raise ValueError("Cannot normalize zero coefficients.")
        coeffs = coeffs / norm

    # 线性组合
    if isinstance(wavefunctions[0], BaseSpinor):
        # 旋量：对每个分量加权求和
        hybrid_psi = np.zeros(ref_shape, dtype=complex)
        for wf, c in zip(wavefunctions, coeffs):
            hybrid_psi += c * wf.psi
        return BaseSpinor(hybrid_psi)
    else:
        # Schrödinger波函数：直接加权求和
        hybrid_psi = np.zeros(ref_shape, dtype=complex)
        for wf, c in zip(wavefunctions, coeffs):
            hybrid_psi += c * wf
        return hybrid_psi
