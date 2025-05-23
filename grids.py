import numpy as np
from typing import Tuple, Union


class GridGenerator:
    def __init__(
        self, range_size: Union[float, Tuple[float, float, float]], points_per_dim: Union[int, Tuple[int, int, int]]
    ):
        """
        初始化网格生成器

        Parameters
        ----------
        range_size : float / tuple, 网格的范围大小
                    如果是float，表示所有维度相同范围[-range_size, range_size]
                    如果是3元tuple，表示每个维度的范围[(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        points_per_dim : int / tuple, 每维度的点数
                    如果是int，表示所有维度相同点数
                    如果是3元tuple，表示每个维度的点数
        """
        self.range_size = range_size
        self.points_per_dim = points_per_dim
        self.x = None
        self.y = None
        self.z = None
        self.X = None
        self.Y = None
        self.Z = None
        self.dx = None
        self.dy = None
        self.dz = None
        self.dV = None

    def generate_grid(self):
        """
        生成3D网格

        Returns
        -------
        X, Y, Z : np.ndarray
            三维坐标
        """
        # 处理范围参数
        if isinstance(self.range_size, (int, float)):
            x_range = (-self.range_size, self.range_size)
            y_range = (-self.range_size, self.range_size)
            z_range = (-self.range_size, self.range_size)
        elif len(self.range_size) == 3:
            x_range, y_range, z_range = self.range_size
        else:
            raise ValueError("range_size should be either a number or a tuple of 3 ranges")

        # 处理点数参数
        if isinstance(self.points_per_dim, int):
            x_points = y_points = z_points = self.points_per_dim
        elif len(self.points_per_dim) == 3:
            x_points, y_points, z_points = self.points_per_dim
        else:
            raise ValueError("points_per_dim should be either an integer or a tuple of 3 integers")

        # 生成坐标轴
        x = np.linspace(x_range[0], x_range[1], x_points)
        y = np.linspace(y_range[0], y_range[1], y_points)
        z = np.linspace(z_range[0], z_range[1], z_points)

        self.x = x
        self.y = y
        self.z = z
        # 计算步长
        self.dx = x[1] - x[0] if len(x) > 1 else 0
        self.dy = y[1] - y[0] if len(y) > 1 else 0
        self.dz = z[1] - z[0] if len(z) > 1 else 0

        self.dV = self.dx * self.dy * self.dz

        # 生成网格
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing="ij")

        return self.X, self.Y, self.Z

    def get_step_sizes(self):
        """返回各维度的步长"""
        return self.dx, self.dy, self.dz


class PlaneGridGenerator:
    """
    平面网格生成器

    Parameters
    ----------
    center : Tuple[float, float, float]
        平面中心点 (x, y, z)
    normal : Tuple[float, float, float]
        平面法向量 (nx, ny, nz)
    size : float
        平面大小（正方形边长）
    points_per_dim : int
        每维度的点数
    rotation_angle : float
        绕法向量旋转的角度（度）
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        normal: Tuple[float, float, float],
        size: Union[float, Tuple[float, float]],
        points_per_dim: Union[int, Tuple[int, int]],
        rotation_angle: float = 0.0,
    ):
        self.center = center
        self.normal = normal
        self.size = size
        self.points_per_dim = points_per_dim
        self.rotation_angle = np.deg2rad(rotation_angle)
        self.X = None
        self.Y = None
        self.Z = None
        self.u = None
        self.v = None
        self.U = None
        self.V = None
        self.du = None
        self.dv = None

    def generate_grid(self):
        """
        生成平面网格

        Returns
        -------
        X, Y, Z : np.ndarray
            平面网格的三维坐标，形状为 (points_per_dim, points_per_dim)
        """
        # 处理范围参数
        if isinstance(self.size, (int, float)):
            u_size = v_size = self.size
        elif len(self.size) == 2:
            u_size, v_size = self.size
        else:
            raise ValueError("size should be either a number or a tuple of 2 numbers")
        
        if isinstance(self.points_per_dim, int):
            u_points = v_points = self.points_per_dim
        elif len(self.points_per_dim) == 2:
            u_points, v_points = self.points_per_dim
        else:
            raise ValueError("points_per_dim should be either an integer or a tuple of 2 integers")

        # 归一化法向量
        normal = np.array(self.normal) / np.linalg.norm(self.normal)

        # 找到与法向量正交的两个基向量（使用Gram-Schmidt过程）
        if abs(normal[2]) < 1.0:
            v1 = np.cross(normal, [0, 0, 1])
        else:
            v1 = np.cross(normal, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)

        # 应用旋转角度
        if self.rotation_angle != 0:
            cos_a = np.cos(self.rotation_angle)
            sin_a = np.sin(self.rotation_angle)
            v1_rot = cos_a * v1 + sin_a * v2
            v2_rot = -sin_a * v1 + cos_a * v2
            v1, v2 = v1_rot, v2_rot

        # 生成二维网格
        u = np.linspace(-u_size / 2, u_size / 2, u_points)
        v = np.linspace(-v_size / 2, v_size / 2, v_points)
        self.u = u
        self.v = v
        self.du = u[1] - u[0] if len(u) > 1 else 0
        self.dv = v[1] - v[0] if len(v) > 1 else 0
        U, V = np.meshgrid(u, v, indexing="ij")
        self.U, self.V = U, V

        # 计算三维坐标
        center = np.array(self.center)
        self.X = center[0] + U * v1[0] + V * v2[0]
        self.Y = center[1] + U * v1[1] + V * v2[1]
        self.Z = center[2] + U * v1[2] + V * v2[2]

        return self.X, self.Y, self.Z

    def get_step_sizes(self):
        """
        返回平面网格的步长

        Returns
        -------
        du, dv : float
            两个维度的步长
        """
        return self.du, self.dv
