import numpy as np
import plotly.graph_objects as go
import wavefunction as wf
from typing import Optional, Union, Tuple
from dataclasses import dataclass, field
from plot_utils import clip_points_by_phi


class MonteCarlo:
    """
    蒙特卡洛方法类，用于在氢原子波函数中生成随机采样点

    Parameters
    ----------
    hydrogen : Union[wf.DiracHydrogen, wf.SchrodingerHydrogen]
        氢原子模型，可以是 DiracHydrogen 或 SchrodingerHydrogen
    R : Union[float, int]
        随机点生成范围，[-R, R]^3 或球形区域的半径
    n_points : int
        每次迭代生成的随机点数量
    """

    def __init__(self, hydrogen: Union[wf.DiracHydrogen, wf.SchrodingerHydrogen], R: Union[float, int], n_points: int):
        self.hydrogen = hydrogen
        self.R = R
        self.n_points = n_points

    def generate_random_points_xyz(self):
        """
        在 [-R, R]^3 立方体内生成均匀分布的随机点

        Returns
        -------
        np.ndarray
            形状为 (n_points, 3) 的随机点坐标数组，包含 x, y, z 坐标
        """
        return np.random.uniform(-self.R, self.R, (self.n_points, 3))

    def generate_random_points_spherical(self):
        """
        在半径为 R 的球形区域内生成均匀分布的随机点

        Returns
        -------
        np.ndarray
            形状为 (n_points, 3) 的随机点坐标数组，包含 x, y, z 坐标
        """
        r = self.R * np.cbrt(np.random.uniform(0, 1, self.n_points))
        theta = np.arccos(2 * np.random.uniform(0, 1, self.n_points) - 1)
        phi = np.random.uniform(0, 2 * np.pi, self.n_points)

        # 转换为笛卡尔坐标
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.vstack((x, y, z)).T

    def compute_probability(self, points, component, use_squared=False, is_real=False):
        """
        计算给定点的波函数概率密度或模

        Parameters
        ----------
        points : np.ndarray
            形状为 (n_points, 3) 的点坐标数组，包含 x, y, z 坐标
        component : str
            要计算的分量，'density' 表示概率密度，'psi1' 到 'psi4'（Dirac）或 'psi'（Schrödinger）
        use_squared : bool
            使用波函数模平方而非波函数模
        is_real : bool
            实 Schrodinger 波函数？

        Returns
        -------
        prob : np.ndarray
            概率密度或波函数模的数组
        phase : np.ndarray or None
            波函数相位（如果适用），否则为 None
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        phase = None
        if isinstance(self.hydrogen, wf.DiracHydrogen):
            psi = self.hydrogen.compute_psi_xyz(x, y, z)
            if component == "density":
                prob = psi.density()
            elif component in ["psi1", "psi2", "psi3", "psi4"]:
                idx = int(component[-1]) - 1
                prob = np.abs(psi.psi[idx])
                phase = np.angle(psi.psi[idx])
                if use_squared:
                    prob = prob**2
            else:
                raise ValueError("Invalid component for BaseSpinor. Choose 'density' or 'psi1' to 'psi4'.")
        else:
            psi = self.hydrogen.compute_psi_xyz(x, y, z, isreal=is_real)
            if component == "density":
                prob = np.abs(psi) ** 2
            elif component == "psi":
                prob = np.abs(psi)
                phase = np.angle(psi)
                if use_squared:
                    prob = prob**2
            else:
                raise ValueError("Invalid component for Schrödinger wavefunction. Choose 'density' or 'psi'.")

        return prob, phase

    def sample_points(self, num_samples, component, method="spherical", use_squared=False, is_real=False):
        """
        使用拒绝采样法生成符合分布的采样点

        Parameters
        ----------
        num_samples : int
            需要的采样点数量
        component : str
            要采样的分量，'density' 或 'psi1' 到 'psi4'（Dirac）或 'psi'（Schrödinger）
        method : str, optional
            随机点生成方法，'spherical'（球形区域）或 'xyz'（立方体区域），默认为 'spherical'
        use_squared : bool
            使用波函数模平方而非波函数模
        is_real : bool
            实 Schrodinger 波函数？

        Returns
        -------
        accepted_points : np.ndarray
            形状为 (num_samples, 3) 的采样点坐标数组
        accepted_phases : np.ndarray or None
            形状为 (num_samples,) 的相位数组（如果适用），否则为 None
        accepted_probs : np.ndarray
            形状为 (num_samples,) 的概率密度或波函数模数组
        """
        accepted_points = []
        accepted_phases = []
        accepted_probs = []
        p_max = 0
        i = 1
        while len(accepted_points) < num_samples:
            if method == "spherical":
                points = self.generate_random_points_spherical()
            else:
                points = self.generate_random_points_xyz()
            prob, phase = self.compute_probability(points, component, use_squared, is_real)  # 获取概率和相位
            p_max = np.maximum(p_max, np.max(prob))
            u = np.random.uniform(0, 1, self.n_points)
            mask = u < prob / p_max
            accepted_points.extend(points[mask])
            accepted_probs.extend(prob[mask])
            if phase is not None:
                accepted_phases.extend(phase[mask])
            print(
                f"iteration {i}, accepted points {len(accepted_points)}, acceptance probability {len(accepted_points) / (i * self.n_points):.4f}"
            )
            i += 1

        accepted_points = np.array(accepted_points)[:num_samples]
        accepted_probs = np.array(accepted_probs)[:num_samples]
        if accepted_phases:
            accepted_phases = np.array(accepted_phases)[:num_samples]
            return accepted_points, accepted_phases, accepted_probs
        else:
            return accepted_points, None, accepted_probs


@dataclass
class MCPlotConfig:
    """
    蒙特卡洛采样点可视化配置

    Parameters
    ----------
    clip : bool
        是否启用 phi 角度裁剪，默认为 False
    phi_min : float
        裁剪的 phi 角度最小值（度），默认为 0
    phi_max : float
        裁剪的 phi 角度最大值（度），默认为 90
    color : Tuple[str, str]
        颜色标度，分别为 (密度颜色, 相位颜色)，默认为 ("#636efa", "hsv")
    use_prob_color : bool
        是否根据概率密度进行着色（从白色到红色），默认为 False
    prob_colorscale : str
        概率密度着色的颜色标度，默认为 'Reds'（从白色到红色）
    use_prob_size : bool
        是否根据概率密度调整点大小，默认为 False
    size_range : Tuple[float, float]
        点大小范围 (最小, 最大)，默认为 (2, 8)
    opacity : float
        散点图的透明度，范围 [0, 1]，默认为 1.0
    line_color : str
        线条颜色，默认为 "white"
    line_width : float
        线条宽度，默认为 0
    camera : dict
        3D 场景的相机设置，默认为 dict(eye=dict(x=1.5, y=1.5, z=1.5), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0))
    xyzrange : Optional[Union[float, Tuple[float, float, float]]]
        轴范围，单个 float 表示三个轴相同范围，元组表示 (x, y, z) 范围，None 表示使用 aspectmode="data"
    width : int
        图像的宽度，默认为 1000
    height : int
        图像的高度，默认为 1000
    """

    clip: bool = False
    phi_min: float = 0
    phi_max: float = 90
    color: Tuple[str, str] = ("#636efa", "hsv")
    use_prob_color: bool = False
    prob_colorscale: str = "Reds"
    use_prob_size: bool = False
    size_range: Tuple[float, float] = (2, 8)
    opacity: float = 1.0
    line_color: str = "white"
    line_width: float = 0
    camera: dict = field(
        default_factory=lambda: dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        )
    )
    xyzrange: Optional[Union[float, Tuple[float, float, float]]] = None
    width: int = 1000
    height: int = 1000


def MC_plot(points, config: MCPlotConfig, phases=None, probs=None):
    """
    使用 Scatter3d 可视化蒙特卡洛采样点

    Parameters
    ----------
    points : np.ndarray
        形状为 (num_samples, 3) 的采样点坐标数组，包含 x, y, z 坐标
    config : MCPlotConfig
        蒙特卡洛采样点可视化配置
    phases : np.ndarray, optional
        形状为 (num_samples,) 的相位数组，用于着色，默认为 None
    probs : np.ndarray, optional
        形状为 (num_samples,) 的概率密度或波函数模数组，用于着色或调整点大小，默认为 None

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly Figure
    """
    # 创建 Plotly 图形
    fig = go.Figure()

    # 处理裁剪
    if config.clip:
        points, phases, probs = clip_points_by_phi(points, config.phi_min, config.phi_max, phases, probs)

    # 选择颜色标度和着色模式
    if config.use_prob_color and probs is not None:
        colorscale = config.prob_colorscale
        color = probs
        cmin, cmax = np.min(probs), np.max(probs)
        colorbar_title = "Probability"
    elif phases is not None:
        colorscale = config.color[1]
        color = phases
        cmin, cmax = -np.pi, np.pi
        colorbar_title = "Phase"
    else:
        colorscale = None
        color = config.color[0]
        cmin, cmax = None, None
        colorbar_title = None

    # 确定点大小
    if config.use_prob_size and probs is not None:
        size_min, size_max = config.size_range
        norm_probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs) + 1e-10)  # 归一化到 [0, 1]
        sizes = size_min + (size_max - size_min) * norm_probs
    else:
        sizes = 3

    # 添加散点图
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=sizes,
                color=color,
                colorscale=colorscale,
                opacity=config.opacity,
                cmin=cmin,
                cmax=cmax,
                showscale=(config.use_prob_color and probs is not None) or phases is not None,
                colorbar=dict(title=colorbar_title) if colorbar_title else None,
                line=dict(color=config.line_color, width=config.line_width),
            ),
        )
    )

    # 设置布局
    scene_dict = dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        camera=config.camera,
        aspectmode="cube" if config.xyzrange is not None else "data",
    )

    if config.xyzrange is not None:
        if isinstance(config.xyzrange, tuple):
            x_range, y_range, z_range = config.xyzrange
        else:
            x_range = y_range = z_range = config.xyzrange
        scene_dict.update(
            xaxis=dict(range=[-x_range, x_range]),
            yaxis=dict(range=[-y_range, y_range]),
            zaxis=dict(range=[-z_range, z_range]),
        )

    fig.update_layout(
        title_text="Monte Carlo Sampling of Hydrogen Wavefunction",
        scene=scene_dict,
        width=config.width,
        height=config.height,
    )

    return fig
