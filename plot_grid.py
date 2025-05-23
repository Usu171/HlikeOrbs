import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator
from skimage import measure

import grids as gr
import wavefunction as wf
from plot_utils import clip_faces_by_phi, clip_current_by_phi, create_phi_mask


@dataclass
class DiracPlotConfig:
    """Dirac 所有轨道 可视化配置

    Parameters
    ----------
    plot_type : str
        要绘制的图表类型: "density", "current", "psi1", "psi2", "psi3", "psi4" 或 "all"
    density_scale : float
        密度等值面大小
    psi_scale : Union[float, Tuple[float, float, float, float]]
        波函数分量等值面的大小（可以是单一值或四个值的元组）
    points_per_unit : float
        current 每个单位长度采样的数量，如果 use_fixed_points 为 True，则为采样点的数量
    use_fixed_points : bool
        是否使用固定数量的采样点
    clip : bool
        是否启用 phi 角度裁剪
    phi_min, phi_max : float, float
        裁剪的 phi 角度范围（度）
    current_exponent : float
        current的指数缩放因子
    current_factor : float
        current的缩放因子
    scale_factors : Optional[Tuple[float, float, float, float]]
        四个波函数分量的缩放因子（可选）
    relative_density : bool
        True时使用相对于最大值的isovalue，False时使用绝对值（密度）
    relative_psi : bool
        True时使用相对于最大值的isovalue，False时使用绝对值（波函数）
    wely : bool
        是否使用 Wely 基
    psi_component : Optional[str]
        real 仅实部，imag 仅虚部
    use_squared : bool
        使用波函数模平方而非波函数模
    camera : dict
        3D 场景的相机设置
    color : Tuple[str, str, str]
        密度的colorscale，波函数的colorscale，current的colorscale
    xyzrange : Optional[Union[float, Tuple[float, float, float]]]
        轴范围，单个 float 表示三个轴相同范围，元组表示 (x, y, z) 范围，None 表示使用 aspectmode="data"
    width, height: int, int
        图像的宽度和高度
    """

    plot_type: str = "all"  # 新增字段，指定绘图类型
    density_scale: float = 0.01
    psi_scale: Union[float, Tuple[float, float, float, float]] = 0.1
    points_per_unit: float = 0.35
    use_fixed_points: bool = False
    clip: bool = False
    phi_min: float = 0
    phi_max: float = 90
    current_exponent: float = 0.2
    current_factor: float = 2
    scale_factors: Optional[Tuple[float, float, float, float]] = None
    relative_density: bool = True
    relative_psi: bool = True
    wely: bool = False
    psi_component: Optional[str] = None
    use_squared: bool = False
    camera: dict = field(
        default_factory=lambda: dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        )
    )
    color: Tuple[str, str, str] = ("#a8c9f5", "hsv", "Blues")
    xyzrange: Optional[Union[float, Tuple[float, float, float]]] = None
    width: int = 1500
    height: int = 1000


def Dirac_plot(spinor: wf.DiracSpinor, grid: gr.GridGenerator, config: DiracPlotConfig):
    """可视化 Dirac 波函数

    Parameters
    ----------
    spinor : wavefunction.DiracSpinor
        DiracSpinor
    grid : grids.GridGenerator
        grid.GridGenerator
    config : DiracPlotConfig
        可视化参数配置

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly Figure
    """
    # 根据 plot_type 设置子图布局
    if config.plot_type == "all":
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[[{"type": "scene"} for _ in range(3)] for _ in range(2)],
            # subplot_titles=["Density", "psi1", "psi2", "Current", "psi3", "psi4"],
            horizontal_spacing=0.01,
            vertical_spacing=0.01,
        )
        subplot_map = {
            "density": (1, 1),
            "current": (2, 1),
            "psi1": (1, 2),
            "psi2": (1, 3),
            "psi3": (2, 2),
            "psi4": (2, 3),
        }
    else:
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "scene"}]],
            subplot_titles=[config.plot_type.capitalize()],
        )
        subplot_map = {config.plot_type: (1, 1)}

    x_min, y_min, z_min = grid.x[0], grid.y[0], grid.z[0]
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    X, Y, Z = grid.X, grid.Y, grid.Z

    # 密度等值面
    if config.plot_type in ["density", "all"]:
        density = spinor.density()
        max_density = np.max(density)
        print(f"Sum = 1? {np.sum(density) * grid.dV}")
        isovalue_density = config.density_scale * max_density if config.relative_density else config.density_scale
        vertices_density, faces_density, _, _ = measure.marching_cubes(
            density, level=isovalue_density, spacing=(dx, dy, dz)
        )
        vertices_density += [x_min, y_min, z_min]

        if config.clip:
            faces_density = clip_faces_by_phi(vertices_density, faces_density, config.phi_min, config.phi_max)

        fig.add_trace(
            go.Mesh3d(
                x=vertices_density[:, 0],
                y=vertices_density[:, 1],
                z=vertices_density[:, 2],
                i=faces_density[:, 0],
                j=faces_density[:, 1],
                k=faces_density[:, 2],
                color=config.color[0],
            ),
            row=subplot_map["density"][0],
            col=subplot_map["density"][1],
        )

    # 流
    if config.plot_type in ["current", "all"]:
        current = spinor.current()

        if config.use_fixed_points:
            n_x = n_y = n_z = max(2, int(config.points_per_unit))
            idx_x = np.linspace(0, len(grid.x) - 1, n_x, dtype=int)
            idx_y = np.linspace(0, len(grid.y) - 1, n_y, dtype=int)
            idx_z = np.linspace(0, len(grid.z) - 1, n_z, dtype=int)
        else:
        # 每个维度的总长度
            Lx, Ly, Lz = grid.x[-1] - grid.x[0], grid.y[-1] - grid.y[0], grid.z[-1] - grid.z[0]
            # 每个维度中采样的点数
            n_x = max(2, int(config.points_per_unit * Lx) + 1)
            n_y = max(2, int(config.points_per_unit * Ly) + 1)
            n_z = max(2, int(config.points_per_unit * Lz) + 1)
            idx_x = np.linspace(0, len(grid.x) - 1, n_x, dtype=int)
            idx_y = np.linspace(0, len(grid.y) - 1, n_y, dtype=int)
            idx_z = np.linspace(0, len(grid.z) - 1, n_z, dtype=int)

        X_, Y_, Z_ = np.meshgrid(grid.x[idx_x], grid.y[idx_y], grid.z[idx_z], indexing='ij')
        current = current[:, idx_x][:, :, idx_y][:, :, :, idx_z]

        norm = np.sqrt(np.sum(current**2, axis=0))
        scaled = (norm / norm.max()) ** config.current_exponent / config.current_factor
        norm_safe = np.where(norm > 0, norm, 1e-10)
        scaled_current = current / norm_safe * scaled

        if config.clip:
            scaled_current = clip_current_by_phi(X_, Y_, Z_, scaled_current, config.phi_min, config.phi_max)

        fig.add_trace(
            go.Cone(
                x=X_.flatten(),
                y=Y_.flatten(),
                z=Z_.flatten(),
                u=scaled_current[0].flatten(),
                v=scaled_current[1].flatten(),
                w=scaled_current[2].flatten(),
                sizemode="absolute",
                colorscale=config.color[2],
                showscale=False,
                name="Current",
            ),
            row=subplot_map["current"][0],
            col=subplot_map["current"][1],
        )

    if config.wely:
        spinor = spinor.to_wely()

    # 缩放波函数 （如果有scale_factors）
    spinor_scaled = spinor.copy()
    if config.scale_factors is not None:
        for idx, scale in enumerate(config.scale_factors):
            spinor_scaled.psi[idx] *= scale

    if config.plot_type in ["psi1", "psi2", "psi3", "psi4"] or config.plot_type == "all":
        if config.plot_type == "all":
            indices = range(4)  # 绘制所有 psi 分量
        else:
            indices = [int(config.plot_type[-1]) - 1]  # 仅绘制指定的 psi 分量

        for i in indices:
            psi_i = spinor_scaled.psi[i]
            if config.psi_component == "real":
                magnitude = np.abs(np.real(psi_i))
            elif config.psi_component == "imag":
                magnitude = np.abs(np.imag(psi_i))
            else:
                magnitude = np.abs(psi_i)
            if config.use_squared:
                magnitude = magnitude**2
            max_mag = np.max(magnitude)
            d2_value = config.psi_scale[i] if isinstance(config.psi_scale, tuple) else config.psi_scale

            if max_mag > 0:
                isovalue = d2_value * max_mag if config.relative_psi else d2_value
                try:
                    vertices, faces, _, _ = measure.marching_cubes(magnitude, level=isovalue, spacing=(dx, dy, dz))
                    vertices += [x_min, y_min, z_min]
                    if config.clip:
                        faces = clip_faces_by_phi(vertices, faces, config.phi_min, config.phi_max)

                    interp = RegularGridInterpolator((grid.x, grid.y, grid.z), psi_i)
                    psi_i_vertices = interp(vertices)
                    phase = np.angle(psi_i_vertices)

                    if config.psi_component == "real":
                        phase = np.angle(np.real(psi_i_vertices))
                    elif config.psi_component == "imag":
                        phase = np.angle(1j * np.imag(psi_i_vertices))
                    else:
                        phase = np.angle(psi_i_vertices)

                    # showscale_ = (i == 0) if config.plot_type == "all" else True
                    fig.add_trace(
                        go.Mesh3d(
                            x=vertices[:, 0],
                            y=vertices[:, 1],
                            z=vertices[:, 2],
                            i=faces[:, 0],
                            j=faces[:, 1],
                            k=faces[:, 2],
                            intensity=phase,
                            colorscale=config.color[1],
                            cmin=-np.pi,
                            cmax=np.pi,
                            showscale=False,
                        ),
                        row=subplot_map[f"psi{i + 1}"][0],
                        col=subplot_map[f"psi{i + 1}"][1],
                    )
                except Exception as e:
                    print(f"Error processing psi{i + 1}: {e}")
                    pass

    # 相机角度
    for i, j in itertools.product(
        range(1, 3 if config.plot_type == "all" else 2), range(1, 4 if config.plot_type == "all" else 2)
    ):
        scene_dict: Dict[str, Any] = dict(
            camera=config.camera,
            xaxis=dict(
                showgrid=False,  # 禁用 x 轴网格
                zeroline=False,  # 禁用零线
                showticklabels=False,  # 隐藏刻度标签
                title="",  # 移除 x 轴标题
                showbackground=False,  # 禁用背景
            ),
            yaxis=dict(
                showgrid=False,  # 禁用 y 轴网格
                zeroline=False,  # 禁用零线
                showticklabels=False,  # 隐藏刻度标签
                title="",  # 移除 y 轴标题
                showbackground=False,  # 禁用背景
            ),
            zaxis=dict(
                showgrid=False,  # 禁用 z 轴网格
                zeroline=False,  # 禁用零线
                showticklabels=False,  # 隐藏刻度标签
                title="",  # 移除 z 轴标题
                showbackground=False,  # 禁用背景
            ),
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
                aspectmode="cube",
            )
        else:
            scene_dict.update(aspectmode="data")

        fig.update_scenes(**scene_dict, row=i, col=j)

    fig.update_layout(
        title_text="",
        width=config.width,
        height=config.height,
    )
    return fig
