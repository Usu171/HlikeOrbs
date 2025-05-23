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
    step : int
        current采样的步长
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
    step: int = 10
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
            subplot_titles=["Density", "psi1", "psi2", "Current", "psi3", "psi4"],
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
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
        X_, Y_, Z_ = (
            X[:: config.step, :: config.step, :: config.step],
            Y[:: config.step, :: config.step, :: config.step],
            Z[:: config.step, :: config.step, :: config.step],
        )
        current = current[:, ::10, ::10, ::10]
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

                    showscale_ = (i == 0) if config.plot_type == "all" else True
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
                            showscale=showscale_,
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
        scene_dict: Dict[str, Any] = dict(camera=config.camera)
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
        title_text=f"Dirac Hydrogen Wavefunction Visualization - {config.plot_type.capitalize() if config.plot_type != 'all' else 'All Components'}",
        width=config.width,
        height=config.height,
    )
    return fig


@dataclass
class SchrodingerPlotConfig:
    """Schrödinger 可视化配置

    Parameters
    ----------
    plot_type : str
        要绘制的图表类型: "density", "psi", 或 "all"
    density_scale : float
        密度等值面大小
    psi_scale : float
        波函数分量等值面大小
    clip : bool
        是否启用 phi 角度裁剪
    phi_min : float
        裁剪的 phi 角度最小值（度）
    phi_max : float
        裁剪的 phi 角度最大值（度）
    relative_density : bool
        True时使用相对于最大值的isovalue，False时使用绝对值（密度）
    relative_psi : bool
        True时使用相对于最大值的isovalue，False时使用绝对值（波函数）
    psi_component : str
        real 仅实部，imag 仅虚部，both 实部和虚部
    camera : dict
        3D 场景的相机设置
    use_squared : bool
        使用波函数模平方而非波函数模
    color : Tuple[str, str]
        密度的colorscale和波函数的colorscale
    xyzrange : Optional[Union[float, Tuple[float, float, float]]]
        轴范围，单个 float 表示三个轴相同范围，元组表示 (x, y, z) 范围，None 表示使用 aspectmode="data"
    width : int
        图像的宽度
    height : int
        图像的高度
    """

    plot_type: str = "all"  # New field to specify plot type
    density_scale: float = 0.01
    psi_scale: float = 0.1
    clip: bool = False
    phi_min: float = 0
    phi_max: float = 90
    relative_density: bool = True
    relative_psi: bool = True
    psi_component: Optional[str] = None
    use_squared: bool = False
    camera: dict = field(
        default_factory=lambda: dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        )
    )
    color: Tuple[str, str] = ("#a8c9f5", "hsv")
    xyzrange: Optional[Union[float, Tuple[float, float, float]]] = None
    width: int = 1000
    height: int = 600


def Schrodinger_plot(psi: np.ndarray, grid: gr.GridGenerator, config: SchrodingerPlotConfig):
    """可视化 Schrödinger 波函数

    Parameters
    ----------
    psi : np.ndarray
        Schrödinger ndarray 波函数
    grid : grids.GridGenerator
        grid.GridGenerator
    config : SchrodingerPlotConfig
        可视化参数配置

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly Figure
    """
    if config.plot_type == "all":
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}] * 2],
            subplot_titles=["Density", "Psi"],
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
        )
        subplot_map = {"density": (1, 1), "psi": (1, 2)}
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

    # 密度
    if config.plot_type in ["density", "all"]:
        density = np.abs(psi) ** 2
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

    # 波函数
    if config.plot_type in ["psi", "all"]:
        if config.psi_component == "real":
            magnitude = np.abs(np.real(psi))
        elif config.psi_component == "imag":
            magnitude = np.abs(np.imag(psi))
        else:
            magnitude = np.abs(psi)
        if config.use_squared:
            magnitude = magnitude**2
        max_mag = np.max(magnitude)
        isovalue_psi = config.psi_scale * max_mag if config.relative_psi else config.psi_scale

        if max_mag > 0:
            try:
                vertices, faces, _, _ = measure.marching_cubes(magnitude, level=isovalue_psi, spacing=(dx, dy, dz))
                vertices += [x_min, y_min, z_min]
                if config.clip:
                    faces = clip_faces_by_phi(vertices, faces, config.phi_min, config.phi_max)

                interp = RegularGridInterpolator((grid.x, grid.y, grid.z), psi)
                psi_vertices = interp(vertices)
                phase = np.angle(psi_vertices)

                if config.psi_component == "real":
                    phase = np.angle(np.real(psi_vertices))
                elif config.psi_component == "imag":
                    phase = np.angle(1j * np.imag(psi_vertices))
                else:
                    phase = np.angle(psi_vertices)

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
                        showscale=True,
                    ),
                    row=subplot_map["psi"][0],
                    col=subplot_map["psi"][1],
                )
            except Exception as e:
                print(f"Error processing wavefunction: {e}")

    for i in range(1, 3 if config.plot_type == "all" else 2):
        scene_dict: Dict[str, Any] = dict(camera=config.camera)
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

        fig.update_scenes(**scene_dict, row=1, col=i)

    fig.update_layout(
        title_text=f"Schrödinger Hydrogen Wavefunction Visualization - {config.plot_type.capitalize() if config.plot_type != 'all' else 'All Components'}",
        width=config.width,
        height=config.height,
    )

    return fig


@dataclass
class PlanePlotConfig:
    """
    平面波函数可视化配置

    Parameters
    ----------
    component : str
        要绘制的分量，'density' 或 'psi1' 到 'psi4'（Dirac）或 'psi'（Schrödinger）
    plot_type : str
        绘制类型，'magnitude'（绝对值）或 'phase'（相位）
    color : Tuple[str, str]
        强度和相位的colorscale，分别为 (density_color, phase_color)
    use_squared : bool
        使用波函数模平方而非波函数模
    is_3d : bool
        是否绘制3D表面图（True）或2D等高线图（False）
    contour_levels : int
        2D等高线图等高线数量
    camera : dict
        3D 场景的相机设置（仅在is_3d=True时使用）
    xyzrange : Optional[Union[float, Tuple[float, float, float]]]
        轴范围，单个 float 表示三个轴相同范围，元组表示 (x, y, z) 范围，None 表示使用 aspectmode="data"（仅在is_3d=True时使用）
    width : int
        图像的宽度
    height : int
        图像的高度
    """

    component: str = "density"
    plot_type: str = "magnitude"
    color: Tuple[str, str] = ("Blues", "hsv")
    use_squared: bool = False
    is_3d: bool = True
    contour_levels: int = 10
    camera: dict = field(
        default_factory=lambda: dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        )
    )
    xyzrange: Optional[Union[float, Tuple[float, float, float]]] = None
    width: int = 1000
    height: int = 900


def plot_plane_wavefunction(
    wavefunction: Union[wf.BaseSpinor, np.ndarray], grid: gr.PlaneGridGenerator, config: PlanePlotConfig
):
    """
    在指定平面上可视化预计算的波函数或密度，3D表面图或2D热图。

    Parameters
    ----------
    wavefunction : Union[BaseSpinor, np.ndarray]
        预计算的Dirac或Schrödinger波函数
    grid : PlaneGridGenerator
        平面网格生成器，包含X, Y, Z, U, V坐标
    config : PlanePlotConfig
        平面可视化配置

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly Figure
    """

    # 计算波函数或密度
    if isinstance(wavefunction, wf.BaseSpinor):
        # Dirac波函数
        psi = wavefunction.psi  # 形状为 (4, points_per_dim, points_per_dim)
        if config.component == "density":
            values = wavefunction.density()
        elif config.component in ["psi1", "psi2", "psi3", "psi4"]:
            idx = int(config.component[-1]) - 1
            values = np.abs(psi[idx]) if config.plot_type == "magnitude" else np.angle(psi[idx])
            if config.plot_type == "magnitude" and config.use_squared:
                values = values**2
        else:
            raise ValueError("Invalid component for BaseSpinor. Choose 'density' or 'psi1' to 'psi4'.")
    else:
        # Schrödinger波函数
        psi = wavefunction  # 形状为 (points_per_dim, points_per_dim)
        if config.component == "density":
            values = np.abs(psi) ** 2
        elif config.component == "psi":
            values = np.abs(psi) if config.plot_type == "magnitude" else np.angle(psi)
            if config.plot_type == "magnitude" and config.use_squared:
                values = values**2
        else:
            raise ValueError("Invalid component for Schrödinger wavefunction. Choose 'density' or 'psi'.")

    # 创建Plotly图形
    fig = go.Figure()

    # 选择颜色标度
    colorscale = (
        config.color[0] if config.plot_type == "magnitude" or config.component == "density" else config.color[1]
    )
    if config.plot_type == "phase":
        cmin, cmax = -np.pi, np.pi
    else:
        cmin, cmax = None, None

    if config.is_3d:
        # 3D表面图，使用X, Y, Z坐标
        fig.add_trace(
            go.Surface(
                x=grid.X,
                y=grid.Y,
                z=grid.Z,
                surfacecolor=values,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                colorbar=dict(title=config.plot_type.capitalize()),
            )
        )
        # 更新布局（3D）
        fig.update_layout(
            title_text=f"{'Dirac' if isinstance(wavefunction, wf.BaseSpinor) else 'Schrödinger'} Wavefunction - {config.component} ({config.plot_type})",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=config.camera,
                aspectmode="data" if config.xyzrange is None else "cube",
            ),
            width=config.width,
            height=config.height,
        )
        if config.xyzrange is not None:
            if isinstance(config.xyzrange, tuple):
                x_range, y_range, z_range = config.xyzrange
            else:
                x_range = y_range = z_range = config.xyzrange
            fig.update_scenes(
                xaxis=dict(range=[-x_range, x_range]),
                yaxis=dict(range=[-y_range, y_range]),
                zaxis=dict(range=[-z_range, z_range]),
            )
    else:
        # 2D热图，使用U, V坐标
        u = grid.v
        v = grid.u
        fig.add_trace(
            go.Contour(
                x=u,
                y=v,
                z=values,
                colorscale=colorscale,
                zmin=cmin,
                zmax=cmax,
                showscale=True,
                colorbar=dict(title=config.plot_type.capitalize()),
                contours=dict(
                    coloring="fill",
                    start=np.min(values),
                    end=np.max(values),
                    size=(np.max(values) - np.min(values)) / config.contour_levels,
                ),
                contours_coloring="heatmap",
            )
        )
        # 更新布局（2D）
        fig.update_layout(
            title_text=f"{'Dirac' if isinstance(wavefunction, wf.BaseSpinor) else 'Schrödinger'} Wavefunction - {config.component} ({config.plot_type})",
            xaxis_title="U",
            yaxis_title="V",
            width=config.width,
            height=config.height,
            xaxis=dict(range=[-grid.size / 2, grid.size / 2], scaleanchor="y", scaleratio=1, constrain="domain"),
            yaxis=dict(range=[-grid.size / 2, grid.size / 2], scaleratio=1),
        )

    return fig


@dataclass
class PlanePlotConfig3D:
    """
    三维平面波函数可视化配置

    Parameters
    ----------
    component : str
        要绘制的分量，'density' 或 'psi1' 到 'psi4'（Dirac）或 'psi'（Schrödinger）
    plot_type : str
        绘制类型，'magnitude'（绝对值）或 'phase'（相位）
    color : Tuple[str, str]
        强度和相位的colorscale，分别为 (density_color, phase_color)
    use_squared : bool
        使用波函数模平方而非波函数模
    show_contours : bool
        是否显示等高线
    contour_levels : int
        等高线数量
    camera : dict
        3D 场景的相机设置
    xyzrange : Optional[Union[float, Tuple[float, float, float]]]
        轴范围，单个 float 表示三个轴相同范围，元组表示 (x, y, z) 范围，None 表示使用 aspectmode="data"
    width : int
        图像的宽度
    height : int
        图像的高度
    """

    component: str = "density"
    plot_type: str = "magnitude"
    color: Tuple[str, str] = ("Blues", "hsv")
    use_squared: bool = False
    show_contours: bool = True
    contour_levels: int = 10
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


def plot_plane_wavefunction_3D(
    wavefunction: Union[wf.BaseSpinor, np.ndarray], grid: gr.PlaneGridGenerator, config: PlanePlotConfig3D
):
    """
    在xy平面上以三维方式可视化预计算的波函数或密度，高度由密度或波函数模确定，并显示等高线。

    Parameters
    ----------
    wavefunction : Union[BaseSpinor, np.ndarray]
        预计算的Dirac或Schrödinger波函数
    grid : PlaneGridGenerator
        平面网格生成器，包含X, Y, Z坐标
    config : PlanePlotConfig3D
        三维平面可视化配置

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly Figure
    """
    # 获取网格坐标，并将Z设为零（xy平面）
    X, Y = grid.U, grid.V

    # 计算波函数或密度
    if isinstance(wavefunction, wf.BaseSpinor):
        # Dirac波函数
        psi = wavefunction.psi  # 形状为 (4, points_per_dim, points_per_dim)
        if config.component == "density":
            values = wavefunction.density()
            height = values  # 高度为密度
        elif config.component in ["psi1", "psi2", "psi3", "psi4"]:
            idx = int(config.component[-1]) - 1
            values = np.abs(psi[idx]) if config.plot_type == "magnitude" else np.angle(psi[idx])
            height = np.abs(psi[idx])  # 高度为模
            if config.use_squared:
                height = height**2
        else:
            raise ValueError("Invalid component for BaseSpinor. Choose 'density' or 'psi1' to 'psi4'.")
    else:
        # Schrödinger波函数
        psi = wavefunction  # 形状为 (points_per_dim, points_per_dim)
        if config.component == "density":
            values = np.abs(psi) ** 2
            height = values  # 高度为密度
        elif config.component == "psi":
            values = np.abs(psi) if config.plot_type == "magnitude" else np.angle(psi)
            height = np.abs(psi)  # 高度为模
            if config.use_squared:
                height = height**2
        else:
            raise ValueError("Invalid component for Schrödinger wavefunction. Choose 'density' or 'psi'.")

    # 创建Plotly图形
    fig = go.Figure()

    # 选择颜色标度
    colorscale = (
        config.color[0] if config.plot_type == "magnitude" or config.component == "density" else config.color[1]
    )
    if config.plot_type == "phase":
        cmin, cmax = -np.pi, np.pi
    else:
        cmin, cmax = None, None

    # 添加表面图，高度由密度或模确定
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=height,  # 使用密度或模作为z轴高度
            surfacecolor=values,  # 颜色由密度/模或相位确定
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=True,
            colorbar=dict(title=config.plot_type.capitalize()),
            contours=dict(
                z=dict(
                    show=config.show_contours,
                    usecolormap=True if config.plot_type == "magnitude" else False,
                    project_z=True,  # 在xy平面投影等高线
                    start=np.min(height),
                    end=np.max(height),
                    size=(np.max(height) - np.min(height)) / config.contour_levels,
                )
            ),
        )
    )

    # 更新布局
    fig.update_layout(
        title_text=f"{'Dirac' if isinstance(wavefunction, wf.BaseSpinor) else 'Schrödinger'} Wavefunction 3D - {config.component} ({config.plot_type})",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Height (Magnitude)", camera=config.camera, aspectmode="cube"
        ),
        width=config.width,
        height=config.height,
    )

    if config.xyzrange is not None:
        if isinstance(config.xyzrange, tuple):
            x_range, y_range, z_range = config.xyzrange
        else:
            x_range = y_range = z_range = config.xyzrange
        fig.update_scenes(
            xaxis=dict(range=[-x_range, x_range]),
            yaxis=dict(range=[-y_range, y_range]),
            zaxis=dict(range=[-z_range, z_range]),
        )

    return fig


@dataclass
class VolumePlotConfig:
    """
    体视化配置

    Parameters
    ----------
    component : str
        要绘制的分量，'density' 或 'psi1' 到 'psi4'（Dirac）或 'psi'（Schrödinger）
    color : str
        密度或模的颜色标度
    use_squared : bool
        使用波函数模平方而非波函数模
    clip : bool
        是否启用 phi 角度裁剪
    phi_min : float
        裁剪的 phi 角度最小值（度）
    phi_max : float
        裁剪的 phi 角度最大值（度）
    opacity : float
        体视化的透明度，范围 [0, 1]
    isovalue : float
        体视化的等值面值（相对于最大值或绝对值）
    relative_isovalue : bool
        True 时使用相对于最大值的 isovalue，False 时使用绝对值
    surface_count:
        体视化的等值面数量
    camera : dict
        3D 场景的相机设置
    xyzrange : Optional[Union[float, Tuple[float, float, float]]]
        轴范围，单个 float 表示三个轴相同范围，元组表示 (x, y, z) 范围，None 表示使用 aspectmode="data"
    width : int
        图像的宽度
    height : int
        图像的高度
    """

    component: str = "density"
    color: str = "Blues"
    use_squared: bool = False
    clip: bool = False
    phi_min: float = 0
    phi_max: float = 90
    opacity: float = 0.5
    isovalue: float = 0.1
    relative_isovalue: bool = True
    surface_count: int = 20
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


def plot_volume_wavefunction(
    wavefunction: Union["wf.BaseSpinor", np.ndarray], grid: "gr.GridGenerator", config: VolumePlotConfig
):
    """
    在三维空间中以体视化的方式可视化预计算的波函数或密度。

    Parameters
    ----------
    wavefunction : Union[BaseSpinor, np.ndarray]
        预计算的Dirac或Schrödinger波函数
    grid : GridGenerator
        体网格生成器，包含X, Y, Z坐标
    config : VolumePlotConfig
        体视化配置

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly Figure
    """
    # 获取网格坐标
    X, Y, Z = grid.X, grid.Y, grid.Z

    # 计算波函数或密度
    if isinstance(wavefunction, wf.BaseSpinor):
        psi = wavefunction.psi  # 形状为 (4, nx, ny, nz)
        if config.component == "density":
            values = wavefunction.density()
        elif config.component in ["psi1", "psi2", "psi3", "psi4"]:
            idx = int(config.component[-1]) - 1
            values = np.abs(psi[idx])
            if config.use_squared:
                values = values**2
        else:
            raise ValueError("Invalid component for BaseSpinor. Choose 'density' or 'psi1' to 'psi4'.")
    else:
        psi = wavefunction  # 形状为 (nx, ny, nz)
        if config.component == "density":
            values = np.abs(psi) ** 2
        elif config.component == "psi":
            values = np.abs(psi)
            if config.use_squared:
                values = values**2
        else:
            raise ValueError("Invalid component for Schrödinger wavefunction. Choose 'density' or 'psi'.")

    # 裁剪 phi 角度范围
    if config.clip:
        values_clipped = np.copy(values)
        phi = np.degrees(np.arctan2(Y, X))
        phi = (phi + 360) % 360
        mask = create_phi_mask(phi, config.phi_min, config.phi_max)
        values_clipped[mask] = 0
        values = values_clipped

    # 确定 isovalue
    max_value = np.max(values)
    isovalue = config.isovalue * max_value if config.relative_isovalue else config.isovalue

    # 创建体视化
    fig = go.Figure()
    fig.add_trace(
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            isomin=isovalue,
            isomax=max_value,
            opacity=config.opacity,
            surface_count=config.surface_count,
            colorscale=config.color,
            showscale=True,
            colorbar=dict(title="Magnitude"),
        )
    )

    # 设置布局
    fig.update_layout(
        title_text=f"{'Dirac' if isinstance(wavefunction, wf.BaseSpinor) else 'Schrödinger'} Wavefunction Volume - {config.component}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=config.camera,
            aspectmode="cube" if config.xyzrange is not None else "data",
        ),
        width=config.width,
        height=config.height,
    )

    if config.xyzrange is not None:
        if isinstance(config.xyzrange, tuple):
            x_range, y_range, z_range = config.xyzrange
        else:
            x_range = y_range = z_range = config.xyzrange
        fig.update_scenes(
            xaxis=dict(range=[-x_range, x_range]),
            yaxis=dict(range=[-y_range, y_range]),
            zaxis=dict(range=[-z_range, z_range]),
        )

    return fig
