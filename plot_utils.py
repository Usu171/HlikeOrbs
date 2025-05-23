import numpy as np


def create_phi_mask(phi, phi_min, phi_max):
    if phi_min < phi_max:
        return (phi >= phi_min) & (phi <= phi_max)
    else:
        return (phi >= phi_min) | (phi <= phi_max)


def clip_faces_by_phi(vertices, faces, phi_min, phi_max):
    phi = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    phi = (phi + 360) % 360

    mask = create_phi_mask(phi, phi_min, phi_max)

    face_in_range = np.all(mask[faces], axis=1)
    faces_clipped = faces[~face_in_range]

    return faces_clipped


def clip_current_by_phi(X, Y, Z, current, phi_min, phi_max):
    phi = np.degrees(np.arctan2(Y, X))
    phi = (phi + 360) % 360

    mask = create_phi_mask(phi, phi_min, phi_max)

    current_clipped = np.copy(current)
    current_clipped[:, mask] = 0

    return current_clipped


def clip_points_by_phi(points, phi_min, phi_max, phases=None, probs=None):
    """
    根据 phi 角度裁剪点集

    Parameters
    ----------
    points : np.ndarray
        形状为 (n, 3) 的点坐标数组
    phi_min : float
        裁剪的 phi 角度最小值（度）
    phi_max : float
        裁剪的 phi 角度最大值（度）
    phases : np.ndarray, optional
        形状为 (n,) 的相位数组，默认为 None
    probs : np.ndarray, optional
        形状为 (n,) 的概率密度数组，默认为 None

    Returns
    -------
    points_clipped : np.ndarray
        裁剪后的点坐标数组
    phases_clipped : np.ndarray or None
        裁剪后的相位数组（如果提供）
    probs_clipped : np.ndarray or None
        裁剪后的概率密度数组（如果提供）
    """
    phi = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    phi = (phi + 360) % 360

    mask = create_phi_mask(phi, phi_min, phi_max)

    points_clipped = points[~mask]
    phases_clipped = phases[~mask] if phases is not None else None
    probs_clipped = probs[~mask] if probs is not None else None

    return points_clipped, phases_clipped, probs_clipped
