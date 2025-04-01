import warnings
from typing import Optional

import numpy as np
import shapely

try:
    from . import fovc

    has_fovc_module = True
except ImportError:
    warnings.warn("Cython FoV extension not available, slow Python version will be used instead")
    has_fovc_module = False


def fov2d_poly(p: np.ndarray, alpha: float, r: float, theta: float, n: int) -> shapely.Polygon:
    """! Create 2D field-of-view polygon.

    @param p Camera position.
    @param alpha Camera angle, clockwise from positive y axis [rad].
    @param r FoV radius.
    @param theta FoV angle [rad].
    @param n Number of points (≥ 3).
    @return The FoV polygon.
    """
    points = [(p[0], p[1])]
    for i in range(n - 1):
        a = alpha - theta / 2.0 + i / (n - 2.0) * theta
        points.append((p[0] + r * np.sin(a), p[1] + r * np.cos(a)))
    points.append((p[0], p[1]))
    return shapely.Polygon(points)


def fov2d_overlap(
    p1: np.ndarray, a1: float, p2: np.ndarray, a2: float, r: float = 50.0, theta: float = np.radians(90.0), n: int = 32
) -> float:
    """! Compute 2D field-of-view overlap.

    @see "Generalized Contrastive Optimization of Siamese Networks for Place Recognition" by Leyva-Vallina et al.

    @note The radius check in this function is incorrect (it should be "np.linalg.norm(p1 - p2) > 2 * r"). This holds
          also for the other methods in this file and for the Cython module (fovc.pyx). As the bug was discovered after
          training all the models it wasn't fixed. Testing suggests that fixing it would not improve model performance.

    @param p1 Position of first camera.
    @param a1 Angle of first camera, clockwise from positive y axis [rad].
    @param p2 Position of second camera.
    @param a2 Angle of second camera, clockwise from positive y axis [rad].
    @param r FoV radius.
    @param theta FoV angle [rad].
    @param n Number of points in the FoV polygons (≥ 3).
    @return The normalized FoV overlap ([0, 1] range).
    """
    if np.linalg.norm(p1 - p2) > r:
        return 0.0

    poly1 = fov2d_poly(p1 - p1, a1, r, theta, n)
    poly2 = fov2d_poly(p2 - p1, a2, r, theta, n)

    return poly1.intersection(poly2).area / poly1.area


def fov2d_overlap_list(
    p1: np.ndarray,
    a1: float,
    p2: np.ndarray,
    a2: np.ndarray,
    r: float = 50.0,
    theta: float = np.radians(90.0),
    n: int = 16,
) -> np.ndarray:
    """! Compute FoV overlap between one camera and a list of other cameras.

    @param p1 Position of first camera (2).
    @param a1 Angle of first camera.
    @param p2 Position of other camera (n x 2).
    @param a2 Angles of other cameras (n).
    @param r FoV radius.
    @param theta FoV angle [rad].
    @param n Number of points in the FoV polygons (≥ 3).
    @return FoV overlaps (n).
    """
    if has_fovc_module:
        return fovc.fov2d_overlap_list(p1, a1, p2, a2, r, theta, n)

    dist = np.linalg.norm(p1 - p2, axis=1)
    overlap = np.zeros(p2.shape[0], dtype=np.float32)
    poly1 = fov2d_poly(p1, a1, r, theta, n)
    area = poly1.area

    for i in range(p2.shape[0]):
        if dist[i] > r:
            continue
        poly2 = fov2d_poly(p2[i], a2[i], r, theta, n)
        overlap[i] = poly1.intersection(poly2).area / area

    return overlap


def fov2d_overlap_pairs(
    pos: np.ndarray,
    ang: np.ndarray,
    r: float = 50.0,
    theta: float = np.radians(90.0),
    n: int = 16,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """! Compute FoV overlap between all camera pairs.

    @param pos Camera positions (n x 2).
    @param ang Image compass angle (n).
    @param r FoV radius.
    @param theta FoV angle [rad].
    @param n Number of points in the FoV polygons (≥ 3).
    @param mask Optional mask array where mask[i, j] = mask[j, i] indicates whether overlap should be computed for
                camera pair i, j (n x n).
    @return Pair-wise FoV overlap (n x n).
    """
    if has_fovc_module:
        return fovc.fov2d_overlap_pairs(pos, ang, r, theta, n, mask)

    mask_ = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1) <= r
    if mask is not None:
        mask_ &= mask

    overlap = np.zeros((pos.shape[0], pos.shape[0]), dtype=np.float32)
    np.fill_diagonal(overlap, 1.0)
    polys = np.zeros(pos.shape[0], dtype=np.object_)

    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if mask_[i, j]:
                if not polys[i]:
                    polys[i] = fov2d_poly(pos[i], ang[i], r, theta, n)
                if not polys[j]:
                    polys[j] = fov2d_poly(pos[j], ang[j], r, theta, n)
                overlap[i, j] = overlap[j, i] = polys[i].intersection(polys[j]).area / polys[i].area

    return overlap
