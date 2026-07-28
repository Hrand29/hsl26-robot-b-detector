"""Чистые numpy-функции обработки облака точек, без зависимости от ROS.

Используются и из detector_node.py, и из офлайн-скриптов для быстрой
проверки на bag-данных.
"""

import numpy as np


def fit_ground_plane(points, iterations=150, inlier_thresh=0.02, min_extent=1.0, rng=None):
    """RANSAC-фит плоскости пола. points — Nx3 в кадре base_link.

    Возвращает (normal, d) для уравнения normal @ p + d = 0, либо None.
    """
    if rng is None:
        rng = np.random.default_rng()

    # пол в base_link лежит у z~0 (сенсор поднят на известную высоту, см.
    # tf_static) - узкая полоса вместо широкой отсекает плоскую деку робота Б:
    # когда Б близко к сенсору, его точки становятся плотными и деку иначе
    # можно принять за пол
    candidates = points[np.abs(points[:, 2]) < 0.08]
    if len(candidates) < 3:
        candidates = points

    n = len(candidates)
    best_inliers = -1
    best_plane = None
    best_mask = None
    for _ in range(iterations):
        p0, p1, p2 = candidates[rng.choice(n, size=3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        d = -normal @ p0

        dist = np.abs(points @ normal + d)
        mask = dist < inlier_thresh
        inliers = int(np.count_nonzero(mask))
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal, d)
            best_mask = mask

    if best_plane is None:
        return None

    # настоящий пол покрывает комнату, а не маленький локальный пятачок
    # (например, деку робота) - отсекаем такие ложные фиты по площади
    inlier_pts = points[best_mask]
    extent = min(inlier_pts[:, 0].ptp(), inlier_pts[:, 1].ptp())
    if extent < min_extent:
        return None

    return best_plane


def remove_ground(points, plane, band=0.02):
    """Снимает только тонкую полосу вокруг плоскости пола (не диапазон высот)."""
    normal, d = plane
    dist = np.abs(points @ normal + d)
    return points[dist >= band]
