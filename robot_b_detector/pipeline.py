"""Чистые numpy-функции обработки облака точек, без зависимости от ROS.

Используются и из detector_node.py, и из офлайн-скриптов для быстрой
проверки на bag-данных.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROBOT_BASE_DIAMETER = 0.35


def fit_ground_plane(points, iterations=150, inlier_thresh=0.02, min_extent=1.0,
                      min_normal_z=0.95, rng=None):
    """RANSAC-фит плоскости пола. points — Nx3 в кадре base_link.

    Возвращает (normal, d) для уравнения normal @ p + d = 0, либо None.
    """
    if rng is None:
        rng = np.random.default_rng()

    # пол лежит у z~0; узкая полоса (не широкая) не даёт спутать с плоской
    # декой робота Б, когда он близко и даёт много точек
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

    # пол горизонтален - наклонная нормаль означает случайный фит по шуму
    normal, _ = best_plane
    if abs(normal[2]) < min_normal_z:
        return None

    # настоящий пол покрывает комнату, а не локальный пятачок вроде деки
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


def voxel_downsample(points, voxel_size=0.03):
    """Ограничивает плотность точек равномерно по облаку (1 точка/воксель).

    Нужно перед кластеризацией: при накоплении нескольких кадров стены и
    другие большие плоскости дают огромное число близких пар точек для
    query_pairs, без даунсэмплинга это не масштабируется по памяти.
    """
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


def cluster_points(points, radius=0.1, min_points=10):
    """Евклидова кластеризация через KD-дерево + граф связности (scipy)."""
    n = len(points)
    if n == 0:
        return []

    tree = cKDTree(points)
    pairs = tree.query_pairs(r=radius, output_type='ndarray')
    if len(pairs) == 0:
        return []

    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    adjacency = coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))
    _, labels = connected_components(adjacency, directed=False)

    clusters = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if len(idx) >= min_points:
            clusters.append(points[idx])
    return clusters


def fit_circle_2d(points_xy, iterations=100, inlier_thresh=0.02, rng=None):
    """RANSAC-фит окружности в плоскости XY по 3 точкам (координаты центра -
    решение системы через circumcenter). Возвращает (center, radius,
    inlier_ratio), либо None.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(points_xy)
    if n < 3:
        return None

    best = None
    best_count = -1
    for _ in range(iterations):
        (ax, ay), (bx, by), (cx, cy) = points_xy[rng.choice(n, size=3, replace=False)]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-9:
            continue  # точки почти на одной прямой - окружность не определена
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
              + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
              + (cx**2 + cy**2) * (bx - ax)) / d
        center = np.array([ux, uy])
        radius = np.linalg.norm(np.array([ax, ay]) - center)

        dist = np.abs(np.linalg.norm(points_xy - center, axis=1) - radius)
        count = int(np.count_nonzero(dist < inlier_thresh))
        if count > best_count:
            best_count = count
            best = (center, radius)

    if best is None:
        return None
    center, radius = best
    return center, radius, best_count / n


def density_above_count(points, center_xy, search_radius=0.3, height_range=(0.2, 0.45)):
    """Сырое число точек рядом с кандидатом в заданном диапазоне высот -
    вынесено отдельно от has_low_density_above для диагностики/логирования."""
    d = np.hypot(points[:, 0] - center_xy[0], points[:, 1] - center_xy[1])
    nearby_z = points[d < search_radius, 2]
    return int(np.count_nonzero((nearby_z > height_range[0]) & (nearby_z < height_range[1])))


def select_robot_candidate(clusters, points, expected_diameter=ROBOT_BASE_DIAMETER,
                            diameter_tolerance=0.2, max_floor_gap=0.05, max_height=0.45,
                            expected_radius=ROBOT_BASE_DIAMETER / 2, radius_tolerance=0.08,
                            max_density_ratio=0.8, rng=None, return_debug=False):
    """Выбирает из кластеров основание робота Б: касается пола, диаметр и
    радиус (через RANSAC-окружность, надёжнее доли инлаеров - сенсор часто
    видит только часть дуги) соответствуют роботу, а density/n рядом с
    кандидатом не выдаёт сплошную поверхность вроде стены (отношение, а не
    абсолютное число точек - устойчивее к дистанции). Среди прошедших все
    фильтры берём самый плотный по числу точек.

    points - полное (недаунсэмпленное) облако без пола, нужно для проверки
    плотности.
    """
    best = None
    best_score = None
    debug_info = []
    for cluster in clusters:
        xy = cluster[:, :2]
        extent = max(xy[:, 0].ptp(), xy[:, 1].ptp())
        if abs(extent - expected_diameter) > diameter_tolerance:
            continue
        if cluster[:, 2].min() > max_floor_gap:
            continue  # не касается пола - не робот
        if cluster[:, 2].max() > max_height:
            continue  # слишком высокий для основания робота (например, колонна)

        circle = fit_circle_2d(xy, rng=rng)
        if circle is None:
            continue
        _, radius, _ = circle
        if abs(radius - expected_radius) > radius_tolerance:
            continue  # не тот радиус - например, угол стены даёт совсем другую кривизну

        center = (xy[:, 0].mean(), xy[:, 1].mean())
        density_count = density_above_count(points, center)
        density_ratio = density_count / len(cluster)
        density_ok = density_ratio <= max_density_ratio
        if return_debug:
            debug_info.append({'center': center, 'n': len(cluster), 'density_ok': density_ok,
                                'density_count': density_count, 'density_ratio': density_ratio})
        if not density_ok:
            continue  # сплошная поверхность рядом - не робот

        score = len(cluster)
        if best_score is None or score > best_score:
            best = cluster
            best_score = score

    if return_debug:
        return best, debug_info
    return best


def estimate_xy(cluster):
    """Центроид кластера в плоскости XY - оценка позиции робота Б."""
    return cluster[:, 0].mean(), cluster[:, 1].mean()
