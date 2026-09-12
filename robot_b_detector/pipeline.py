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
    for _ in range(iterations):
        p0, p1, p2 = candidates[rng.choice(n, size=3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        d = -normal @ p0

        # скорим инлаеры по узкой прифутольной полосе (candidates), а не по
        # всему накопленному облаку - на порядок дешевле за итерацию, а
        # порог inlier_thresh=0.02 всё равно отсекает всё, что не у пола
        dist = np.abs(candidates @ normal + d)
        inliers = int(np.count_nonzero(dist < inlier_thresh))
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal, d)

    if best_plane is None:
        return None

    # пол горизонтален - наклонная нормаль означает случайный фит по шуму
    normal, d = best_plane
    if abs(normal[2]) < min_normal_z:
        return None

    # настоящий пол покрывает комнату, а не локальный пятачок вроде деки -
    # полный проход по points нужен только один раз, для победившей плоскости
    full_mask = np.abs(points @ normal + d) < inlier_thresh
    inlier_pts = points[full_mask]
    if len(inlier_pts) == 0:
        return None
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

    Ключ вокселя закодирован в один int64 (не строка (N,3) для
    np.unique(axis=0)) - построчный unique по 2D-массиву в numpy на порядок
    медленнее (внутри сортирует по byte-view всей строки), скалярный ключ
    даёт тот же результат через быстрый 1D-путь.
    """
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    # смещение в положительную область на случай отрицательных координат,
    # затем упаковка xyz в один int64 (по ~20 бит на ось - с запасом для
    # полигона ~6x6м при voxel_size=0.03: диапазон вокселей ±200)
    keys += 1 << 19
    packed = (keys[:, 0] << 40) | (keys[:, 1] << 20) | keys[:, 2]
    _, idx = np.unique(packed, return_index=True)
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


def touches_floor_nearby(points, center_xy, search_radius=0.3, max_floor_gap=0.05):
    """Есть ли у пола точки рядом с кандидатом - по ВСЕМУ накопленному
    облаку (points), а не только внутри одного кластера.

    У близкого (0.3-0.7м) круглого объекта лидар на мачте видит его нижнюю
    кромку почти по касательной - там реальный, физический провал плотности
    скана (не баг), из-за которого euclidean-кластеризация (cluster_points)
    иногда рвёт связность между этой кромкой и остальным объектом, и
    floor-точки остаются вне кластера-кандидата. Проверка по облаку целиком,
    а не по составу кластера, устойчива к этому разрыву - тот же приём, что
    уже применяется в density_above_count.
    """
    d = np.hypot(points[:, 0] - center_xy[0], points[:, 1] - center_xy[1])
    nearby_z = points[d < search_radius, 2]
    return bool(np.any(nearby_z <= max_floor_gap)) if len(nearby_z) else False


def select_robot_candidate(clusters, points, expected_diameter=ROBOT_BASE_DIAMETER,
                            diameter_tolerance=0.2, max_floor_gap=0.05, max_height=0.45,
                            expected_radius=ROBOT_BASE_DIAMETER / 2, radius_tolerance=0.08,
                            max_density_ratio=None, max_aspect_ratio=5.0,
                            circle_iterations=30, rng=None, return_debug=False):
    """Выбирает из кластеров основание робота Б: касается пола, диаметр и
    радиус (через RANSAC-окружность, надёжнее доли инлаеров - сенсор часто
    видит только часть дуги) соответствуют роботу, а density/n рядом с
    кандидатом не выдаёт сплошную поверхность вроде стены (отношение, а не
    абсолютное число точек - устойчивее к дистанции). Среди прошедших все
    фильтры берём самый плотный по числу точек.

    points - полное (недаунсэмпленное) облако без пола, нужно для проверки
    плотности.

    Кластеры сортируются по размеру по убыванию - т.к. итоговый скор это
    len(cluster), первый прошедший все фильтры кандидат гарантированно
    лучший, и без return_debug можно выйти сразу, не считая RANSAC-окружность
    (самая дорогая проверка) для остальных кластеров. aspect_ratio - дешёвая
    проверка формы bbox ДО RANSAC: вытянутые кластеры (сегмент стены, угол
    препятствия) отсекаются до дорогого circle-fit, а не после него.
    """
    ordered = sorted(clusters, key=len, reverse=True)
    best = None
    debug_info = []
    for cluster in ordered:
        xy = cluster[:, :2]
        extent_x = xy[:, 0].ptp()
        extent_y = xy[:, 1].ptp()
        extent = max(extent_x, extent_y)
        if abs(extent - expected_diameter) > diameter_tolerance:
            continue
        if cluster[:, 2].max() > max_height:
            continue  # слишком высокий для основания робота (например, колонна)
        # порог намеренно высокий (5.0): при виде только части дуги (обычная
        # ситуация на близкой дистанции) bbox цели сам по себе вытянут
        # (живой тест на цилиндре-мишени в base_link: aspect~2.1) - фильтр
        # только для действительно линейных объектов вроде сегмента стены
        # (живые данные lab1: aspect 19-133), а не финальная проверка формы -
        # ту делает RANSAC-окружность ниже, она устойчива к частичной дуге
        aspect_ratio = extent / max(min(extent_x, extent_y), 1e-6)
        if aspect_ratio > max_aspect_ratio:
            continue  # вытянутый bbox - не круглое основание (сегмент стены/угол)

        circle = fit_circle_2d(xy, iterations=circle_iterations, rng=rng)
        if circle is None:
            continue
        circle_center, radius, _ = circle
        if abs(radius - expected_radius) > radius_tolerance:
            continue  # не тот радиус - например, угол стены даёт совсем другую кривизну

        # проверка "касается пола" - по всему облаку рядом с ЦЕНТРОМ ОКРУЖНОСТИ
        # (не центроидом кластера - тот смещён к видимой дуге), см. docstring
        # touches_floor_nearby: сам кластер может не содержать floor-точки
        # из-за разрыва связности у основания, это не значит, что их нет
        if not touches_floor_nearby(points, circle_center, max_floor_gap=max_floor_gap):
            continue  # рядом с кандидатом нет точек у пола - не робот

        # density_above_count сравнивает точки рядом с кандидатом с числом
        # точек В ЭТОМ КЛАСТЕРЕ - но euclidean-кластеризация дробит близкий
        # (0.3-0.7м) круглый объект на несколько несвязных кусков (та же
        # причина, что и разрыв у пола выше - неравномерная плотность скана
        # на кривизне), так что "кластер" тут - только фрагмент, а не вся
        # цель. Живой тест: ratio=37.6 на настоящей цели, а не на стене -
        # ложный отказ, не защита. Форму уже надёжно проверяют diameter/
        # aspect/radius выше (стены в HSL26 отсекаются ими, не этим), и по
        # регламенту в лабиринте кроме стен и второго робота ничего нет -
        # отдельная проверка "не сплошная поверхность" избыточна и опаснее,
        # чем её отсутствие. Оставлена выключаемой (max_density_ratio=None
        # по умолчанию = не применяется), density_ratio всё равно считается
        # и идёт в debug_info - пригодится, если понадобится откалибровать
        # заново на другой геометрии
        center = (xy[:, 0].mean(), xy[:, 1].mean())
        density_count = density_above_count(points, center)
        density_ratio = density_count / len(cluster)
        density_ok = max_density_ratio is None or density_ratio <= max_density_ratio
        if return_debug:
            debug_info.append({'center': center, 'n': len(cluster), 'density_ok': density_ok,
                                'density_count': density_count, 'density_ratio': density_ratio})
        if not density_ok:
            continue  # сплошная поверхность рядом - не робот

        if best is None:
            best = cluster
            if not return_debug:
                break  # ordered по убыванию n - дальше только кластеры хуже

    if return_debug:
        return best, debug_info
    return best


def estimate_xy(cluster):
    """Центроид кластера в плоскости XY - оценка позиции робота Б."""
    return cluster[:, 0].mean(), cluster[:, 1].mean()
