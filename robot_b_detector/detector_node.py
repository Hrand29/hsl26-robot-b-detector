"""ROS2-нода: подписывается на /livox/lidar и публикует debug-топики.

Удаление пола: плоскость пересчитывается не на каждом кадре, а раз в
RECALIBRATION_PERIOD_SEC по накопленным точкам - иначе разные RANSAC-фиты
вместе с Decay Time в RViz визуально "отращивают" пол обратно.

Отбор кандидата-робота: кластеризация по скользящему окну
ACCUMULATION_WINDOW_SEC, не по одному кадру - у Livox неповторяющийся
паттерн сканирования, один кадр слишком разреженный на дальних дистанциях.

Позиция: центроид кандидата -> маркеры (сфера + текстовая подпись) на
/debug/position. Публикуем только если у кандидата достаточно точек
(MIN_CONFIDENT_POINTS) - на меньшем n (частый случай при окклюзии в
движении) позиция ненадёжна, честнее промолчать, чем показать её как есть.
"""

import collections

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from visualization_msgs.msg import Marker, MarkerArray

from robot_b_detector import pipeline

TARGET_FRAME = 'base_link'
RECALIBRATION_PERIOD_SEC = 3.0
ACCUMULATION_WINDOW_SEC = 1.5
MIN_CONFIDENT_POINTS = 50
# кластеризация + RANSAC-окружность - самая дорогая часть конвейера, не
# должна пересчитываться на каждый кадр лидара (8-30Гц). Скорость погони
# ограничена, 5Гц позиции хватает с запасом - и держит CPU-бюджет узла
# ограниченным независимо от частоты лидара, когда рядом крутится SLAM/Nav2
DETECTION_PERIOD_SEC = 0.2
# основание робота Б не выше max_height в select_robot_candidate (0.45) -
# обрезаем раньше, до накопления в буфер, чтобы кластеризация не тратила
# время на верхнюю часть стен/препятствий (высота 1м в лабиринте HSL26)
CROP_MAX_HEIGHT = 0.5
# лидар на своей же мачте видит и собственную платформу (mast.urdf.xacro:
# 0.25x0.25м на z~0.29 - диагональ ~0.177м от base_link) - без фильтра эти
# self-хиты (живой тест: плотный blob вплоть до 0.06м от начала координат)
# засоряют density_above_count в select_robot_candidate, т.к. попадают в её
# height_range=(0.2,0.45) и рядом с любым близким кандидатом дают ложное
# "сплошная поверхность = не робот". Живой тест показал чистый разрыв между
# self-хитами и целью (пусто на 0.06-0.22м) - 0.2м отсекает свою платформу
# с запасом и не задевает цель даже на минимальной дистанции поимки по
# регламенту (0.45м между центрами, ближняя точка цели радиусом 0.175м
# оказалась бы на ~0.275м от base_link - ещё выше порога)
SELF_FILTER_RADIUS = 0.2


class DetectorNode(Node):
    def __init__(self):
        super().__init__('robot_b_detector')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        lidar_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(PointCloud2, '/livox/lidar', self.on_cloud, lidar_qos)
        self.floor_removed_pub = self.create_publisher(PointCloud2, '/debug/floor_removed', 10)
        self.candidate_pub = self.create_publisher(PointCloud2, '/debug/candidate_cluster', 10)
        self.position_pub = self.create_publisher(MarkerArray, '/debug/position', 10)

        self._ground_plane = None
        self._calibration_buffer = []
        self._last_calibration = self.get_clock().now()
        self._last_detection = self.get_clock().now()
        self._accum_buffer = collections.deque()  # [(t_sec, points), ...]

    def on_cloud(self, msg: PointCloud2):
        try:
            # transform статический -> берём последнюю доступную, не привязываемся к stamp
            transform = self.tf_buffer.lookup_transform(TARGET_FRAME, msg.header.frame_id, Time())
        except Exception as exc:
            self.get_logger().warn(f'tf lookup failed: {exc}', throttle_duration_sec=2.0)
            return

        cloud_base = do_transform_cloud(msg, transform)
        # read_points_numpy падает на смешанных типах полей во всём сообщении
        # (баг в реализации Humble), поэтому берём x,y,z через read_points
        structured = point_cloud2.read_points(cloud_base, field_names=('x', 'y', 'z'))
        points = point_cloud2.structured_to_unstructured(structured)
        if len(points) < 3:
            return

        self._calibration_buffer.append(points)
        now = self.get_clock().now()
        elapsed = (now - self._last_calibration).nanoseconds / 1e9
        if elapsed >= RECALIBRATION_PERIOD_SEC:
            combined = np.concatenate(self._calibration_buffer)
            plane = pipeline.fit_ground_plane(combined)
            self._calibration_buffer = []
            self._last_calibration = now
            if plane is not None:
                self._ground_plane = plane

        if self._ground_plane is None:
            return  # ещё не откалибровались

        remaining = pipeline.remove_ground(points, self._ground_plane)
        out = point_cloud2.create_cloud_xyz32(cloud_base.header, remaining)
        self.floor_removed_pub.publish(out)

        now_sec = now.nanoseconds / 1e9
        # обрезка высоты + self-filter - только для буфера кластеризации,
        # /debug/floor_removed выше публикует remaining целиком, без обрезки
        cropped = remaining[remaining[:, 2] < CROP_MAX_HEIGHT]
        radial = np.hypot(cropped[:, 0], cropped[:, 1])
        cropped = cropped[radial > SELF_FILTER_RADIUS]
        self._accum_buffer.append((now_sec, cropped))
        while now_sec - self._accum_buffer[0][0] > ACCUMULATION_WINDOW_SEC:
            self._accum_buffer.popleft()

        elapsed_detection = (now - self._last_detection).nanoseconds / 1e9
        if elapsed_detection < DETECTION_PERIOD_SEC:
            return
        self._last_detection = now

        accumulated = np.concatenate([p for _, p in self._accum_buffer])
        down = pipeline.voxel_downsample(accumulated)
        clusters = pipeline.cluster_points(down)
        candidate, debug_info = pipeline.select_robot_candidate(
            clusters, accumulated, return_debug=True)

        confident = candidate is not None and len(candidate) >= MIN_CONFIDENT_POINTS
        if candidate is not None:
            cx, cy = pipeline.estimate_xy(candidate)
            status = '' if confident else ' (низкая уверенность)'
            winner = f'Робот Б=({cx:.2f},{cy:.2f}) n={len(candidate)}{status}'
        else:
            winner = 'Робот Б=нет'
        others = [
            f"({c['center'][0]:.2f},{c['center'][1]:.2f}) n={c['n']} "
            f"ratio={c['density_ratio']:.2f} ok={c['density_ok']}" for c in debug_info
        ]
        self.get_logger().info(
            winner + ' | кандидаты: ' + ('; '.join(others) if others else 'нет'),
            throttle_duration_sec=1.0)

        if candidate is not None:
            candidate_msg = point_cloud2.create_cloud_xyz32(cloud_base.header, candidate)
            self.candidate_pub.publish(candidate_msg)

        if confident:
            z = float(candidate[:, 2].mean())
            lifetime = Duration(seconds=0.5).to_msg()

            sphere = Marker()
            sphere.header = cloud_base.header
            sphere.ns = 'robot_b'
            sphere.id = 0
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(cx)
            sphere.pose.position.y = float(cy)
            sphere.pose.position.z = z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.1
            sphere.color.r = 1.0
            sphere.color.a = 1.0
            sphere.lifetime = lifetime

            label = Marker()
            label.header = cloud_base.header
            label.ns = 'robot_b'
            label.id = 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(cx)
            label.pose.position.y = float(cy)
            label.pose.position.z = z + 0.15
            label.pose.orientation.w = 1.0
            label.scale.z = 0.1
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.lifetime = lifetime
            label.text = f'x={cx:.2f}, y={cy:.2f}'

            self.position_pub.publish(MarkerArray(markers=[sphere, label]))


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
