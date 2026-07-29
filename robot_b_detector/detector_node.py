"""ROS2-нода: подписывается на /livox/lidar и публикует debug-топики.

Stage 1 (удаление пола): плоскость пересчитывается не на каждом кадре, а раз
в RECALIBRATION_PERIOD_SEC по накопленным точкам - иначе разные RANSAC-фиты
вместе с Decay Time в RViz визуально "отращивают" пол обратно.

Stage 2 (кандидат-робот): кластеризация по скользящему окну
ACCUMULATION_WINDOW_SEC, не по одному кадру - у Livox неповторяющийся
паттерн сканирования, один кадр слишком разреженный на дальних дистанциях.
"""

import collections

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from robot_b_detector import pipeline

TARGET_FRAME = 'base_link'
RECALIBRATION_PERIOD_SEC = 3.0
ACCUMULATION_WINDOW_SEC = 1.5


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

        self._ground_plane = None
        self._calibration_buffer = []
        self._last_calibration = self.get_clock().now()
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
        self._accum_buffer.append((now_sec, remaining))
        while now_sec - self._accum_buffer[0][0] > ACCUMULATION_WINDOW_SEC:
            self._accum_buffer.popleft()

        accumulated = np.concatenate([p for _, p in self._accum_buffer])
        down = pipeline.voxel_downsample(accumulated)
        clusters = pipeline.cluster_points(down)
        candidate, debug_info = pipeline.select_robot_candidate(
            clusters, accumulated, return_debug=True)

        if candidate is not None:
            cx, cy = candidate[:, 0].mean(), candidate[:, 1].mean()
            winner = f'ПОБЕДИТЕЛЬ=({cx:.2f},{cy:.2f}) n={len(candidate)}'
        else:
            winner = 'ПОБЕДИТЕЛЬ=нет'
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


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
