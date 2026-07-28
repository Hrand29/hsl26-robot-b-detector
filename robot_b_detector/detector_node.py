"""ROS2-нода: подписывается на /livox/lidar и публикует debug-топики по мере
готовности стадий пайплайна.

Stage 1: перевод облака в base_link, удаление пола -> /debug/floor_removed

Плоскость пола пересчитывается не на каждом кадре: RANSAC даёт немного разную
плоскость каждый раз, а Decay Time в RViz копит кадры за пару секунд — из-за
этого пол визуально "отрастает обратно". Вместо этого калибруемся раз в
RECALIBRATION_PERIOD_SEC по накопленным за интервал точкам: стабильно внутри
интервала и подстраивается, если сенсор сдвинется.
"""

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

        self._ground_plane = None
        self._calibration_buffer = []
        self._last_calibration = self.get_clock().now()

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


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
