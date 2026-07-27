"""ROS2-нода: подписывается на /livox/lidar и публикует позу найденного робота Б.

TODO: implement
- подписка на sensor_msgs/PointCloud2 (топик /livox/lidar)
- перевод облака в base_link по tf_static
- удаление плоскости пола
- кластеризация, отбор кандидата по габаритам робота
- публикация geometry_msgs/PoseStamped и visualization_msgs/Marker
"""

import rclpy
from rclpy.node import Node


class DetectorNode(Node):
    def __init__(self):
        super().__init__('robot_b_detector')
        # TODO: subscription на /livox/lidar


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
