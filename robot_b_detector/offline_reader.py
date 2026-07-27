"""Офлайн-чтение mcap-файлов без запуска ROS 2 — для отладки алгоритма на bag-записях.

TODO: implement
- чтение /livox/lidar (sensor_msgs/PointCloud2) через mcap_ros2.reader
- парсинг point_step=26 (x,y,z,intensity float32 + tag,line uint8 + timestamp float64)
- применение tf_static (base_link -> livox) к точкам
"""
