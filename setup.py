from setuptools import find_packages, setup

package_name = 'robot_b_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/detector.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/livox_view.rviz']),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='Roman Komarov',
    maintainer_email='komarov.iphone@gmail.com',
    description='Детекция и локализация робота Turtlebot2 в облаках точек Livox MID-360 (HSL26)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node = robot_b_detector.detector_node:main',
        ],
    },
)
