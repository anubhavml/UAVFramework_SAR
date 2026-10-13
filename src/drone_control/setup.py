# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
from glob import glob

from setuptools import find_packages, setup

package_name = 'drone_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Anubhav Pandey',
    maintainer_email='p20220020@pilani.bits-pilani.ac.in',
    description='YOLOv11 Unified-PDD gesture recognition and PX4 offboard control for SAR UAVs.',
    license='AGPL-3.0-only',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drone_control = drone_control.control_node:main',
            'gesture_node = drone_control.gesture_node:main',
            'sensor_sim = drone_control.sensor_sim_node:main',
        ],
    },
)
