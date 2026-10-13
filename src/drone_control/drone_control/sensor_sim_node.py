#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Sensor simulation node: physically based fog and RealSense D435i noise on the Gazebo images.

Subscribes  /camera (bgr8), /depth_camera (32FC1) and, if present, /camera_gt_depth (long-range
            z-depth with the RGB camera's field of view, used for the fog; else /depth_camera)
Publishes   /sim/camera, /sim/depth_camera (same headers / stamps as the inputs), at rate_hz

Parameters: visibility (m, 0 = no fog), airlight (0-255), d435i_noise, rgb_noise,
            rgb_hfov, depth_hfov, gt_depth_hfov.  Models in drone_control/sensor_model.py.
"""
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from drone_control import sensor_model as sm


class SensorSim(Node):
    def __init__(self):
        super().__init__('sensor_sim')
        for name, default in (('visibility', 0.0), ('airlight', 217.0), ('d435i_noise', False),
                              ('rgb_noise', False), ('rgb_hfov', 1.204), ('depth_hfov', 1.274),
                              ('gt_depth_hfov', 1.204), ('rate_hz', 15.0), ('seed', 0)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.vis, self.airlight = float(p('visibility')), float(p('airlight'))
        self.d435i, self.rgb_noise = p('d435i_noise'), p('rgb_noise')
        self.rgb_hfov, self.depth_hfov, self.gt_hfov = p('rgb_hfov'), p('depth_hfov'), p('gt_depth_hfov')
        self.rng = np.random.default_rng(p('seed'))
        self.br = CvBridge()
        self.rgb = self.depth = self.gt = None
        for topic, attr in (('/camera', 'rgb'), ('/depth_camera', 'depth'), ('/camera_gt_depth', 'gt')):
            self.create_subscription(Image, topic, lambda m, a=attr: setattr(self, a, m),
                                     qos_profile_sensor_data)
        self.rgb_pub = self.create_publisher(Image, '/sim/camera', qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(Image, '/sim/depth_camera', qos_profile_sensor_data)
        self.create_timer(1.0 / p('rate_hz'), self.step)
        self.get_logger().info(f'sensor_sim: visibility={self.vis or "off"} m  '
                               f'd435i_noise={self.d435i}  rgb_noise={self.rgb_noise}')

    def step(self):
        if self.rgb is None or self.depth is None:
            return
        rgb_msg, depth_msg = self.rgb, self.depth
        rgb = self.br.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.br.imgmsg_to_cv2(depth_msg, '32FC1')
        if self.vis > 0:
            src, hfov = ((self.br.imgmsg_to_cv2(self.gt, '32FC1'), self.gt_hfov)
                         if self.gt is not None else (depth, self.depth_hfov))
            z_rgb = sm.align_depth(src, hfov, (rgb.shape[1], rgb.shape[0]), self.rgb_hfov)
            rgb = sm.fog_rgb(rgb, z_rgb, self.rgb_hfov, self.vis, self.airlight)
            depth = sm.fog_depth(depth, self.depth_hfov, self.vis)
        if self.d435i:
            depth = sm.d435i_depth_noise(depth, self.depth_hfov, self.rng)
        if self.rgb_noise:
            rgb = sm.rgb_noise(rgb, rng=self.rng)
        out_rgb = self.br.cv2_to_imgmsg(rgb, 'bgr8')
        out_rgb.header = rgb_msg.header
        out_d = self.br.cv2_to_imgmsg(np.ascontiguousarray(depth, np.float32), '32FC1')
        out_d.header = depth_msg.header
        self.rgb_pub.publish(out_rgb)
        self.depth_pub.publish(out_d)


def main(args=None):
    rclpy.init(args=args)
    node = SensorSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
