#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Perception + Gesture Recognition node (paper Sec. 3.4-3.5, Algorithm 1 "Unified-PDD").

Subscribes  /camera (RGB, bgr8) and /depth_camera (32FC1) from the x500_depth OakD-Lite
Publishes   /gesture/command    std_msgs/String   LEFT | RIGHT | NEUTRAL | NONE (temporally filtered)
            /gesture/distance   std_msgs/Float32  estimated distance to the selected person (m)
            /gesture/annotated  sensor_msgs/Image boxes, keypoints/skeleton, distance, direction

input_mode = rgb    pose on the RGB frame, distance from the co-located depth frame (default)
input_mode = depth  pose on the colour-mapped depth frame (pseudo-RGB), as in depth.py
input_mode = rgbd   RGB + depth aligned to it, 4-channel input (trained RGB-D models T2/T3)
input_mode = fused  depth-weighted pose fusion, paper Eq. (1)-(2) (drone_control.fusion): pose on
                    the RGB and the depth frame, joints fused in 3D with the temporal and body-plane
                    estimates, Algorithm 1 (gesture rules + line 13b) decides; fusion_lambda = lambda

model = a trained gesture model (<workspace>/train, classes LEFT / RIGHT / NEUTRAL): the
class of the detection is the direction (GesturePipeline.trained: no joint-angle rule, no
additions, nearest person commands); detected from the model's class names.

Robustness additions on top of Algorithm 1 (each one a parameter, see RUN_SAR.md):
  refine       person found but no pointing arm -> run pose again on a magnified crop of that
               person (a thin arm against clutter is only a few pixels wide at 640 input)
  face_camera  the operator faces the UAV, so fix YOLO's back-view left/right swap (= Algorithm 1 line 13b,
               sides as the UAV sees them; launch uav_view:=true, the default, also with robust:=false)
  kp_conf      LEFT/RIGHT only if the shoulder/elbow/wrist/hip on that side are confidently
               located (rejects phantom "people" YOLO finds in fog or clutter)
  dehaze       (rgb only) when no arm is found, or nobody is found in a hazy (low-contrast) frame,
               retry on a contrast-enhanced (CLAHE) image: fog washes out the white figure and
               especially its thin arm
  one_arm      a command needs exactly one extended arm (two-arm signals are not commands)
  both_arms_hover  the gesture rules in Algorithm 1 (on by default, also with
               robust:=false): exactly one arm stretched out, other arm seen down or up -> that
               direction; both arms stretched out, no arm, or other arm not seen (fog) -> hover
  lateral_check  LEFT/RIGHT from where the extended wrist is (outside the shoulders, image-left =
               operator's RIGHT); fixes crossed skeletons where YOLO swaps the arms
  nearest_operator  only the nearest person (the operator) can command the UAV
The per-frame logic lives in drone_control/pipeline.py (shared with <workspace>/eval).

Also publishes /gesture/timing  std_msgs/Float32MultiArray [processing time (s, wall),
            frame age at decision (s, simulation time; NaN without /clock)]
            /gesture/frame   std_msgs/String JSON per processed frame (before the temporal filter):
            raw label, confidence and distance of the selected person (0 / null if none), people
"""
import json
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, String
from ultralytics import YOLO

from drone_control import pdd
from drone_control.pipeline import DEFAULT_WEIGHTS, GesturePipeline  # noqa: F401 (re-export)


class GestureNode(Node):
    def __init__(self):
        super().__init__('gesture_node')
        self.declare_parameter('model', DEFAULT_WEIGHTS if os.path.exists(DEFAULT_WEIGHTS)
                               else 'yolo11n-pose.pt')
        self.declare_parameter('rgb_topic', '/camera')
        self.declare_parameter('depth_topic', '/depth_camera')
        self.declare_parameter('input_mode', 'rgb')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('rgb_hfov', 1.204)      # OakD-Lite IMX214 (model.sdf)
        self.declare_parameter('depth_hfov', 1.274)    # OakD-Lite StereoOV7251 (model.sdf)
        self.declare_parameter('window', 7)
        self.declare_parameter('min_votes', 5)
        self.declare_parameter('show', True)
        self.declare_parameter('window_width', 0)   # show: window width in px (0 = image size)
        self.declare_parameter('device', '')           # '' = auto (CUDA GPU if available), 'cpu', '0'
        self.declare_parameter('half', False)          # FP16 on GPU: slower, NaN outputs on some GPUs
        self.declare_parameter('threads', 0)           # CPU inference threads; 0 = Ultralytics default (8)
        self.declare_parameter('imgsz', 0)             # inference size; 0 = 640 (rgb) / 1280 (depth)
        self.declare_parameter('refine', True)
        self.declare_parameter('face_camera', True)
        self.declare_parameter('kp_conf', 0.5)
        self.declare_parameter('dehaze', True)
        self.declare_parameter('haze_std', 20.0)       # frame grey-level std below this = hazy
        self.declare_parameter('one_arm', True)
        self.declare_parameter('both_arms_hover', True)
        self.declare_parameter('lateral_check', True)
        self.declare_parameter('nearest_operator', True)
        self.declare_parameter('fusion_lambda', 3000.0)  # input_mode fused: lambda of Eq. (2)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.mode = p('input_mode')
        self.show = p('show')
        self.window_width = int(p('window_width'))
        self.window_sized = False
        import torch
        device = p('device') or ('0' if torch.cuda.is_available() else 'cpu')
        model = YOLO(p('model'), task='pose')
        trained = {'LEFT', 'RIGHT', 'NEUTRAL'} <= {n.upper() for n in model.names.values()}
        channels = model.model.yaml.get('channels', 3)
        if (self.mode == 'rgbd') != (channels == 4) or (self.mode == 'rgbd' and not trained):
            raise RuntimeError(f"input_mode={self.mode} does not fit {p('model')} "
                               f"({channels}-channel input{', trained' if trained else ''}); "
                               "rgbd needs a trained RGB-D model and vice versa")
        common = dict(conf=p('conf'), device=device, half=bool(p('half')) and device != 'cpu',
                      rgb_hfov=p('rgb_hfov'), depth_hfov=p('depth_hfov'), threads=p('threads'))
        if p('imgsz'):
            common['imgsz'] = p('imgsz')
        if trained:
            self.pipe = GesturePipeline.trained(model, mode=self.mode, **common)
        elif self.mode == 'fused':
            from drone_control.fusion import FusedPipeline
            self.pipe = FusedPipeline(model, lam=p('fusion_lambda'), **common)
        else:
            self.pipe = GesturePipeline(
                model, mode=self.mode, refine=p('refine'), face_camera=p('face_camera'),
                kp_conf=p('kp_conf'), dehaze=p('dehaze'), haze_std=p('haze_std'),
                one_arm=p('one_arm'), lateral_check=p('lateral_check'),
                nearest_operator=p('nearest_operator'),
                both_arms_hover=p('both_arms_hover'), **common)
        q = self.pipe
        gpu = torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'
        rule = (f"trained gesture classes {sorted(q.classes.values())} (no joint-angle rule)"
                if trained else
                f"refine={q.refine} face_camera={q.face} kp_conf={q.kp_conf} dehaze={q.dehaze} "
                f"one_arm={q.one_arm} lateral_check={q.lateral} both_arms_hover={q.both_arms_hover}")
        self.get_logger().info(
            f"YOLOv11 pose model: {p('model')}  input_mode={self.mode}  imgsz={q.imgsz}  "
            f"device={gpu} threads={q.threads or 'default'}{' (FP16)' if q.half else ''}  "
            f"{rule} nearest_operator={q.nearest}  topics {p('rgb_topic')} {p('depth_topic')}")

        self.br = CvBridge()
        self.rgb = None
        self.depth = None
        self.filter = pdd.TemporalFilter(p('window'), p('min_votes'))

        self.create_subscription(Image, p('rgb_topic'), self.rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, p('depth_topic'), self.depth_cb, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(String, '/gesture/command', 10)
        self.dist_pub = self.create_publisher(Float32, '/gesture/distance', 10)
        self.img_pub = self.create_publisher(Image, '/gesture/annotated', 1)
        self.timing_pub = self.create_publisher(Float32MultiArray, '/gesture/timing', 10)
        self.frame_pub = self.create_publisher(String, '/gesture/frame', 10)
        self.sim_t = None
        self.create_subscription(Clock, '/clock', self.clock_cb, 10)
        # Inference runs on a timer on the latest frame, so a slow (CPU) model never backlogs.
        self.create_timer(1.0 / p('rate_hz'), self.step)

    def rgb_cb(self, msg):
        self.rgb = msg

    def depth_cb(self, msg):
        self.depth = msg

    def clock_cb(self, msg):
        self.sim_t = msg.clock.sec + msg.clock.nanosec * 1e-9

    def step(self):
        src = self.depth if self.mode == 'depth' else self.rgb
        if src is None or (self.mode in ('rgbd', 'fused') and self.depth is None):
            return
        t0 = time.monotonic()
        try:
            depth = (self.br.imgmsg_to_cv2(self.depth, desired_encoding='32FC1')
                     if self.depth is not None else None)
            rgb = (self.br.imgmsg_to_cv2(self.rgb, desired_encoding='bgr8')
                   if self.rgb is not None and self.mode != 'depth' else None)
        except CvBridgeError as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return
        frame, people, raw, best = self.pipe.process(rgb, depth)
        if frame is None:
            return
        cmd = self.filter.update(raw)
        self.cmd_pub.publish(String(data=cmd))
        if best is not None:
            self.dist_pub.publish(Float32(data=best.distance))
        stamp = src.header.stamp.sec + src.header.stamp.nanosec * 1e-9
        age = self.sim_t - stamp if self.sim_t is not None else float('nan')
        self.timing_pub.publish(Float32MultiArray(data=[time.monotonic() - t0, age]))
        self.frame_pub.publish(String(data=json.dumps(dict(
            raw=raw, conf=round(best.conf, 4) if best else 0.0,
            dist=round(best.distance, 3) if best else None, n=len(people), stamp=stamp))))
        self.publish_annotated(frame, people, raw, cmd)

    # COCO skeleton (keypoint index pairs) and colours: head, arms, torso, legs
    SKELETON = [((0, 1), 0), ((0, 2), 0), ((1, 3), 0), ((2, 4), 0), ((5, 7), 1), ((7, 9), 1),
                ((6, 8), 1), ((8, 10), 1), ((5, 6), 2), ((5, 11), 2), ((6, 12), 2), ((11, 12), 2),
                ((11, 13), 3), ((13, 15), 3), ((12, 14), 3), ((14, 16), 3)]
    COLOURS = [(0, 255, 0), (255, 128, 0), (255, 0, 255), (0, 165, 255)]

    def publish_annotated(self, frame, people, raw, cmd):
        img = frame.copy()
        lw = max(2, frame.shape[1] // 640)
        for pr in people:
            x1, y1, x2, y2 = (int(v) for v in pr.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), lw)
            for (a, b), c in self.SKELETON:
                if np.any(pr.keypoints[a]) and np.any(pr.keypoints[b]):
                    cv2.line(img, tuple(int(v) for v in pr.keypoints[a]),
                             tuple(int(v) for v in pr.keypoints[b]), self.COLOURS[c], lw)
            cv2.putText(img, f'person {pr.conf:.2f}  {pr.direction} {pr.distance:.2f}m',
                        (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.25 * lw, (0, 255, 0), lw)
        banner = f'Direction: {cmd}' if people else 'No person detected'
        cv2.putText(img, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(img, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(img, f'frame: {raw}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        self.img_pub.publish(self.br.cv2_to_imgmsg(img, encoding='bgr8'))
        if self.show:
            if self.window_width > 0 and not self.window_sized:   # display only
                cv2.namedWindow('Unified-PDD', cv2.WINDOW_NORMAL)
                h, w = img.shape[:2]
                cv2.resizeWindow('Unified-PDD', self.window_width, int(self.window_width * h / w))
                cv2.moveWindow('Unified-PDD', 70, 40)
                self.window_sized = True
            cv2.imshow('Unified-PDD', img)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:                       # a second Ctrl+C during shutdown must not print a traceback
            node.destroy_node()
            cv2.destroyAllWindows()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
