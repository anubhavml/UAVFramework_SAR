# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Helpers for the SAR HITL evaluation: evaluation worlds, seeded placements, Gazebo control."""
import math
import os
import re
import subprocess
import threading
import time

import cv2
import numpy as np

WS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # workspace root
WORLDS = f'{WS}/src/sar_sim/worlds'
EVAL = f'{WS}/eval'
GESTURES = ('right', 'left', 'neutral')            # sar_human_<g> models
# more parked humans (the demo uses all seven poses), e.g. SAR_GESTURES=right,left,neutral,both_out
if os.environ.get('SAR_GESTURES'):
    GESTURES = tuple(os.environ['SAR_GESTURES'].split(','))
LABEL = {'right': 'RIGHT', 'left': 'LEFT', 'neutral': 'NEUTRAL', 'both_out': 'NEUTRAL', 'both_up': 'NEUTRAL',
         'rightout_leftup': 'RIGHT', 'leftout_rightup': 'LEFT'}
PARK_Z = -30.0                                     # unused humans wait below the ground plane
RIG_TOPICS = ('rgb', 'depth_oakd', 'depth_d435i', 'gt_depth')

HUMAN_RE = re.compile(r'<include>\s*<uri>model://sar_human_\w+</uri>\s*<name>sar_human</name>'
                      r'\s*<pose>([^<]+)</pose>\s*</include>')
TREE_RE = re.compile(r'<uri>model://(?:oak_tree|pine_tree)</uri>\s*<name>[^<]+</name>'
                     r'\s*<pose>(\S+) (\S+)')


def world_info(world):
    """-> (sdf text, human stand pose (x, y, yaw), tree trunk positions)."""
    s = open(f'{WORLDS}/{world}.sdf').read()
    x, y, _, _, _, yaw = map(float, HUMAN_RE.search(s).group(1).split())
    trees = [(float(a), float(b)) for a, b in TREE_RE.findall(s)]
    return s, (x, y, yaw), trees


def make_eval_world(world, out_dir, rig=True, suffix=''):
    """Copy of a project world with the three gesture humans (parked) and the camera rig."""
    s, (hx, hy, hyaw), _ = world_info(world)
    inc = lambda uri, name, pose: (f'<include>\n      <uri>model://{uri}</uri>\n'  # noqa: E731
                                   f'      <name>{name}</name>\n      <pose>{pose}</pose>\n'
                                   f'    </include>')
    block = '\n    '.join([inc(f'sar_human_{g}', f'human_{g}', f'{hx} {hy + 3 * i} {PARK_Z} 0 0 {hyaw}')
                           for i, g in enumerate(GESTURES)] +
                          ([inc('eval_rig', 'rig', f'{hx - 5} {hy} 1.9 0 0 0')] if rig else []))
    os.makedirs(out_dir, exist_ok=True)
    path = f'{out_dir}/{world}{suffix}.sdf'
    with open(path, 'w') as f:
        f.write(HUMAN_RE.sub(block, s, count=1))
    return path


def placements(world, distances, n, seed=0):
    """n seeded camera placements per distance, shared by all gestures and conditions.
    Camera within +-15 deg of the direction the person faces, aimed at them +-8 deg (so they are
    not always centred), height 1.75-2.05 m (hover height ~1.9 m); person turned +-10 deg away
    from facing the camera.  Positions within 1.2 m of a tree trunk are rejected."""
    _, (hx, hy, hyaw), trees = world_info(world)
    base = os.path.basename(world).replace('_fog', '')
    rng = np.random.default_rng([seed, sum(map(ord, base))])
    out = []
    for d in distances:
        k = 0
        while k < n:
            bearing = hyaw + math.radians(rng.uniform(-15, 15))
            cx, cy = hx + d * math.cos(bearing), hy + d * math.sin(bearing)
            aim = math.atan2(hy - cy, hx - cx) + math.radians(rng.uniform(-8, 8))
            h = rng.uniform(1.75, 2.05)
            human_yaw = math.atan2(cy - hy, cx - hx) + math.radians(rng.uniform(-10, 10))
            if any(math.hypot(cx - tx, cy - ty) < 1.2 for tx, ty in trees):
                continue
            out.append(dict(d=d, k=k, cam=(cx, cy, h, aim), human=(hx, hy, human_yaw)))
            k += 1
    return out


def _decode(msg):
    h, w = msg.height, msg.width
    if msg.pixel_format_type == 13:                      # R_FLOAT32
        return np.frombuffer(msg.data, np.float32).reshape(h, w).copy()
    img = np.frombuffer(msg.data, np.uint8).reshape(h, w, -1)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class Gz:
    """Minimal gz-transport client: sim clock, latest rig images, set_pose."""

    def __init__(self, world, topics=RIG_TOPICS, prefix='rig/', track=False):
        # Pose_V must be imported before pose_pb2, or its callbacks fail with
        # "No message class registered for 'gz.msgs.Pose'"
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.clock_pb2 import Clock
        from gz.msgs10.image_pb2 import Image
        from gz.msgs10.pose_pb2 import Pose
        from gz.transport13 import Node
        self.Pose, self.Boolean = Pose, Boolean
        self.world, self.node = world, Node()
        self.lock = threading.Lock()
        self.sim_t, self.latest = None, {}
        self.poses, self.dyn_poses, self.visible_since = {}, {}, {}
        self.node.subscribe(Clock, f'/world/{world}/clock', self._clock)
        if track:   # ground truth: when a (static) person model appears; the drone's true pose
            self.node.subscribe(Pose_V, f'/world/{world}/pose/info', self._poses)
            # pose/info does not update moving models; dynamic_pose/info does
            self.node.subscribe(Pose_V, f'/world/{world}/dynamic_pose/info',
                                lambda m: setattr(self, 'dyn_poses', {
                                    p.name: (p.position.x, p.position.y, p.position.z)
                                    for p in m.pose}))
        for t in topics:
            self.node.subscribe(Image, prefix + t, lambda m, t=t: self._image(t, m))
        self.topics = topics

    def _clock(self, m):
        self.sim_t = m.sim.sec + m.sim.nsec * 1e-9

    def _poses(self, m):
        stamp = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
        self.poses = {p.name: (p.position.x, p.position.y, p.position.z) for p in m.pose}
        for name, (_, _, z) in self.poses.items():      # sim time a model left the parking spot
            if z < PARK_Z / 2:
                self.visible_since.pop(name, None)
            else:
                self.visible_since.setdefault(name, stamp)

    def _image(self, topic, m):
        with self.lock:
            self.latest[topic] = m

    def wait_clock(self, t_min=0.0, timeout=120):
        t0 = time.time()
        while (self.sim_t is None or self.sim_t < t_min) and time.time() - t0 < timeout:
            time.sleep(0.05)
        if self.sim_t is None or self.sim_t < t_min:
            raise RuntimeError(f'no /world/{self.world}/clock >= {t_min}')
        return self.sim_t

    def set_pose(self, name, x, y, z, yaw=0.0):
        req = self.Pose(name=name)
        req.position.x, req.position.y, req.position.z = x, y, z
        req.orientation.z, req.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        for attempt in range(10):     # can time out while gz starts up or renders heavy fog
            ok, rep = self.node.request(f'/world/{self.world}/set_pose', req, self.Pose,
                                        self.Boolean, 10000)
            if ok and rep.data:
                return
            time.sleep(2.0)
        raise RuntimeError(f'set_pose {name} failed')

    def set_poses(self, poses, timeout_ms=3000, attempts=10):
        """Move several models in one step: [(name, x, y, z, yaw[, pitch])].  Uses the gz CLI in
        a subprocess: the Python binding's request() holds the GIL while it waits (up to its
        timeout), which stalls every other thread of the caller (e.g. an rclpy logger)."""
        def quat(yaw, pitch=0.0):                        # yaw about z, then pitch (+ = nose down)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
            return f'x: {-sy * sp} y: {cy * sp} z: {sy * cp} w: {cy * cp}'
        req = ' '.join(
            f'pose {{name: "{n}" position {{x: {x} y: {y} z: {z}}} orientation {{{quat(*a)}}}}}'
            for n, x, y, z, *a in poses)
        cmd = ['gz', 'service', '-s', f'/world/{self.world}/set_pose_vector',
               '--reqtype', 'gz.msgs.Pose_V', '--reptype', 'gz.msgs.Boolean',
               '--timeout', str(timeout_ms), '--req', req]
        for attempt in range(attempts):
            out = subprocess.run(cmd, capture_output=True, text=True).stdout
            if 'data: true' in out:
                return
            time.sleep(1.0)
        raise RuntimeError(f'set_pose_vector {[p[0] for p in poses]} failed')

    def grab(self, after, timeout=60):
        """Latest image of every topic rendered at sim time >= after -> ({topic: array}, stamp)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                msgs = dict(self.latest)
            stamps = {t: m.header.stamp.sec + m.header.stamp.nsec * 1e-9
                      for t, m in msgs.items()}
            if len(msgs) == len(self.topics) and min(stamps.values()) >= after:
                return {t: _decode(m) for t, m in msgs.items()}, min(stamps.values())
            time.sleep(0.02)
        raise RuntimeError(f'no fresh images after sim t={after:.2f}')
