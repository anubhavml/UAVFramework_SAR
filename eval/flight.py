#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Gesture-switch flight experiment driver (started by run_flight.sh).

The person stands 5 m in front of the drone.  Once the drone hovers, the gesture model is
switched NEUTRAL -> RIGHT -> LEFT -> NEUTRAL at recorded simulation times (max_turn is limited
so the person stays in view).  Logs every gesture command, /gesture/timing sample and vehicle
heading with both simulation and wall-clock time to <out>/log.jsonl; flight_report.py analyses it.
"""
import argparse
import json
import math
import os
import threading
import time

import rclpy
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, String

import simtools as st

# (gesture shown, simulation seconds it is held), after the drone hovers
SCHEDULE = [('neutral', 8.0), ('right', 12.0), ('left', 16.0), ('neutral', 8.0)]
if os.environ.get('FLIGHT_SCHEDULE'):     # gesture videos: e.g. neutral:6,both_out:16,neutral:6
    SCHEDULE = [(g, float(s)) for g, s in (x.split(':') for x in os.environ['FLIGHT_SCHEDULE'].split(','))]


class Driver(Node):
    def __init__(self, gz, log):
        super().__init__('flight_driver')
        self.gz, self.log = gz, log
        self.pos = self.status = None
        self.last_pos_log = 0.0
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position',
                                 self.pos_cb, qos)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status',
                                 lambda m: setattr(self, 'status', m), qos)
        self.create_subscription(String, '/gesture/command',
                                 lambda m: self.write('cmd', m.data), 10)
        self.create_subscription(Float32MultiArray, '/gesture/timing',
                                 lambda m: self.write('timing', list(m.data)), 10)
        self.create_subscription(String, '/gesture/frame',          # per-frame label/confidence
                                 lambda m: self.write('frame', json.loads(m.data)), 10)
        self.lock = threading.Lock()
        # callbacks run in their own thread, so messages are stamped on arrival even while the
        # main thread waits for a set_pose reply
        self.spinner = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self.spinner.start()

    def write(self, kind, data, sim=None):
        with self.lock:
            self.log.write(json.dumps(dict(wall=time.monotonic(),
                                           sim=self.gz.sim_t if sim is None else sim,
                                           kind=kind, data=data)) + '\n')

    def pos_cb(self, m):
        self.pos = m
        if time.monotonic() - self.last_pos_log >= 0.05:          # 20 Hz is plenty
            self.last_pos_log = time.monotonic()
            self.write('pos', [m.x, m.y, m.z, math.degrees(m.heading), self.true_z()])

    def true_z(self):
        """Height of the drone in Gazebo (ground truth; PX4's z can drift before take-off)."""
        z = [p[2] for n, p in self.gz.dyn_poses.items() if n.startswith('x500')]
        return z[0] if z else None

    def spin_until(self, cond, timeout):
        t0 = time.monotonic()
        while not cond() and time.monotonic() - t0 < timeout:
            time.sleep(0.01)
        return cond()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('out')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gz = st.Gz(a.world, topics=(), track=True)
    gz.wait_clock(1.0)
    _, (hx, hy, hyaw), _ = st.world_info(a.world)
    parked = {g: (hx, hy + 3 * i, st.PARK_Z) for i, g in enumerate(st.GESTURES)}

    def show(g):
        """Park the other models and bring human_<g> in (one atomic set_pose_vector); returns
        the simulation time at which Gazebo first reports it in place (replies can lag by
        seconds under load)."""
        gz.visible_since.pop(f'human_{g}', None)
        gz.set_poses([(f'human_{o}', *parked[o], hyaw) for o in st.GESTURES if o != g]
                     + [(f'human_{g}', hx, hy, 0.0, hyaw)])
        t0 = time.monotonic()
        while f'human_{g}' not in gz.visible_since and time.monotonic() - t0 < 30:
            time.sleep(0.01)
        return gz.visible_since.get(f'human_{g}', gz.sim_t)

    rclpy.init()
    with open(f'{a.out}/log.jsonl', 'w') as log:
        drv = Driver(gz, log)
        drv.write('swap', 'neutral', sim=show('neutral'))
        if not drv.spin_until(lambda: drv.pos is not None, 120):
            drv.write('error', 'no vehicle position'); return
        # true height from Gazebo: PX4's local z drifts by up to ~2 m on the ground in some runs
        # (EKF "vertical velocity unstable"), and the first sample may already be in the air
        airborne = lambda: (drv.status is not None and drv.status.nav_state == 14  # noqa: E731
                            and (drv.true_z() or 0.0) > 0.8)
        if not drv.spin_until(airborne, 240):
            drv.write('error', 'drone did not take off'); return
        # the climb to take-off height ends slowly; start once within 0.2 m of it (or after 40 s)
        drv.spin_until(lambda: (drv.true_z() or 0.0) > 1.3, 40)
        t = gz.sim_t + 4.0                                            # settle in the hover
        drv.spin_until(lambda: gz.sim_t >= t, 120)
        for g, hold in SCHEDULE:
            t_swap = show(g)
            drv.write('swap', g, sim=t_swap)
            drv.spin_until(lambda: gz.sim_t >= t_swap + hold, 300)
        drv.write('end', None)
    rclpy.shutdown()                  # ends rclpy.spin in the logger thread
    drv.spinner.join(timeout=5)       # (exiting while it runs aborts with a core dump)


if __name__ == '__main__':
    main()
