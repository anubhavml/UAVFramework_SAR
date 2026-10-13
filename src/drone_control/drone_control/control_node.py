#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Control Node (paper Sec. 3.6): gesture commands -> PX4 offboard setpoints over uXRCE-DDS.

The UAV turns (yaws in place) toward the side the operator points to.  The operator faces the
drone, so the arm they point with is mirrored from the drone's point of view:

  /gesture/command  RIGHT (operator's right arm) -> yaw LEFT  (towards where they point)
                    LEFT  (operator's left arm)  -> yaw RIGHT
                    NEUTRAL / NONE / stale       -> hold heading and hover

Flight sequence: stream offboard heartbeat -> OFFBOARD + ARM -> climb to `takeoff_alt` above the
take-off point and hold that position -> follow gestures by turning at `yaw_rate` deg/s, at most
`max_turn` deg either side of the take-off heading.  Ctrl+C (or `land_on_exit`) sends LAND.

Demonstrations only (eval/demo_flight.py): /demo/advance (std_msgs/Float32, metres) moves the
hold point that far ahead of the take-off point along the take-off heading, at `approach_speed`
m/s, and turns the UAV back to the take-off heading (facing the person). Without it, unchanged.

react:=fly (demonstrations; default react:=turn = the behaviour above, as in the paper): a RIGHT or
LEFT command held for `fly_trigger` s makes the UAV go where the operator points: turn `fly_turn`
deg to that side, fly `fly_distance` m along the new heading at `fly_speed` m/s and hover there
until /demo/advance (the demo's "come back" button) brings it back to the hold point, facing the
operator again.  The next turn command is accepted only after a NEUTRAL/NONE (the operator
lowers the arm).  /demo/react (std_msgs/String,
JSON {"react": "turn"|"fly", "fly_turn": deg, "fly_distance": m, "fly_limit_left": m,
"fly_limit_right": m}) changes these while flying; the limits (free space either side, from the
demo driver's obstacle map) shorten the flight, and below 0.5 m the UAV stays and says why.
The current action is published on /drone_control/action (std_msgs/String).
Message/topic usage follows px4_ros_com (release/1.15) offboard_control.py.
"""
import json
import math
import time

import rclpy
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand,
                          VehicleLocalPosition, VehicleStatus)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float32, String

NAN = float('nan')


class DroneControl(Node):
    def __init__(self):
        super().__init__('drone_control')
        self.declare_parameter('takeoff_alt', 1.5)     # m above take-off point (camera ~1.9 m above ground)
        self.declare_parameter('yaw_rate', 15.0)       # turn rate while a gesture is held (deg/s)
        self.declare_parameter('max_turn', 90.0)       # max turn either side of take-off heading (deg)
        self.declare_parameter('cmd_timeout', 1.0)     # s without a gesture -> hover
        self.declare_parameter('land_on_exit', True)
        self.declare_parameter('approach_speed', 1.0)  # m/s, demo approach (/demo/advance)
        self.declare_parameter('react', 'turn')        # turn (paper) | fly (go where pointed, return)
        self.declare_parameter('fly_turn', 90.0)       # deg, react:=fly
        self.declare_parameter('fly_distance', 5.0)    # m
        self.declare_parameter('fly_speed', 1.0)       # m/s
        self.declare_parameter('fly_trigger', 1.0)     # s a RIGHT/LEFT command must be held
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.alt = p('takeoff_alt')
        self.yaw_rate, self.max_turn = math.radians(p('yaw_rate')), math.radians(p('max_turn'))
        self.cmd_timeout = p('cmd_timeout')
        self.land_on_exit = p('land_on_exit')
        self.approach_speed = p('approach_speed')
        self.advance, self.advance_sp = 0.0, 0.0     # metres ahead of take-off: current, target
        self.react = {k: p(k) for k in ('react', 'fly_turn', 'fly_distance', 'fly_speed', 'fly_trigger')}
        self.react.update(fly_limit_left=math.inf, fly_limit_right=math.inf)
        self.man = None             # react:=fly manoeuvre in progress: dict(phase, dir, yaw, t)
        self.off = [0.0, 0.0]       # its offset from the hold point (m, NED x, y)
        self.dir_since = None       # (command, time) of the current RIGHT/LEFT command
        self.rearm = True           # a NEUTRAL/NONE was seen since the last manoeuvre

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.sp_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.pos_cb, qos)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.status_cb, qos)
        self.create_subscription(String, '/gesture/command', self.gesture_cb, 10)
        self.create_subscription(Float32, '/demo/advance', self.advance_cb, 10)
        self.create_subscription(String, '/demo/react', self.react_cb, 10)
        self.action_pub = self.create_publisher(String, '/drone_control/action', 10)
        # Gazebo's simulation clock (bridged by the launch file).  Turns are timed with it so the
        # turn rate is `yaw_rate` deg/s of simulation time even when Gazebo runs slower than real
        # time; without /clock (real vehicle) the wall clock is used.  (PX4's own timestamps are
        # re-synced to the agent's wall clock over uXRCE-DDS, with jumps, so they cannot be used.)
        self.create_subscription(Clock, '/clock', self.clock_cb, 10)
        self.sim_t = None           # latest /clock time (s)
        self.last_t = None          # (sim or wall) time of the previous loop iteration

        self.pos = None
        self.status = VehicleStatus()
        self.home = None            # (x, y, z, heading) at take-off
        self.yaw_sp = None          # current heading setpoint (rad, NED: + = clockwise)
        self.airborne = False
        self.gesture, self.gesture_t = 'NONE', 0.0
        self.last_action = None
        self.counter = 0
        self.create_timer(0.1, self.loop)
        self.get_logger().info('drone_control started: waiting for PX4 (/fmu/out/*) ...')

    # ---------------------------------------------------------------- callbacks
    def pos_cb(self, msg):
        self.pos = msg

    def status_cb(self, msg):
        self.status = msg

    def gesture_cb(self, msg):
        self.gesture, self.gesture_t = msg.data, self.now()

    def advance_cb(self, msg):
        self.advance_sp = float(msg.data)
        if self.man is not None:
            if self.man['phase'] in ('turn', 'fly', 'hold'):
                self.man.update(phase='return', t=self.now())  # come back first
        elif self.home is not None:
            self.yaw_sp = self.home[3]                       # face the person again
        self.get_logger().info(f'demo: hold point {self.advance_sp:.1f} m ahead of take-off')

    def react_cb(self, msg):
        try:
            new = json.loads(msg.data)
        except ValueError:
            return
        old = dict(self.react)
        self.react.update({k: (str(v) if k == 'react' else float(v)) for k, v in new.items()
                           if k in self.react})
        main = ('react', 'fly_turn', 'fly_distance')
        if any(old[k] != self.react[k] for k in main):
            self.get_logger().info('demo: reaction ' + ', '.join(f'{k}={self.react[k]}' for k in main))

    def clock_cb(self, msg):
        self.sim_t = msg.clock.sec + msg.clock.nanosec * 1e-9

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------------------------------------------------- PX4 helpers
    def ts(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def vehicle_command(self, command, p1=0.0, p2=0.0):
        m = VehicleCommand(command=command, param1=float(p1), param2=float(p2),
                           target_system=1, target_component=1, source_system=1,
                           source_component=1, from_external=True, timestamp=self.ts())
        self.cmd_pub.publish(m)

    def heartbeat(self):
        self.mode_pub.publish(OffboardControlMode(position=True, velocity=False, acceleration=False,
                                                  attitude=False, body_rate=False, timestamp=self.ts()))

    def setpoint(self, pos=(NAN, NAN, NAN), vel=(NAN, NAN, NAN), yaw=NAN):
        self.sp_pub.publish(TrajectorySetpoint(position=[float(v) for v in pos],
                                               velocity=[float(v) for v in vel],
                                               yaw=float(yaw), timestamp=self.ts()))

    def land(self):
        self.vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('LAND command sent')

    # ---------------------------------------------------------------- main loop (10 Hz)
    def loop(self):
        self.heartbeat()
        if self.pos is None or not self.pos.xy_valid:
            self.setpoint(vel=(0.0, 0.0, 0.0))
            return
        armed = self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED
        offboard = self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        self.counter += 1
        # After 1 s of setpoints, and every 2 s until PX4 accepts (it can deny arming while the
        # estimator settles, e.g. "vertical velocity unstable"): request OFFBOARD + ARM.
        if self.counter >= 10 and not (armed and offboard) and self.counter % 20 == 10:
            if not armed:
                # take-off reference: captured on the ground, right before each arm attempt
                self.home = (self.pos.x, self.pos.y, self.pos.z, self.pos.heading)
                self.yaw_sp = self.pos.heading
            self.vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)   # OFFBOARD
            self.vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.get_logger().info(f'OFFBOARD + ARM requested (armed={armed}, offboard={offboard}, '
                                   f'take-off ref z={self.home[2]:+.2f} m NED, '
                                   f'heading={math.degrees(self.home[3]):.0f} deg)')
        if self.home is None:
            self.setpoint(pos=(self.pos.x, self.pos.y, self.pos.z), yaw=self.pos.heading)
            return
        hx, hy, hz, yaw0 = self.home
        z_sp = hz - self.alt                                   # NED: up is -z
        t = self.sim_t if self.sim_t is not None else time.monotonic()
        dt = 0.0 if self.last_t is None else min(max(t - self.last_t, 0.0), 0.5)
        self.last_t = t

        if not (armed and offboard):
            self.airborne = False
        elif self.pos.z < z_sp + 0.3:
            self.airborne = True                               # latched until disarm / mode change
        stale = self.now() - self.gesture_t > self.cmd_timeout
        g = 'NONE' if stale else self.gesture

        # Operator's RIGHT arm points to the drone's LEFT (they face each other): yaw left (-).
        turn = {'RIGHT': -1.0, 'LEFT': 1.0}.get(g, 0.0) if self.airborne else 0.0
        if not turn:
            self.dir_since = None
            if self.man is None:
                self.rearm = True       # NEUTRAL/NONE seen after the last manoeuvre (react:=fly)
        elif self.dir_since is None or self.dir_since[0] != g:
            self.dir_since = (g, self.now())
        if self.man is not None or self.react['react'] == 'fly':
            action = self.fly(turn, yaw0, dt)                  # react:=fly
            turn = 0.0
        elif turn:
            nxt = self.yaw_sp + turn * self.yaw_rate * dt
            if abs(wrap(nxt - yaw0)) > self.max_turn:
                turn = 0.0                                         # turn limit reached
            elif abs(wrap(nxt - self.pos.heading)) <= math.radians(20):
                self.yaw_sp = wrap(nxt)                            # never lead the real heading >20 deg
        step = self.approach_speed * dt                       # demo approach (/demo/advance)
        self.advance += max(-step, min(step, self.advance_sp - self.advance))
        self.setpoint(pos=(hx + self.advance * math.cos(yaw0) + self.off[0],
                           hy + self.advance * math.sin(yaw0) + self.off[1], z_sp),
                      yaw=self.yaw_sp)

        if self.man is not None:
            pass                                               # action set by fly()
        elif not (armed and offboard):
            action = 'WAITING FOR ARM/OFFBOARD'
        elif not self.airborne:
            action = 'TAKEOFF'
        elif turn:
            action = f"TURN {'LEFT' if turn < 0 else 'RIGHT'}"
        else:
            action = 'HOVER'
        phase = action.split(' (')[0].split(' 0')[0].rstrip('0123456789./m ')   # without the numbers
        if phase != self.last_action:
            rel = math.degrees(wrap(self.yaw_sp - yaw0))
            self.get_logger().info(f'{action}  (gesture={g}, heading {rel:+.0f} deg from take-off)')
            self.last_action = phase
        if action != getattr(self, 'pub_action', None) or self.counter % 5 == 0:
            self.action_pub.publish(String(data=action))
            self.pub_action = action

    def fly(self, turn, yaw0, dt):
        """react:=fly: go where the operator points and come back.  Returns the action text."""
        r, now = self.react, self.now()
        if self.man is None:
            side = 'LEFT' if turn < 0 else 'RIGHT'
            free = r[f'fly_limit_{side.lower()}']
            if (turn and self.rearm and self.airborne and self.dir_since is not None
                    and now - self.dir_since[1] >= r['fly_trigger']):
                if min(free, r['fly_distance']) < 0.5:
                    return f'HOVER (no free space to the {side}: obstacle)'
                self.man = dict(phase='turn', dir=turn, t=now, dist=min(free, r['fly_distance']),
                                yaw=wrap(yaw0 + turn * math.radians(r['fly_turn'])))
                self.rearm = False
            elif turn and not self.rearm and self.airborne:
                return 'HOVER (lower the arm before the next direction)'
            else:
                return 'HOVER' if self.airborne else 'TAKEOFF'
        m = self.man
        side = 'LEFT' if m['dir'] < 0 else 'RIGHT'
        rate = max(self.yaw_rate, math.radians(30.0)) * dt
        if m['phase'] in ('turn', 'face'):
            target = m['yaw'] if m['phase'] == 'turn' else yaw0
            err = wrap(target - self.yaw_sp)
            if abs(wrap(self.yaw_sp + math.copysign(min(rate, abs(err)), err) - self.pos.heading)) <= math.radians(20):
                self.yaw_sp = wrap(self.yaw_sp + math.copysign(min(rate, abs(err)), err))
            text = f'TURN {side} to fly there' if m['phase'] == 'turn' else 'FACE THE OPERATOR'
            if abs(wrap(target - self.pos.heading)) < math.radians(4) and abs(err) < 1e-3:
                if m['phase'] == 'turn':
                    m.update(phase='fly', t=now)
                else:
                    self.man = None
                    return 'HOVER'
            return text
        if m['phase'] == 'fly':
            goal = (m['dist'] * math.cos(m['yaw']), m['dist'] * math.sin(m['yaw']))
        else:
            goal = (0.0, 0.0)                                  # hold / return
            if m['phase'] == 'hold':
                goal = tuple(self.off)
        step = r['fly_speed'] * dt
        dx, dy = goal[0] - self.off[0], goal[1] - self.off[1]
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            k = min(1.0, step / dist)
            self.off = [self.off[0] + k * dx, self.off[1] + k * dy]
        # how far the vehicle really is from where the setpoint is heading
        hx, hy = self.home[0], self.home[1]
        base = (hx + self.advance * math.cos(yaw0), hy + self.advance * math.sin(yaw0))
        real = math.hypot(self.pos.x - base[0], self.pos.y - base[1])
        lag = math.hypot(self.pos.x - base[0] - goal[0], self.pos.y - base[1] - goal[1])
        if m['phase'] == 'fly':
            if dist < 1e-3 and lag < 0.3:
                m.update(phase='hold', t=now)
            return f'FLY {side} {real:.1f}/{m["dist"]:.1f} m'
        if m['phase'] == 'hold':                           # until /demo/advance ("come back")
            return f'AT THE POINTED PLACE ({side}, {real:.1f} m), waiting for "come back"'
        if dist < 1e-3 and lag < 0.3:                          # return
            m.update(phase='face', t=now)
        return f'RETURN ({real:.1f} m away)'


def wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def main(args=None):
    # Keep the context alive on Ctrl+C so the LAND command can still be published.
    rclpy.init(args=args, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    node = DroneControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.land_on_exit:
            node.land()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
