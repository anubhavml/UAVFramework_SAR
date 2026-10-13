#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Live demonstration driver (started by run_gui_demo.sh; not part of the evaluation).

    demo_flight.py <world> <out dir> [--source synthetic|real] [--mode rgb|depth]
                   [--distances 10,7,5,3] [--gestures right,left,both_out] [--once]
                   [--react turn|fly] [--fly-turn 90] [--fly-distance 5] [--panel]

synthetic: the UAV takes off --distances[0] m in front of the Gazebo person, then moves towards
  them (control node /demo/advance, 1 m/s) and stops at each distance; at every stop the person
  shows each gesture (NEUTRAL in between) and the UAV reacts. Then it flies back and repeats.
real: the drone's camera stream is replaced by UAV-GESTURE clips (the ones of the gesture videos,
  demo_common.REAL), published on /video/camera (+ estimated depth on /video/depth_camera).
  The UAV flies the same approach; it is emulated by digital zoom: each frame is cropped around the
  real person and enlarged by recorded distance (data/real_clip_distances.json, 6.5-8.4 m) /
  current distance, never below 1 (the person cannot appear farther than recorded); depth alike.
A "Demo status" window shows the distance, the gesture shown, the expected and the actual
reaction. The gesture node runs the perception pipeline as launched (--mode): Algorithm 1 of the
paper with the stock YOLOv11n-pose on the RGB or the depth image (rgb, depth), on the
depth-weighted fused pose of Eq. (1)-(2) (fused), or YOLOv11-L_total, RGB-D (rgbd).

--react fly: the UAV goes where the operator points (control node react:=fly: turn --fly-turn deg,
  fly --fly-distance m and hover there until "come back"); --react turn: it only turns (paper).
  The free space either side (obstacles.py, 1.5 m clearance to trees, buildings, walls) is sent
  to the control node, which shortens the flight or stays when there is no room.
--panel: no automatic sequence and no "Demo status" window; a "Demo control" window with buttons
  moves the UAV towards / away from the person, brings it back ("come back"), picks the gesture
  the person shows (the UAV reacts through Algorithm 1), switches the reaction (turn / fly, angle,
  distance) while flying, and stops the whole demo (eval/stop_gui_demo.sh). Once the windows are
  open it brings Gazebo, QGroundControl, the drone camera window and itself in front of the
  component terminals.
--visibility V (camera fog): synthetic: sensor_sim fogs the Gazebo images (launch visibility:=V);
  real: this driver fogs each video frame with the same model (drone_control.sensor_model,
  Koschmieder, airlight 217) using the estimated depth (person at the shown distance).
real + fly: when the UAV has turned the person out of its camera's view (> half the 69 deg field of
  view), the camera stream shows nothing (black), as a real camera would not see the person.
"""
import argparse
import math
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

import demo_common as gv
import simtools as st

MANOEUVRE = ('TURN LEFT to', 'TURN RIGHT to', 'FLY', 'AT THE', 'RETURN', 'FACE')   # react:=fly busy
HFOV_VIDEO = 1.204                                     # rad, the video sensor (launch SENSORS['video'])
HALF_FOV = math.degrees(HFOV_VIDEO) / 2                # deg, OAK-D Lite RGB (and the video sensor)
HOLD = {'right': 7.0, 'left': 7.0, 'rightout_leftup': 7.0, 'leftout_rightup': 7.0}   # s, turns
HOLD_OTHER, NEUTRAL_GAP = 5.0, 3.0
HERE = os.path.dirname(os.path.abspath(__file__))
EXPECT = {'RIGHT': 'turn LEFT', 'LEFT': 'turn RIGHT', 'NEUTRAL': 'hover'}
PIPELINE = {'rgb': 'Algorithm 1, RGB', 'depth': 'Algorithm 1, depth', 'fused': 'Algorithm 1, fused RGB-D pose',
            'rgbd': 'YOLOv11-L_total, RGB-D'}      # --mode -> what the gesture node runs


class Demo(Node):
    def __init__(self, a):
        super().__init__('demo_driver')
        self.a = a
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pos = self.status = None
        self.cmd, self.raw, self.action = '-', '-', '-'
        self.world_xy = None                  # UAV spawn point in the Gazebo world (x, y)
        self.home = None                      # (x, y, heading) once airborne
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position',
                                 lambda m: setattr(self, 'pos', m), qos)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', lambda m: setattr(self, 'status', m), qos)
        self.create_subscription(String, '/gesture/command', lambda m: setattr(self, 'cmd', m.data), 10)
        self.create_subscription(String, '/gesture/frame', self.frame_cb, 10)
        self.create_subscription(String, '/drone_control/action', lambda m: setattr(self, 'action', m.data), 10)
        self.adv_pub = self.create_publisher(Float32, '/demo/advance', 10)
        self.react_pub = self.create_publisher(String, '/demo/react', 10)
        self.react = dict(react=a.react, fly_turn=a.fly_turn, fly_distance=a.fly_distance)
        self.target = a.dist[0]               # distance the UAV is sent to (m from the person)
        self.img_pub = self.create_publisher(Image, '/video/camera', 2)
        self.dep_pub = self.create_publisher(Image, '/video/depth_camera', 2)
        # what the status window shows
        self.state = dict(step='starting', shown='neutral', target=None, clip='')

    def frame_cb(self, m):
        import json
        try:
            self.raw = json.loads(m.data).get('raw', '-')
        except ValueError:
            pass

    def airborne(self):
        return (self.status is not None and self.status.nav_state == 14 and self.pos is not None
                and -self.pos.z > 1.2)

    def ahead(self):
        """metres flown from the take-off point along the take-off heading."""
        if self.home is None or self.pos is None:
            return 0.0
        hx, hy, yaw0 = self.home
        return (self.pos.x - hx) * math.cos(yaw0) + (self.pos.y - hy) * math.sin(yaw0)

    def busy(self):
        """react:=fly manoeuvre in progress"""
        return self.action.startswith(MANOEUVRE)

    def go_to(self, d):
        self.target = d
        self.adv_pub.publish(Float32(data=float(self.a.dist[0] - d)))

    def send_react(self, obstacles=None):
        """reaction settings + free space either side (world obstacle points) to the control node"""
        import json
        msg = dict(self.react)
        if obstacles is not None and self.pos is not None and self.world_xy is not None:
            import obstacles as ob
            # PX4 local NED (x north, y east) -> Gazebo ENU world (x east, y north)
            x, y = self.world_xy[0] + self.pos.y, self.world_xy[1] + self.pos.x
            face = math.pi / 2 - (self.home[2] if self.home else self.pos.heading)
            t = math.radians(self.react['fly_turn'])
            msg.update(fly_limit_left=ob.safe_distance(obstacles, x, y, face + t, 20.0),
                       fly_limit_right=ob.safe_distance(obstacles, x, y, face - t, 20.0))
            self.free = (msg['fly_limit_left'], msg['fly_limit_right'])
        self.react_pub.publish(String(data=json.dumps(msg)))

    def turned(self):
        if self.home is None or self.pos is None:
            return 0.0
        return math.degrees((self.pos.heading - self.home[2] + math.pi) % (2 * math.pi) - math.pi)


def image_msg(img, enc, stamp):
    m = Image(height=img.shape[0], width=img.shape[1], encoding=enc, is_bigendian=0,
              step=img.shape[1] * img.itemsize * (img.shape[2] if img.ndim == 3 else 1),
              data=np.ascontiguousarray(img).tobytes())
    m.header.stamp = stamp
    return m


def status_window(node, a, stop):
    """The 'Demo status' window: 10 Hz."""
    cv2.namedWindow('Demo status', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Demo status', 980, 410)
    if a.panel:                                      # smaller, bottom-left, above the camera window
        cv2.resizeWindow('Demo status', 735, 308)
        cv2.moveWindow('Demo status', 10, 735)
    first = True
    while not stop.is_set():
        img = np.full((410, 980, 3), 30, np.uint8)
        s = node.state
        src = ('SYNTHETIC person in Gazebo' if a.source == 'synthetic' else
               f"REAL person, UAV-GESTURE ({s['clip']})")
        gv.put(img, f'{src}  |  {PIPELINE[a.mode]}', (20, 40), 0.65, gv.WHITE)
        d0 = a.dist[0]
        if a.source == 'real' and 'zoom' in s:
            tgt = f"   (stop at {s['target']:.1f} m)" if s['target'] is not None else ''
            gv.put(img, f"UAV distance: {d0 - node.ahead():4.1f} m{tgt}   person shown at {s['shown_d']:.1f} m",
                   (20, 90), 0.8, gv.WHITE)
            gv.put(img, f"(recorded at {s['rec']:.1f} m, digital zoom x{s['zoom']:.2f}: approach emulated by zoom)",
                   (20, 118), 0.55, gv.GREY, 1)
        if a.source == 'synthetic':
            now = d0 - node.ahead()
            tgt = f"   (stop at {s['target']:.0f} m)" if s['target'] is not None else ''
            gv.put(img, f'Distance to the person: {now:4.1f} m{tgt}', (20, 90), 0.9, gv.WHITE)
        gv.put(img, f"Step: {s['step']}", (20, 150), 0.7, gv.GREY)
        desc = gv.GESTURES.get(s['shown'], ('', s['shown'], 'NEUTRAL', ''))
        want = desc[2]
        gv.put(img, f'Person shows: {desc[1]}', (20, 190), 0.9, gv.CYAN)
        gv.put(img, f'Expected: {want}  ->  UAV should {EXPECT[want]}', (20, 235), 0.8, gv.CYAN)
        gv.put(img, f'Algorithm 1, this frame: {node.raw}', (20, 290), 0.85, gv.ok_colour(node.raw, want))
        gv.put(img, f'Command: {node.cmd}', (560, 290), 0.85, gv.ok_colour(node.cmd, want))
        lim = (f'turn limit +-{a.max_turn:.0f}' if node.react['react'] == 'turn' else
               f"turn {node.react['fly_turn']:.0f}, fly {node.react['fly_distance']:.0f} m and return")
        gv.put(img, f'UAV heading: {node.turned():+4.0f} deg from facing the person'
                    f'  ({lim})', (20, 345), 0.75, gv.WHITE)
        gv.put(img, f'UAV: {node.action}', (20, 385), 0.75,
               gv.CYAN if node.busy() else gv.GREY)
        cv2.imshow('Demo status', img)
        if first and a.panel:
            try:
                cv2.setWindowProperty('Demo status', cv2.WND_PROP_TOPMOST, 1)
            except (cv2.error, AttributeError):
                pass
            first = False
        cv2.waitKey(100)
    cv2.destroyAllWindows()


def wait(node, cond, timeout, stop):
    t0 = time.monotonic()
    while not stop.is_set() and not cond() and time.monotonic() - t0 < timeout:
        time.sleep(0.05)


class Synthetic:
    """the person in Gazebo: show(g) puts the human with gesture g in place, the others parked"""

    def __init__(self, node, a, stop):
        self.node, self.stop = node, stop
        self.gz = st.Gz(a.world, topics=())
        self.gz.wait_clock(0.0, timeout=180)
        _, (self.hx, self.hy, self.hyaw), _ = st.world_info(a.world)
        self.parked = {g: (self.hx, self.hy + 3 * i, st.PARK_Z) for i, g in enumerate(st.GESTURES)}

    def show(self, g):
        hx, hy, hyaw = self.hx, self.hy, self.hyaw
        self.gz.set_poses([(f'human_{o}', *self.parked[o], hyaw) for o in st.GESTURES if o != g]
                          + [(f'human_{g}', hx, hy, 0.0, hyaw)])
        self.node.state['shown'] = g

    def hold(self, seconds):            # simulation seconds
        t = self.gz.sim_t + seconds
        wait(self.node, lambda: self.gz.sim_t >= t, 10 * seconds + 30, self.stop)


class Real:
    """Real UAV-GESTURE clips as the drone's camera, the approach emulated by digital zoom: the UAV
    flies towards the (absent) person as in the synthetic demo, and each frame is cropped around
    the real person and enlarged by recorded distance / current distance (never below 1: the
    person cannot be shown farther than recorded); estimated depth is cropped and scaled alike.
    A thread streams the clip of the current gesture (looping its 5 s); show(g) switches clips.
    When the UAV has turned the person out of its camera's view, it streams an empty view."""

    def __init__(self, node, a, stop):
        import json
        import scipy.io as sio
        from depth_estimator import DepthEstimator, H, W
        self.node, self.a, self.stop, self.W, self.H = node, a, stop, W, H
        here = os.path.dirname(os.path.abspath(__file__))
        self.rec = json.load(open(f'{here}/data/real_clip_distances.json'))   # recorded distance per clip (m)
        # estimated depth: the depth camera (depth, fused, rgbd) and the camera fog (visibility) need it
        self.est = (DepthEstimator(depth_device()) if a.mode != 'rgb' or a.visibility > 0 else None)
        self.vids = {v['video']: v for v in gv.videos()}
        self.centre = {}                # person centre per clip (original pixels): median of the annotated joints
        for g, (vid, start, _) in gv.REAL.items():
            j = sio.loadmat(self.vids[vid]['mat'])['pos_img'][:, :, start:start + gv.CLIP]   # (2, 18, n)
            pts = j.transpose(2, 1, 0).reshape(-1, 2)
            pts = pts[np.abs(pts).sum(1) > 0]
            self.centre[g] = np.median(pts, 0)
        gz = st.Gz(a.world, topics=())
        gz.wait_clock(0.0, timeout=180)
        _, (hx, hy, hyaw), _ = st.world_info(a.world)
        gz.set_poses([(f'human_{g}', hx, hy + 3 * i, st.PARK_Z, hyaw) for i, g in enumerate(st.GESTURES)])
        self.g = 'neutral'
        node.state.update(zoom=1.0, rec=self.rec['neutral'], shown_d=self.rec['neutral'])
        threading.Thread(target=self.stream, daemon=True).start()

    def show(self, g):
        self.g = g
        self.node.state['shown'] = g

    def hold(self, seconds):            # wall-clock seconds (the clips play in real time)
        t = time.monotonic() + seconds
        wait(self.node, lambda: time.monotonic() >= t, seconds + 1, self.stop)

    def frame(self, img, g, mirror, depth_full):
        """crop + zoom for the UAV's current distance -> (image W x H, depth or None, zoom, shown distance)"""
        a, node, W, H = self.a, self.node, self.W, self.H
        cur = a.dist[0] - node.ahead() if node.home is not None else a.dist[0]
        zoom = max(1.0, self.rec[g] / max(cur, 0.5))
        h0, w0 = img.shape[:2]
        cw, ch = w0 / zoom, h0 / zoom
        cx = min(max(self.centre[g][0], cw / 2), w0 - cw / 2)
        cy = min(max(self.centre[g][1], ch / 2), h0 - ch / 2)
        x0, y0 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
        out = cv2.resize(img[y0:y0 + int(ch), x0:x0 + int(cw)], (W, H), interpolation=cv2.INTER_AREA)
        dep = None
        if depth_full is not None:          # same crop in the depth map (W x H of the full frame)
            sx, sy = W / w0, H / h0
            d = depth_full[int(y0 * sy):int((y0 + ch) * sy), int(x0 * sx):int((x0 + cw) * sx)]
            dep = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) / zoom   # person at rec / zoom
        if mirror:
            out = np.ascontiguousarray(out[:, ::-1])
            dep = None if dep is None else np.ascontiguousarray(dep[:, ::-1])
        return out, dep, zoom, self.rec[g] / zoom

    def stream(self):
        node, W, H = self.node, self.W, self.H
        while not self.stop.is_set():
            g = self.g
            vid, start, mirror = gv.REAL[g]
            v = self.vids[vid]
            node.state['clip'] = f"person S{v['subject']}, '{v['cls']}'" + (', mirrored' if mirror else '')
            cap = cv2.VideoCapture(v['path'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            t_next = time.monotonic()
            n, k = 0, 0
            while not self.stop.is_set() and self.g == g:
                ok, img = cap.read()
                if not ok or k >= gv.CLIP:                       # loop the 5 s clip
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    k = 0
                    ok, img = cap.read()
                    if not ok:
                        break
                n, k = n + 1, k + 1
                if n % 2:                                        # 12.5 Hz to the gesture node
                    stamp = node.get_clock().now().to_msg()
                    if abs(node.turned()) > HALF_FOV:            # person outside the camera's view
                        node.state['clip'] = 'person out of the camera view (UAV turned away)'
                        out, dep = np.zeros((H, W, 3), np.uint8), np.full((H, W), np.inf, np.float32)
                    else:
                        node.state['clip'] = (f"person S{v['subject']}, '{v['cls']}'"
                                              + (', mirrored' if mirror else ''))
                        full = (self.est(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA))
                                if self.est else None)
                        out, dep, zoom, shown_d = self.frame(img, g, mirror, full)
                        node.state.update(zoom=zoom, rec=self.rec[g], shown_d=shown_d)
                        if self.a.visibility > 0:                # camera fog, as sensor_sim does
                            from drone_control import sensor_model as sm
                            out = sm.fog_rgb(out, dep, HFOV_VIDEO, self.a.visibility)
                            dep = sm.fog_depth(dep, HFOV_VIDEO, self.a.visibility)
                    node.img_pub.publish(image_msg(out, 'bgr8', stamp))
                    if self.a.mode != 'rgb':
                        node.dep_pub.publish(image_msg(dep.astype(np.float32), '32FC1', stamp))
                t_next += 1 / 25
                time.sleep(max(0.0, t_next - time.monotonic()))
            cap.release()


def depth_device():
    """the depth estimator on the second GPU if there is one (YOLO uses the first), else the first,
    else the CPU"""
    import torch
    n = torch.cuda.device_count()
    return 'cuda:1' if n > 1 else 'cuda:0' if n == 1 else 'cpu'


def take_off(node, a, src, stop):
    src.show('neutral')
    node.state['step'] = 'taking off'
    wait(node, node.airborne, 300, stop)
    if node.pos is not None:
        node.home = (node.pos.x, node.pos.y, node.pos.heading)
    src.hold(4.0)


def react_updates(node, a, stop):
    """free space either side to the control node, 1 Hz (not during a manoeuvre)"""
    import obstacles as ob
    pts = ob.points(a.world)
    while not stop.is_set():
        if not node.busy():
            node.send_react(pts)
        time.sleep(1.0)


def run_auto(node, a, src, stop):
    """the automatic sequence: approach, stop at each distance, show each gesture"""
    take_off(node, a, src, stop)
    d0 = a.dist[0]
    while not stop.is_set():
        for d in a.dist:
            node.state.update(step=f'flying to {d:g} m from the person', target=d)
            node.go_to(d)
            wait(node, lambda: abs(d0 - node.ahead() - d) < 0.25, 60, stop)
            src.hold(2.0)
            for g in a.gestures:
                node.state['step'] = f'at {d:g} m: neutral, then {g}'
                src.show('neutral')
                src.hold(NEUTRAL_GAP)
                src.show(g)
                if node.react['react'] == 'fly' and gv.GESTURES[g][2] != 'NEUTRAL':
                    wait(node, node.busy, HOLD.get(g, HOLD_OTHER) + 5, stop)   # until the UAV sets off
                    src.show('neutral')                                        # the arm goes down
                    node.state['step'] = f'at {d:g} m: the UAV goes where {g} points, then comes back'
                    wait(node, lambda: node.action.startswith('AT THE'), 60, stop)
                    src.hold(2.0)
                    node.go_to(d)                                  # "come back"
                    wait(node, lambda: not node.busy(), 120, stop)
                else:
                    src.hold(HOLD.get(g, HOLD_OTHER))
                    src.show('neutral')
                node.go_to(d)                                    # face the person again
                src.hold(2.0)
        if a.once:
            break
        node.state.update(step='flying back to the start', target=d0)
        node.go_to(d0)
        wait(node, lambda: abs(node.ahead()) < 0.25, 60, stop)
        src.hold(2.0)
    stop.set()


PANEL_GESTURES = [   # (gesture, button text): the person shows it, the UAV reacts through Algorithm 1
    ('right', "Point to the UAV's LEFT\n(right arm out)"),
    ('neutral', 'Arms down\n(hover)'),
    ('left', "Point to the UAV's RIGHT\n(left arm out)"),
    ('rightout_leftup', "UAV's LEFT\n(right arm out, left arm up)"),
    ('both_out', 'Both arms out\n(hover)'),
    ('leftout_rightup', "UAV's RIGHT\n(left arm out, right arm up)"),
    (None, ''),
    ('both_up', 'Both arms up\n(hover)'),
]


def run_panel(node, a, src, stop):
    """the "Demo control" window (Tk, main thread); take-off runs in a thread"""
    import tkinter as tk
    d0 = a.dist[0]
    threading.Thread(target=lambda: (take_off(node, a, src, stop),
                                     node.state.update(step='waiting for your buttons')),
                     daemon=True).start()
    root = tk.Tk()
    root.title('Demo control')
    root.geometry('-10+40')                          # right edge of the screen, always on top
    root.attributes('-topmost', True)
    root.configure(padx=10, pady=8)
    F = ('DejaVu Sans', 11)
    B = ('DejaVu Sans', 11, 'bold')

    def section(text):
        f = tk.LabelFrame(root, text=text, font=B, padx=8, pady=6)
        f.pack(fill='x', pady=4)
        return f

    # --- UAV movement
    mv = section('1. UAV movement (towards / away from the person)')
    dist_lbl = tk.Label(mv, font=F, width=46, anchor='w')
    dist_lbl.grid(row=0, column=0, columnspan=6, sticky='w')
    move_btns = []

    def go(d):
        d = min(max(d, 2.0), d0 + 5.0)
        node.go_to(d)
        node.state.update(step=f'flying to {d:g} m from the person', target=d)

    for i, (txt, fn) in enumerate([('<< Back 1 m', lambda: go(node.target + 1.0)),
                                   ('Forward 1 m >>', lambda: go(node.target - 1.0))]):
        b = tk.Button(mv, text=txt, font=B, width=14, command=fn)
        b.grid(row=1, column=3 * i, columnspan=3, pady=3, sticky='we')
        move_btns.append(b)
    tk.Label(mv, text='Go to:', font=F).grid(row=2, column=0, sticky='w')
    for i, d in enumerate(sorted(set(a.dist) | {d0}, reverse=True)[:5]):
        b = tk.Button(mv, text=f'{d:g} m', font=F, width=6, command=lambda d=d: go(d))
        b.grid(row=2, column=1 + i, padx=2)
        move_btns.append(b)
    b = tk.Button(mv, text="Come back and face the person", font=B, command=lambda: go(node.target))
    b.grid(row=3, column=0, columnspan=6, sticky='we', pady=3)
    move_btns.append(b)

    # --- gesture
    gs = section('2. Direction: the person shows the gesture, the UAV reacts (Algorithm 1)')
    gest_btns = {}
    for i, (g, txt) in enumerate(PANEL_GESTURES):
        if g is None or g not in gv.GESTURES or g not in st.GESTURES:
            continue
        b = tk.Button(gs, text=txt, font=F, width=24, height=2, command=lambda g=g: src.show(g))
        b.grid(row=i // 3, column=i % 3, padx=3, pady=3, sticky='we')
        gest_btns[g] = b
    auto_lower = tk.BooleanVar(value=True)
    tk.Checkbutton(gs, text='lower the arm when the UAV sets off (fly mode)', font=F,
                   variable=auto_lower).grid(row=3, column=0, columnspan=3, sticky='w')

    # --- reaction
    rc = section('3. UAV reaction to LEFT / RIGHT')
    mode = tk.StringVar(value=node.react['react'])
    turn = tk.DoubleVar(value=node.react['fly_turn'])
    fdist = tk.DoubleVar(value=node.react['fly_distance'])

    def changed(*_):
        node.react.update(react=mode.get(), fly_turn=float(turn.get()), fly_distance=float(fdist.get()))
        node.send_react()

    tk.Radiobutton(rc, text=f'Turn only, up to {a.max_turn:.0f} deg while the arm is out (paper)', font=F,
                   variable=mode, value='turn', command=changed).grid(row=0, column=0, columnspan=2, sticky='w')
    tk.Radiobutton(rc, text='Turn, fly there and wait for "come back"', font=F,
                   variable=mode, value='fly', command=changed).grid(row=1, column=0, columnspan=2, sticky='w')
    tk.Label(rc, text='turn (deg)', font=F).grid(row=2, column=0, sticky='w')
    tk.Scale(rc, from_=20, to=90, resolution=5, orient='horizontal', length=300, variable=turn,
             command=changed).grid(row=2, column=1, sticky='w')
    tk.Label(rc, text='fly (m)', font=F).grid(row=3, column=0, sticky='w')
    tk.Scale(rc, from_=1, to=8, resolution=0.5, orient='horizontal', length=300, variable=fdist,
             command=changed).grid(row=3, column=1, sticky='w')

    # --- status
    st_lbl = tk.Label(root, font=F, justify='left', anchor='w', wraplength=760)
    st_lbl.pack(fill='x', pady=4)
    def stop_demo():
        """eval/stop_gui_demo.sh closes every window and terminal of the demo (this one too)"""
        st_lbl.config(text='Stopping the demo: closing Gazebo, PX4, QGroundControl, ROS 2 and the terminals ...')
        root.update()
        subprocess.Popen(['setsid', 'bash', os.path.join(HERE, 'stop_gui_demo.sh')],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

    tk.Button(root, text='Stop demo (close everything)', font=B, bg='#f2b8b5',
              command=stop_demo).pack(anchor='e')
    root.protocol('WM_DELETE_WINDOW', stop_demo)
    last = {'busy': False}

    def tick():
        if stop.is_set():
            root.destroy()
            return
        up = node.home is not None
        now = d0 - node.ahead() if up else d0
        dist_lbl.config(text=f'Distance to the person: {now:4.1f} m   (sent to {node.target:g} m)'
                        if up else 'Taking off ... (buttons work once airborne)')
        for b in move_btns + list(gest_btns.values()):
            b.config(state='normal' if up else 'disabled')
        shown = node.state.get('shown', 'neutral')
        for g, b in gest_btns.items():
            b.config(relief='sunken' if g == shown else 'raised',
                     bg='#9fd3ff' if g == shown else root.cget('bg'))
        busy = node.busy()
        if (busy and not last['busy'] and auto_lower.get()
                and gv.GESTURES.get(shown, ('', '', 'NEUTRAL'))[2] != 'NEUTRAL'):
            src.show('neutral')
        last['busy'] = busy
        free = getattr(node, 'free', None)
        desc = list(gv.GESTURES.get(shown, ('', shown, 'NEUTRAL', '')))
        real = (f"\nreal video: person shown at {node.state['shown_d']:.1f} m (recorded at "
                f"{node.state['rec']:.1f} m, zoom x{node.state['zoom']:.2f}); {node.state.get('clip', '')}"
                if a.source == 'real' and 'zoom' in node.state else '')
        side = {'RIGHT': 'LEFT', 'LEFT': 'RIGHT'}.get(desc[2])
        verb = 'goes' if node.react['react'] == 'fly' else 'turns'
        fog = (f"particle fog x{a.fog_density:g} of the paper's" if a.world.endswith('_fog') else '') + \
              (', ' if a.world.endswith('_fog') and a.visibility > 0 else '') + \
              (f'camera fog, visibility {a.visibility:g} m' if a.visibility > 0 else '')
        st_lbl.config(text=f"{PIPELINE[a.mode]}\n" + (f"Fog: {fog}\n" if fog else '') + f"Person shows: {desc[1]}  ->  expected: {desc[2]}, "
                           f"{f'UAV {verb} {side}' if side else 'UAV hovers'}\n"
                           f"UAV: {node.action}\n"
                           f"Algorithm 1 (this frame): {node.raw}    command: {node.cmd}\n"
                           f"heading {node.turned():+.0f} deg from facing the person"
                           + (f"    free space: left {free[0]:.1f} m, right {free[1]:.1f} m" if free else '')
                           + real)
        root.after(200, tick)

    def to_front():
        """Gazebo, QGroundControl, the drone camera window, then this panel in front of the terminals"""
        import xwin
        for _ in range(60):
            if stop.is_set() or any(t.startswith('Unified-PDD') for _, t in xwin.windows()):
                break
            time.sleep(1.0)
        time.sleep(1.0)
        xwin.raise_in_order(['Gazebo', 'QGroundControl', 'Unified-PDD', 'Demo control'])

    threading.Thread(target=to_front, daemon=True).start()
    root.after(200, tick)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('out')
    ap.add_argument('--source', choices=('synthetic', 'real'), default='synthetic')
    ap.add_argument('--mode', choices=tuple(PIPELINE), default='rgb')
    ap.add_argument('--distances', default='10,7,5,3')
    ap.add_argument('--gestures', default='right,left,both_out')
    ap.add_argument('--max-turn', type=float, default=20.0)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--react', choices=('turn', 'fly'), default='turn')
    ap.add_argument('--fly-turn', type=float, default=90.0)
    ap.add_argument('--fly-distance', type=float, default=5.0)
    ap.add_argument('--panel', action='store_true')
    ap.add_argument('--visibility', type=float, default=0.0)
    ap.add_argument('--fog-density', type=float, default=1.0)
    a = ap.parse_args()
    a.dist = [float(x) for x in a.distances.split(',')]
    a.gestures = a.gestures.split(',')
    unknown = [g for g in a.gestures if g not in gv.GESTURES]
    if unknown:
        sys.exit(f'unknown gesture(s) {unknown}; choose from {list(gv.GESTURES)}')
    os.makedirs(a.out, exist_ok=True)
    rclpy.init()
    node = Demo(a)
    stop = threading.Event()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    if not a.panel:
        threading.Thread(target=status_window, args=(node, a, stop), daemon=True).start()
    try:
        src = (Synthetic if a.source == 'synthetic' else Real)(node, a, stop)
        _, (hx, hy, _), _ = st.world_info(a.world)
        node.world_xy = (hx - a.dist[0], hy)        # spawn point, as in run_gui_demo.sh
        node.send_react()
        threading.Thread(target=react_updates, args=(node, a, stop), daemon=True).start()
        (run_panel if a.panel else run_auto)(node, a, src, stop)
    except KeyboardInterrupt:
        stop.set()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
