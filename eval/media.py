#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Documentation images and flight videos (not part of the evaluation).

    media.py stills <world> <out_dir>   start Gazebo headless, render the still images, stop it
    media.py world <world>              write the flight world with the third-person camera
    media.py record <world> <out_dir>   during a flight: save third-person + onboard frames
    media.py video <out_dir>            frames + flight log -> <out_dir>/<name>.mp4
    media.py sensors <capture.npz> <out_dir>  one evaluation frame as the pipelines saw it

Stills per world: overview, the flight scene (drone hovering 5 m from the person), the view
from the drone's camera position, the third-person view used in videos; plus the three gesture
models and both simulated drones (x500_depth with OakD-Lite, x500_d435i) close up.
"""
import glob
import json
import math
import os
import re
import signal
import subprocess
import sys

import cv2
import numpy as np

import simtools as st

UAVS = {'uav_oakd': 'x500_depth', 'uav_d435i': 'x500_d435i'}
PARKED_UAV = lambda i: (0.0, 6.0 * i, st.PARK_Z)                       # noqa: E731
# per world: overview target and distance (m) for the elevated view
# per world: overview target, distance (m) and direction (unit x, y) of the elevated view;
# which side of the drone the third-person camera is on (keeps trees out of the way)
OVERVIEW = {'forest': ((8.5, 0.0, 0.0), 16, (-0.95, -0.3)),
            'city': ((14.0, 0.0, 0.0), 32, (-0.6, 0.8)),
            'wall': ((3.0, -3.0, 0.0), 17, (-0.78, -0.62))}
SIDE = {'forest': -1.0}


def look(cam, target):
    """camera position and the point it looks at -> (x, y, z, yaw, pitch) for set_poses."""
    dx, dy, dz = (t - c for t, c in zip(target, cam))
    return (*cam, math.atan2(dy, dx), math.atan2(-dz, math.hypot(dx, dy)))


def chase_pose(world):
    """Third-person camera behind and beside the drone (which hovers 5 m in front of the
    person, at x = hx - 5): both stay in view while the drone turns +-20 deg."""
    _, (hx, hy, _), _ = st.world_info(world)
    side = SIDE.get(world.replace('_fog', ''), 1.0)
    return look((hx - 8.5, hy + 2.6 * side, 2.6), (hx - 2.8, hy, 1.2))


def include(uri, name, pose, static=False):
    return (f'<include>\n      <uri>model://{uri}</uri>\n      <name>{name}</name>\n'
            f'      <pose>{pose}</pose>\n' + ('      <static>true</static>\n' if static else '')
            + '    </include>')


def pose_str(x, y, z, yaw=0.0, pitch=0.0):
    return f'{x:.3f} {y:.3f} {z:.3f} 0 {pitch:.4f} {yaw:.4f}'


def media_world(world, chase=False, stills=False):
    """Evaluation world (three parked gesture humans) plus the third-person camera (flight
    videos) or the photo camera and both drones as static models (stills)."""
    path = st.make_eval_world(world, f'{st.EVAL}/data/worlds', rig=False,
                              suffix='_media' if stills else '_video')
    extra = []
    if chase:
        extra.append(include('chase_cam', 'chase_cam', pose_str(*chase_pose(world))))
    if stills:
        extra.append(include('photo_cam', 'photo_cam', pose_str(0, 0, st.PARK_Z)))
        extra += [include(m, n, pose_str(*PARKED_UAV(i)), static=True)
                  for i, (n, m) in enumerate(UAVS.items())]
    s = open(path).read()
    i = s.rindex('</world>')
    open(path, 'w').write(s[:i] + '  ' + '\n    '.join(extra) + '\n  ' + s[i:])
    return path


def shots(world):
    """[(file name, photo camera pose, {model: pose})]; models not listed are parked."""
    _, (hx, hy, hyaw), _ = st.world_info(world)
    base = world.replace('_fog', '')
    person = lambda g: {f'human_{g}': (hx, hy, 0.0, hyaw)}             # noqa: E731
    hover = {'uav_oakd': (hx - 5.0, hy, 1.5, 0.0)}                     # flight start pose
    tgt, dist, (ux, uy) = OVERVIEW[base]
    side = SIDE.get(base, 1.0)
    out = [(f'{world}_overview', look((tgt[0] + dist * ux, tgt[1] + dist * uy, dist * 0.6),
                                      tgt), {**person('right'), **hover}),
           (f'{world}_scene', look((hx - 8.0, hy - 3.2 * side, 2.6), (hx - 2.5, hy, 1.1)),
            {**person('right'), **hover}),
           (f'{world}_onboard_view', look((hx - 4.88, hy + 0.03, 1.74), (hx, hy, 1.6)),
            person('right')),
           (f'{world}_video_camera', chase_pose(world), {**person('right'), **hover})]
    if world == 'city':
        out += [(f'gesture_{g}', look((hx - 3.5, hy, 1.4), (hx, hy, 1.05)), person(g))
                for g in st.GESTURES]
    if world == 'wall':
        for n in UAVS:
            p = (hx - 5.0, hy, 1.2)
            out.append((f'drone_{UAVS[n]}', look((p[0] + 1.25, p[1] - 1.05, p[2] + 0.75),
                                                 (p[0], p[1], p[2] + 0.2)),
                        {n: (*p, 0.0)}))
            out.append((f'drone_{UAVS[n]}_camera', look((p[0] + 0.75, p[1] - 0.3, p[2] + 0.4),
                                                        (p[0] + 0.12, p[1], p[2] + 0.24)),
                        {n: (*p, 0.0)}))
    return out


def stills(world, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sdf = media_world(world, stills=True)
    log = open(f'{out_dir}/gz_{world}.log', 'w')
    gz_proc = subprocess.Popen(['gz', 'sim', '-s', '-r', '--headless-rendering', sdf],
                               stdout=log, stderr=log, start_new_session=True)
    try:
        gz = st.Gz(world, topics=('rgb',), prefix='photo/')
        gz.wait_clock(1.0, timeout=180)
        _, (hx, hy, hyaw), _ = st.world_info(world)
        parked = {f'human_{g}': (hx, hy + 3 * i, st.PARK_Z, hyaw)
                  for i, g in enumerate(st.GESTURES)}
        parked.update({n: (*PARKED_UAV(i), 0.0) for i, n in enumerate(UAVS)})
        for name, cam, models in shots(world):
            gz.set_poses([('photo_cam', *cam)] + [(n, *models.get(n, p))
                                                  for n, p in parked.items()])
            img, _ = gz.grab(after=gz.sim_t + 1.0, timeout=120)     # rendered after the move
            cv2.imwrite(f'{out_dir}/{name}.png', img['rgb'])
            print(name, flush=True)
    finally:
        os.killpg(gz_proc.pid, signal.SIGKILL)
        gz_proc.wait()


def sensors(npz, out_dir):
    """An evaluation capture under every condition, exactly as evaluate.py builds them: RGB
    clear and with Koschmieder fog, and the depth colormap images (OakD-Lite, D435i + noise)
    that the depth pipelines run on."""
    from drone_control import pdd
    import evaluate as ev
    os.makedirs(out_dir, exist_ok=True)
    meta, rgb, depths = ev.load(npz)
    tag = f"{meta['world']}_{meta['d']:g}m_{meta['gesture']}"
    for cond, img, dd in ev.conditions(meta, rgb, depths, [50, 20, 10, 5], seed=1):
        cv2.imwrite(f'{out_dir}/sensors_{tag}_rgb_{cond}.png', img)
        for sensor, depth in dd.items():
            if cond in ('clear', 'particle', 'phys10'):
                cv2.imwrite(f'{out_dir}/sensors_{tag}_depth_{sensor}_{cond}.png',
                            pdd.preprocess_depth(depth)[1])
    print(tag)


def record(world, out_dir):
    """Saves every third-person frame (Gazebo, stamped with its simulation time) and every
    annotated onboard frame (/gesture/annotated, stamped with /clock on arrival) as JPEG."""
    import rclpy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Image as RosImage
    gz = st.Gz(world, topics=())
    from gz.msgs10.image_pb2 import Image
    fdir = f'{out_dir}/frames'
    os.makedirs(fdir, exist_ok=True)
    q = [cv2.IMWRITE_JPEG_QUALITY, 88]

    def chase(m):
        t = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        cv2.imwrite(f'{fdir}/chase_{t:09.3f}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), q)
    gz.node.subscribe(Image, 'chase/rgb', chase)

    rclpy.init()
    node = rclpy.create_node('media_recorder')
    clock = [None]
    node.create_subscription(Clock, '/clock', lambda m: clock.__setitem__(
        0, m.clock.sec + m.clock.nanosec * 1e-9), 10)

    def onboard(m):
        if clock[0] is None:
            return
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        cv2.imwrite(f'{fdir}/onboard_{clock[0]:09.3f}.jpg', cv2.resize(img, (1280, 720)), q)
    node.create_subscription(RosImage, '/gesture/annotated', onboard, 2)
    signal.signal(signal.SIGTERM, lambda *_: rclpy.shutdown())
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass


def frames(fdir, kind):
    files = sorted(glob.glob(f'{fdir}/{kind}_*.jpg'))
    return np.array([float(re.search(r'_([\d.]+)\.jpg$', f).group(1)) for f in files]), files


def latest(times, values, t):
    i = np.searchsorted(times, t, side='right') - 1
    return values[i] if i >= 0 else None


def text(img, s, org, scale=0.9, colour=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def video(out_dir, fps=15.0):
    """Side by side at simulation speed (1 s of video = 1 s simulated): third-person view and
    the drone's annotated camera, with the gesture shown, the command and the heading."""
    name = os.path.basename(out_dir.rstrip('/'))
    ct, cf = frames(f'{out_dir}/frames', 'chase')
    ot, of = frames(f'{out_dir}/frames', 'onboard')
    ev = [json.loads(line) for line in open(f'{out_dir}/log.jsonl')]
    sw = [(e['sim'], e['data']) for e in ev if e['kind'] == 'swap']
    cm = [(e['sim'], e['data']) for e in ev if e['kind'] == 'cmd']
    ps = [(e['sim'], e['data']) for e in ev if e['kind'] == 'pos' and e['data'][4] is not None]
    st_t, st_v = np.array([s[0] for s in sw]), [s[1] for s in sw]
    cm_t, cm_v = np.array([c[0] for c in cm]), [c[1] for c in cm]
    ps_t, ps_v = np.array([p[0] for p in ps]), [p[1] for p in ps]
    end = next(e['sim'] for e in ev if e['kind'] == 'end')
    lift = next(t for t, p in ps if p[4] > 0.5)                    # leaves the ground
    h0 = latest(ps_t, ps_v, lift)[3]
    t0, t1 = max(ct[0], lift - 3.0), min(ct[-1], end + 1.0)
    W, H, BAR = 1280, 720, 80
    path = f'{out_dir}/{name}.mp4'
    enc = subprocess.Popen([ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
                            '-pix_fmt', 'bgr24', '-s', f'{2 * W}x{H + BAR}', '-r', str(fps),
                            '-i', '-', '-vf', 'scale=1920:-2', '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', '-crf', '21', path], stdin=subprocess.PIPE)
    cache = {}

    def load(f):
        if f not in cache:
            cache.clear()
            cache[f] = cv2.imread(f) if f else None
        return cache[f]
    blank = np.zeros((H, W, 3), np.uint8)
    for t in np.arange(t0, t1, 1.0 / fps):
        left = load(latest(ct, cf, t)).copy()
        o = latest(ot, of, t)
        right = cv2.imread(o) if o else blank.copy()
        text(left, 'Third-person view (Gazebo, simulation time)', (20, 40))
        text(right, "Drone camera + gesture detection", (20, H - 25), 0.8)
        bar = np.full((BAR, 2 * W, 3), 40, np.uint8)
        shown = latest(st_t, st_v, t)
        cmd = latest(cm_t, cm_v, t)
        p = latest(ps_t, ps_v, t)
        text(bar, f'{t - lift:+5.1f} s after take-off', (20, 52))
        text(bar, f'Person shows: {st.LABEL.get(shown, "-")}', (420, 52), 1.0, (120, 230, 255))
        text(bar, f'Command sent: {cmd or "-"}', (1000, 52), 1.0,
             (120, 255, 120) if cmd == st.LABEL.get(shown) else (120, 120, 255))
        if p:      # PX4 heading is clockwise from north: negative change = turned left
            turn = (p[3] - h0 + 180) % 360 - 180
            way = 'left' if turn < -0.5 else 'right' if turn > 0.5 else ''
            text(bar, f'Turned {abs(turn):3.0f} deg {way:5s}  height {p[4]:.2f} m', (1580, 52))
        enc.stdin.write(np.vstack([np.hstack([left, right]), bar]).tobytes())
    enc.stdin.close()
    enc.wait()
    print(path)


def ffmpeg():
    """ffmpeg with libx264 (OpenCV's own writer has no H.264 here): imageio-ffmpeg's static
    binary, from IMAGEIO_FFMPEG_PATH (a pip --target directory) or the Python path."""
    sys.path.insert(0, os.environ.get('IMAGEIO_FFMPEG_PATH', ''))
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


if __name__ == '__main__':
    cmd, *args = sys.argv[1:]
    if cmd == 'stills':
        stills(*args)
    elif cmd == 'world':
        print(media_world(args[0], chase=True))
    elif cmd == 'record':
        record(*args)
    elif cmd == 'video':
        video(*args)
    elif cmd == 'sensors':
        sensors(*args)
    else:
        sys.exit(__doc__)
