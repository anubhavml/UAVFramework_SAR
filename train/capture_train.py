#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Capture training / validation frames from a running training world (see run_capture_train.sh).

Per frame: a random training human of the world's split at a random spot and heading near the
origin, the camera rig at 0.8-12 m within +-55 deg of where the person faces, at 1.4-2.5 m
height, aimed at them +-10 deg (pitch -5..+8 deg); 10% of frames have nobody in view.
Same file format as eval/capture.py, plus the poses needed for labelling:
    data/captures/<world>/<i>.npz   rgb (JPEG), depth_oakd, depth_d435i, gt_depth (float16), meta
    meta: world, i, human (name or null), label, human_pose (x, y, yaw), cam (x, y, z, yaw, pitch)
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'eval'))
import simtools as st  # noqa: E402

PARK_Z = -30.0


def blocked(x, y, obstacles, margin):
    for o in obstacles:
        if o['kind'] == 'tree':
            if math.hypot(x - o['x'], y - o['y']) < o['r'] + margin:
                return True
        else:
            c, s = math.cos(-o['yaw']), math.sin(-o['yaw'])
            dx, dy = x - o['x'], y - o['y']
            lx, ly = c * dx - s * dy, s * dx + c * dy
            if abs(lx) < o['hx'] + margin and abs(ly) < o['hy'] + margin:
                return True
    return False


def placement(rng, humans, obstacles):
    while True:
        r, a = 2.5 * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        hx, hy, hyaw = r * math.cos(a), r * math.sin(a), rng.uniform(-math.pi, math.pi)
        d = rng.uniform(0.8, 12.0)
        bearing = hyaw + math.radians(rng.uniform(-55, 55))
        cx, cy = hx + d * math.cos(bearing), hy + d * math.sin(bearing)
        if blocked(hx, hy, obstacles, 0.4) or blocked(cx, cy, obstacles, 0.8):
            continue
        aim = math.atan2(hy - cy, hx - cx) + math.radians(rng.uniform(-10, 10))
        cam = (cx, cy, rng.uniform(1.4, 2.5), aim, math.radians(rng.uniform(-5, 8)))
        human = None if rng.random() < 0.1 else humans[rng.integers(len(humans))]
        return human, (hx, hy, hyaw), cam, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--warmup', type=float, default=3.0)
    ap.add_argument('--out', default=f'{HERE}/data/captures')
    a = ap.parse_args()
    info = json.load(open(f'{HERE}/worlds/{a.world}.json'))
    labels = {h['name']: h['label'] for h in json.load(open(f'{HERE}/models/humans.json'))}
    out = f'{a.out}/{a.world}'
    os.makedirs(out, exist_ok=True)
    gz = st.Gz(a.world)
    gz.wait_clock(a.warmup, timeout=300)
    rng = np.random.default_rng([sum(map(ord, a.world)), 7])
    places = [placement(rng, info['humans'], info['obstacles']) for _ in range(a.n)]
    shown = None
    for i, (human, hp, cam, d) in enumerate(places):
        if os.path.exists(f'{out}/{i:05d}.npz'):            # resumable
            continue
        moves = [('rig', *cam)]
        if shown and shown != human:
            moves.append((shown, 0.0, 3.0 * info['humans'].index(shown), PARK_Z, 0.0))
        if human:
            moves.append((human, hp[0], hp[1], 0.0, hp[2]))
        gz.set_poses(moves)
        shown = human
        imgs, stamp = gz.grab(gz.sim_t + 0.25)
        meta = dict(world=a.world, i=i, human=human, label=labels.get(human, 'NONE'),
                    human_pose=hp, cam=cam, d=d, stamp=stamp)
        ok, jpg = cv2.imencode('.jpg', imgs['rgb'], [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.savez_compressed(f'{out}/{i:05d}.npz', rgb=np.frombuffer(jpg.tobytes(), np.uint8),
                            meta=json.dumps(meta),
                            **{k: imgs[k].astype(np.float16) for k in st.RIG_TOPICS if k != 'rgb'})
        if i % 50 == 0:
            print(f'{a.world}: {i}/{a.n}', flush=True)
    print(f'{a.world}: done')


if __name__ == '__main__':
    main()
