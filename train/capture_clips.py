#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Capture short clips for T4's temporal loss from a running training world (run_capture_clips.sh).

Per clip (PROTOCOL.md, "T4 clip capture"): one training human of the world's split at a random
placement (capture_train.placement, always with a person), then FRAMES consecutive frames
~0.1 s simulation time apart; between frames the camera drifts like a hovering drone
(position random walk sigma 2 cm, yaw 0.5 deg, pitch 0.3 deg) and the person sways (yaw 1 deg).
    data/clips/<world>/<clip>_<f>.npz   same format as capture_train.py (+ meta clip, f)
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
from capture_train import PARK_Z, placement  # noqa: E402

FRAMES = 4
STEP = 0.1                                    # simulation seconds between frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--warmup', type=float, default=3.0)
    ap.add_argument('--out', default=f'{HERE}/data/clips')
    a = ap.parse_args()
    info = json.load(open(f'{HERE}/worlds/{a.world}.json'))
    labels = {h['name']: h['label'] for h in json.load(open(f'{HERE}/models/humans.json'))}
    out = f'{a.out}/{a.world}'
    os.makedirs(out, exist_ok=True)
    gz = st.Gz(a.world)
    gz.wait_clock(a.warmup, timeout=300)
    rng = np.random.default_rng([sum(map(ord, a.world)), 11])      # not the single frames' seed
    shown = None
    for c in range(a.n):
        human = None
        while human is None:                                       # clips always show a person
            human, hp, cam, d = placement(rng, info['humans'], info['obstacles'])
        jit = rng.normal(size=(FRAMES, 5)) * [0.02, 0.02, 0.02, math.radians(0.5), math.radians(0.3)]
        sway = rng.normal(size=FRAMES) * math.radians(1.0)
        jit[0], sway[0] = 0, 0
        cams, yaws = np.cumsum(jit, 0) + cam, np.cumsum(sway) + hp[2]
        if all(os.path.exists(f'{out}/{c:04d}_{f}.npz') for f in range(FRAMES)):   # resumable
            continue
        for f in range(FRAMES):
            moves = [('rig', *cams[f]), (human, hp[0], hp[1], 0.0, yaws[f])]
            if f == 0 and shown and shown != human:
                moves.append((shown, 0.0, 3.0 * info['humans'].index(shown), PARK_Z, 0.0))
            gz.set_poses(moves)
            shown = human
            imgs, stamp = gz.grab(gz.sim_t + (0.25 if f == 0 else STEP))
            meta = dict(world=a.world, clip=c, f=f, i=c * FRAMES + f, human=human,
                        label=labels[human], human_pose=(hp[0], hp[1], float(yaws[f])),
                        cam=[float(v) for v in cams[f]], d=d, stamp=stamp)
            ok, jpg = cv2.imencode('.jpg', imgs['rgb'], [cv2.IMWRITE_JPEG_QUALITY, 95])
            np.savez_compressed(f'{out}/{c:04d}_{f}.npz', rgb=np.frombuffer(jpg.tobytes(), np.uint8),
                                meta=json.dumps(meta),
                                **{k: imgs[k].astype(np.float16) for k in st.RIG_TOPICS if k != 'rgb'})
        if c % 10 == 0:
            print(f'{a.world}: clip {c}/{a.n}', flush=True)
    print(f'{a.world}: done')


if __name__ == '__main__':
    main()
