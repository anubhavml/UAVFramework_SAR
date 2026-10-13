#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Capture the evaluation frames from a running evaluation world (see run_capture.sh).

For every seeded placement (simtools.placements) and every gesture model (right / left /
neutral), moves the rig and the person, then saves the first rig images rendered after the move:
    data/captures/<world>/<d>m_<k>_<gesture>_<frame>.npz
        rgb (JPEG bytes, q95), depth_oakd / depth_d435i / gt_depth (float16 z-depth, inf = none),
        meta (json: world, d, k, gesture, cam, human, stamp)
"""
import argparse
import json
import os

import cv2
import numpy as np

import simtools as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--distances', type=float, nargs='+', default=[1, 3, 5, 7, 10])
    ap.add_argument('--n', type=int, default=8, help='placements per distance')
    ap.add_argument('--frames', type=int, default=1, help='frames per placement and gesture')
    ap.add_argument('--gap', type=float, default=2.5, help='sim seconds between frames')
    ap.add_argument('--warmup', type=float, default=0.0, help='sim seconds before the first frame')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=f'{st.EVAL}/data/captures')
    a = ap.parse_args()

    out = f'{a.out}/{a.world}'
    os.makedirs(out, exist_ok=True)
    gz = st.Gz(a.world)
    gz.wait_clock(a.warmup)
    _, (hx, hy, _), _ = st.world_info(a.world)
    parked = {g: (hx, hy + 3 * i, st.PARK_Z) for i, g in enumerate(st.GESTURES)}
    places = st.placements(a.world, a.distances, a.n, a.seed)
    n = 0
    with open(f'{out}/index.jsonl', 'a') as index:    # resumable: existing frames are kept
        for p in places:
            names = [f"{p['d']:04.1f}m_{p['k']:02d}_{g}_{f}" for g in st.GESTURES
                     for f in range(a.frames)]
            if all(os.path.exists(f'{out}/{nm}.npz') for nm in names):
                n += len(names)
                continue
            cx, cy, h, aim = p['cam']
            gz.set_pose('rig', cx, cy, h, aim)
            for g in st.GESTURES:
                for o in st.GESTURES:
                    if o != g:
                        gz.set_pose(f'human_{o}', *parked[o], p['human'][2])
                gz.set_pose(f'human_{g}', p['human'][0], p['human'][1], 0.0, p['human'][2])
                t = gz.sim_t + 0.25                        # first frame rendered after the move
                for f in range(a.frames):
                    imgs, stamp = gz.grab(t)
                    meta = dict(world=a.world, d=p['d'], k=p['k'], gesture=g, frame=f,
                                cam=p['cam'], human=p['human'], stamp=stamp)
                    name = f"{p['d']:04.1f}m_{p['k']:02d}_{g}_{f}"
                    ok, jpg = cv2.imencode('.jpg', imgs['rgb'], [cv2.IMWRITE_JPEG_QUALITY, 95])
                    np.savez_compressed(
                        f'{out}/{name}.npz', rgb=np.frombuffer(jpg.tobytes(), np.uint8),
                        meta=json.dumps(meta),
                        **{k: imgs[k].astype(np.float16) for k in st.RIG_TOPICS if k != 'rgb'})
                    index.write(json.dumps(dict(meta, file=f'{name}.npz')) + '\n')
                    n += 1
                    t = stamp + a.gap
            print(f"  {a.world}: d={p['d']} m placement {p['k']} done ({n} frames)", flush=True)
    print(f'{a.world}: {n} frames -> {out}')


if __name__ == '__main__':
    main()
