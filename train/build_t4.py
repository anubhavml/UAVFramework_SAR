#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""T4 inputs (PROTOCOL.md, "T4 clip capture and training layout"), on top of build_dataset.py:

    data/yolo/rgbd/dist/train/<stem>.txt   person distance (m) per label row of the training
                                           images: median rendered depth over the projected
                                           torso (shoulders + hips in view), else over the
                                           central third of the person box
    data/yolo/clips_{train,val}.npy        (clips, 4, 384, 640, 4) uint8: B, G, R + depth
                                           (one depth sensor per clip, as build_dataset.py),
                                           1280x720 letterboxed to 640x384 (pad 114)
    data/yolo/clips_{train,val}_boxes.npy  (clips, 4, 4) person box cx, cy, w, h in those pixels
Labels come from build_dataset.label (the same function as the T1-T3 labels); clips in which
any frame has no labelled person are left out.
"""
import glob
import json
import os

import cv2
import numpy as np

from build_dataset import DEPTH, FAR, H, HERE, NEAR, OUT, RGB_HFOV, W, label, load_obj, sm

TORSO = (5, 6, 11, 12)
LW, LH, PAD = 640, 384, 12                     # letterbox 1280x720 -> 640x360 + 12 px top/bottom


def humans_meshes():
    humans = {h['name']: h for h in json.load(open(f'{HERE}/models/humans.json'))}
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = load_obj(f'{HERE}/models/{name}/meshes/human.obj')
        return humans[name], cache[name]
    return get


def clip_depth(depths, sensor, rng):
    """build_dataset.depth_channel with the sensor fixed for the whole clip (noise per frame)."""
    key, hfov = DEPTH[sensor]
    d = depths[key]
    if sensor == 'd435i':
        d = sm.d435i_depth_noise(d, hfov, rng)
    a = sm.align_depth(d, hfov, (W, H), RGB_HFOV)
    a = np.nan_to_num(np.asarray(a, np.float32), nan=FAR, posinf=FAR, neginf=FAR)
    return (np.clip((a - NEAR) / (FAR - NEAR), 0, 1) * 255).astype(np.uint8)


def torso_distance(row, gt):
    v = np.array(row.split()[1:], float)
    cx, cy, bw, bh = v[:4] * [W, H, W, H]
    kp = v[4:].reshape(17, 3)
    t = kp[list(TORSO)]
    t = t[t[:, 2] == 2]                        # visible (not hidden behind something)
    gh, gw = gt.shape
    if len(t) >= 2 and np.ptp(t[:, 0]) * W > 2 and np.ptp(t[:, 1]) * H > 2:
        x1, x2 = t[:, 0].min() * gw, t[:, 0].max() * gw
        y1, y2 = t[:, 1].min() * gh, t[:, 1].max() * gh
    else:
        x1, x2 = (cx - bw / 6) / W * gw, (cx + bw / 6) / W * gw
        y1, y2 = (cy - bh / 6) / H * gh, (cy + bh / 6) / H * gh
    roi = gt[int(y1):max(int(y2), int(y1) + 1), int(x1):max(int(x2), int(x1) + 1)]
    roi = roi[np.isfinite(roi) & (roi > 0.05)]
    return float(np.median(roi)) if roi.size else float('nan')


def distances(get):
    os.makedirs(f'{OUT}/rgbd/dist/train', exist_ok=True)
    n = bad = 0
    for f in sorted(glob.glob(f'{HERE}/data/captures/train_*/*.npz')):
        z = np.load(f)
        meta = json.loads(str(z['meta']))
        stem = f"{meta['world']}_{meta['i']:05d}"
        row = None
        if meta['human']:
            h, mesh = get(meta['human'])
            row = label(meta, h, mesh, z['gt_depth'].astype(np.float32))
        lab = open(f'{OUT}/rgbd/labels/train/{stem}.txt').read().strip()
        assert (row or '') == lab, f'{stem}: label differs from the T1-T3 dataset'
        txt = ''
        if row:
            d = torso_distance(row, z['gt_depth'].astype(np.float32))
            if not np.isfinite(d):
                d, bad = float(meta['d']), bad + 1              # no valid depth pixel: geometry
            txt, n = f'{d:.3f}\n', n + 1
        open(f'{OUT}/rgbd/dist/train/{stem}.txt', 'w').write(txt)
    print(f'distances: {n} labelled training images ({bad} from the placement geometry)')


def clips(get, split, worlds=None):
    worlds = worlds or sorted(w for w in os.listdir(f'{HERE}/data/clips')
                              if w.startswith('val' if split == 'val' else 'train'))
    keep = []
    for w in worlds:
        ids = sorted({os.path.basename(f)[:4] for f in glob.glob(f'{HERE}/data/clips/{w}/*.npz')})
        for c in ids:
            frames = []
            for i in range(4):
                z = np.load(f'{HERE}/data/clips/{w}/{c}_{i}.npz')
                meta = json.loads(str(z['meta']))
                h, mesh = get(meta['human'])
                row = label(meta, h, mesh, z['gt_depth'].astype(np.float32))
                if row is None:
                    break
                frames.append((z, row))
            if len(frames) == 4:
                keep.append((w, c, frames))
    arr = np.lib.format.open_memmap(f'{OUT}/clips_{split}.npy', 'w+', np.uint8, (len(keep), 4, LH, LW, 4))
    boxes = np.zeros((len(keep), 4, 4), np.float32)
    for k, (w, c, frames) in enumerate(keep):
        rng = np.random.default_rng([sum(map(ord, w)), int(c), 3])
        sensor = 'oakd' if rng.random() < 0.5 else 'd435i'      # one sensor per clip
        for i, (z, row) in enumerate(frames):
            rgb = cv2.resize(cv2.imdecode(z['rgb'], cv2.IMREAD_COLOR), (W, H), interpolation=cv2.INTER_AREA)
            depths = {key: z[key].astype(np.float32) for key in ('depth_oakd', 'depth_d435i', 'gt_depth')}
            dch = clip_depth(depths, sensor, rng)
            img = cv2.resize(np.dstack([rgb, dch]), (LW, LH - 2 * PAD), interpolation=cv2.INTER_AREA)
            arr[k, i] = 114
            arr[k, i, PAD:LH - PAD] = img
            cx, cy, bw, bh = np.array(row.split()[1:5], float)
            boxes[k, i] = [cx * LW, cy * (LH - 2 * PAD) + PAD, bw * LW, bh * (LH - 2 * PAD)]
    arr.flush()
    np.save(f'{OUT}/clips_{split}_boxes.npy', boxes)
    print(f'clips {split}: {len(keep)} kept of {sum(len({os.path.basename(f)[:4] for f in glob.glob(f"{HERE}/data/clips/{w}/*.npz")}) for w in worlds)}')


def main():
    get = humans_meshes()
    distances(get)
    for split in ('train', 'val'):
        clips(get, split)


if __name__ == '__main__':
    main()
