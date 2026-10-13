#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Real-only extras after build_real.py (PROTOCOL.md, "Amendment ... real-only training"):

  data/yolo_real/{rgb,rgbd}.yaml           real-only dataset definitions (replace any
                                           synthetic + real yaml an earlier build wrote)
  data/yolo_real/rgbd/dist/train/*.txt     T4R distance targets: median of the input depth
                                           channel (estimated depth) over the torso
                                           (shoulders + hips annotated), else over the central
                                           third of the box
  data/yolo_real/clips_train{,_boxes}.npy  T4R clips: 4 labelled frames 3 video frames apart,
                                           one clip starting every 24 frames, same label;
                                           letterboxed 640x384 like build_t4.py
"""
import glob
import os
import sys

import cv2
import numpy as np
import scipy.io as sio

from build_dataset import FAR, FLIP_IDX, NEAR
from build_real import OUT, SPLIT, H, W, DepthEstimator, encode_depth, ug, yolo_row
from build_t4 import LH, LW, PAD

CLIP_EVERY, CLIP_GAP, FRAMES = 24, 3, 4


def yamls():
    for kind, ch in (('rgb', 3), ('rgbd', 4)):
        open(f'{OUT}/{kind}.yaml', 'w').write(
            f'path: {OUT}/{kind}\ntrain: images/train\nval: images/val\nchannels: {ch}\n'
            f'kpt_shape: [17, 3]\nflip_idx: {FLIP_IDX}\nnames:\n  0: LEFT\n  1: RIGHT\n  2: NEUTRAL\n')
    for f in glob.glob(os.path.join(os.path.dirname(OUT), 'yolo', '*_real.yaml')):
        os.remove(f)                                   # the earlier synthetic + real definition


def distances():
    n = 0
    for lab in sorted(glob.glob(f'{OUT}/rgbd/labels/train/*.txt')):
        stem = os.path.basename(lab)[:-4]
        img = cv2.imread(f'{OUT}/rgbd/images/train/{stem}.tiff', cv2.IMREAD_UNCHANGED)
        depth = NEAR + img[..., 3].astype(np.float32) / 255 * (FAR - NEAR)
        out = []
        for row in open(lab).read().split('\n'):
            if not row.strip():
                continue
            v = np.array(row.split()[1:], float)
            cx, cy, bw, bh = v[:4] * [W, H, W, H]
            k = v[4:].reshape(17, 3)[[5, 6, 11, 12]]
            if (k[:, 2] > 0).all():
                x1, y1 = k[:, :2].min(0) * [W, H]
                x2, y2 = k[:, :2].max(0) * [W, H]
            else:
                x1, x2, y1, y2 = cx - bw / 6, cx + bw / 6, cy - bh / 6, cy + bh / 6
            roi = depth[int(y1):max(int(y2), int(y1) + 1), int(x1):max(int(x2), int(x1) + 1)]
            out.append(f'{float(np.median(roi)):.3f}')
            n += 1
        open(f'{OUT}/rgbd/dist/train/{stem}.txt', 'w').write('\n'.join(out) + ('\n' if out else ''))
    print(f'distances: {n} real training labels (from the estimated depth channel)')


def clips():
    est = DepthEstimator()
    frames, boxes = [], []
    for v in ug.videos():
        if v['subject'] not in SPLIT['train']:
            continue
        p = sio.loadmat(v['mat'])['pos_img']
        lab, _, _ = ug.frame_labels(p)
        starts = [s for s in range(0, len(lab) - CLIP_GAP * (FRAMES - 1), CLIP_EVERY)
                  if lab[s] is not None and all(lab[s + CLIP_GAP * i] == lab[s] for i in range(FRAMES))]
        if not starts:
            continue
        cap = cv2.VideoCapture(v['path'])
        want = {s + CLIP_GAP * i: (s, i) for s in starts for i in range(FRAMES)}
        got = {}
        f = 0
        while cap.grab():
            if f in want:
                _, img = cap.retrieve()
                sx, sy = W / img.shape[1], H / img.shape[0]
                r = yolo_row(p[:, :, f].T, lab[f], sx, sy)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                x = cv2.resize(np.dstack([img, encode_depth(est(img))]), (LW, LH - 2 * PAD),
                               interpolation=cv2.INTER_AREA)
                canvas = np.full((LH, LW, 4), 114, np.uint8)
                canvas[PAD:LH - PAD] = x
                got[want[f]] = (canvas, [r[1] * LW, r[2] * (LH - 2 * PAD) + PAD, r[3] * LW, r[4] * (LH - 2 * PAD)])
            f += 1
        for s in starts:
            frames.append(np.stack([got[(s, i)][0] for i in range(FRAMES)]))
            boxes.append([got[(s, i)][1] for i in range(FRAMES)])
        print(v['video'], len(starts), 'clips', flush=True)
    np.save(f'{OUT}/clips_train.npy', np.stack(frames))
    np.save(f'{OUT}/clips_train_boxes.npy', np.array(boxes, np.float32))
    print(f'clips: {len(frames)} real training clips')


if __name__ == '__main__':
    steps = sys.argv[1:] or ['yamls', 'distances', 'clips']
    for s in steps:
        globals()[s]()
