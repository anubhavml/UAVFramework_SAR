#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Real training / validation images from UAV-GESTURE (PROTOCOL.md, "Training with real
images: T1R–T4R") -> data/yolo_real/{rgb,rgbd}/{images,labels}/{train,val} and data/yolo_real/{rgb,rgbd}.yaml
(real only; the T4R distance targets and clips: build_real_t4.py).

Labels from the authors' joint annotations (eval/uavgesture.frame_labels); depth estimated
with Depth-Anything-V2-Metric-Outdoor-Small (estimate_depth, also used by the real test).
"""
import json
import os
import sys

import cv2
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(WS, 'eval'))
import uavgesture as ug  # noqa: E402
from build_dataset import FAR, FLIP_IDX, NEAR  # noqa: E402

SPLIT = {'train': (4, 6, 9, 12, 15), 'val': (5, 13), 'test': (1, 3, 11, 14)}
STEP = 6
W, H = 1280, 720
OUT = f'{HERE}/data/yolo_real'
# OpenPose COCO-18 index for each of the 17 COCO keypoints
OP = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
CLASSES = {'LEFT': 0, 'RIGHT': 1, 'NEUTRAL': 2}
SWAP = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT', 'NEUTRAL': 'NEUTRAL'}
from depth_estimator import DEPTH_MODEL, DepthEstimator  # noqa: E402,F401




def encode_depth(d):
    """Metres -> the depth channel of the RGB-D images (as build_dataset / rgbd_model.to_rgbd)."""
    return (np.clip((np.nan_to_num(d, nan=FAR) - NEAR) / (FAR - NEAR), 0, 1) * 255).astype(np.uint8)


def yolo_row(j, label, sx, sy):
    """j: (18, 2) joint pixels of the original video (0 = missing) -> label row at W x H."""
    ok = np.abs(j).sum(1) > 0
    p = j[ok]
    x1, y1 = p.min(0)
    x2, y2 = p.max(0)
    w, h = x2 - x1, y2 - y1
    head = 0.6 * (j[1, 1] - j[0, 1]) if ok[0] and ok[1] else 0.1 * h
    x1, x2 = x1 - 0.1 * w, x2 + 0.1 * w
    y1, y2 = y1 - max(head, 0), y2 + 0.05 * h
    x1, x2 = np.clip([x1 * sx, x2 * sx], 0, W)
    y1, y2 = np.clip([y1 * sy, y2 * sy], 0, H)
    kps = []
    for i in OP:
        kps += [j[i, 0] * sx / W, j[i, 1] * sy / H, 2] if ok[i] else [0.0, 0.0, 0]
    return [CLASSES[label], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H] + kps


def mirror_row(r):
    cls = CLASSES[SWAP[{v: k for k, v in CLASSES.items()}[r[0]]]]
    k = np.array(r[5:], float).reshape(17, 3)[FLIP_IDX]
    k[:, 0] = np.where(k[:, 2] > 0, 1 - k[:, 0], 0)
    return [cls, 1 - r[1], r[2], r[3], r[4]] + k.ravel().tolist()


def fmt(r):
    return ' '.join(str(int(v)) if i == 0 or (i > 4 and (i - 5) % 3 == 2) else f'{v:.6f}'
                    for i, v in enumerate(r))


def main():
    est = DepthEstimator()
    stats = {}
    for split in ('train', 'val'):
        for kind in ('rgb', 'rgbd'):
            for sub in ('images', 'labels'):
                os.makedirs(f'{OUT}/{kind}/{sub}/{split}', exist_ok=True)
        os.makedirs(f'{OUT}/rgbd/dist/{split}', exist_ok=True)
        for v in ug.videos():
            if v['subject'] not in SPLIT[split]:
                continue
            p = sio.loadmat(v['mat'])['pos_img']
            lab, _, _ = ug.frame_labels(p)
            cap = cv2.VideoCapture(v['path'])
            f = 0
            while cap.grab():
                if f % STEP == 0 and lab[f] is not None:
                    _, img = cap.retrieve()
                    sx, sy = W / img.shape[1], H / img.shape[0]
                    row = yolo_row(p[:, :, f].T, lab[f], sx, sy)
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    dch = encode_depth(est(img))
                    for m, (im, dc, r) in enumerate(((img, dch, row),
                                                     (img[:, ::-1], dch[:, ::-1], mirror_row(row)))):
                        stem = f"{v['video']}_{f:05d}_{'m' if m else 'o'}"
                        cv2.imwrite(f'{OUT}/rgb/images/{split}/{stem}.jpg', np.ascontiguousarray(im),
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                        cv2.imwrite(f'{OUT}/rgbd/images/{split}/{stem}.tiff', np.dstack([im, dc]))
                        for kind in ('rgb', 'rgbd'):
                            open(f'{OUT}/{kind}/labels/{split}/{stem}.txt', 'w').write(fmt(r) + '\n')
                        open(f'{OUT}/rgbd/dist/{split}/{stem}.txt', 'w').write('nan\n')
                        key = f"{split}/{({v_: k for k, v_ in CLASSES.items()})[r[0]]}"
                        stats[key] = stats.get(key, 0) + 1
                f += 1
            print(split, v['video'], flush=True)
    for kind, ch in (('rgb', 3), ('rgbd', 4)):          # real only (PROTOCOL amendment)
        open(f'{OUT}/{kind}.yaml', 'w').write(
            f'path: {OUT}/{kind}\ntrain: images/train\nval: images/val\nchannels: {ch}\n'
            f'kpt_shape: [17, 3]\nflip_idx: {FLIP_IDX}\nnames:\n  0: LEFT\n  1: RIGHT\n  2: NEUTRAL\n')
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
