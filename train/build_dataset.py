#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Training captures -> YOLO pose datasets (labels from the generator, see PROTOCOL.md).

    python3 build_dataset.py              -> data/yolo/{rgb,rgbd}/{images,labels}/{train,val}
                                             data/yolo/{rgb,rgbd}.yaml

Classes 0 LEFT, 1 RIGHT, 2 NEUTRAL; 17 COCO keypoints, visibility 2 = visible, 1 = in the image
but hidden (ground-truth depth shows something more than 0.15 m in front of it), 0 = outside.
Box = projected mesh, clipped to the image (truncated people are labelled, as in COCO); kept if
>= 10 px high and at least one keypoint is visible. Images are stored at 1280x720.
  rgb   JPEG; clear-world frames get Koschmieder fog (random visibility 4-60 m) with p = 0.35
  rgbd  4-channel TIFF: the same BGR + depth aligned to the RGB image (OakD-Lite, or D435i with
        its noise model, 50/50), linear 0.1-20 m -> 0-255, no return = 255
"""
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'eval'))
from drone_control import sensor_model as sm  # noqa: E402

RGB_HFOV, GT_HFOV = 1.204, 1.204
DEPTH = {'oakd': ('depth_oakd', 1.274), 'd435i': ('depth_d435i', sm.D435I_DEPTH_HFOV)}
W, H = 1280, 720
NEAR, FAR = 0.1, 20.0
CLASSES = {'LEFT': 0, 'RIGHT': 1, 'NEUTRAL': 2}
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
OUT = f'{HERE}/data/yolo'


def load_obj(path):
    return np.array([list(map(float, l.split()[1:4])) for l in open(path) if l.startswith('v ')])


def to_world(p, hx, hy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.c_[c * p[:, 0] - s * p[:, 1] + hx, s * p[:, 0] + c * p[:, 1] + hy, p[:, 2]]


def project(pw, cam, w, h, hfov):
    """World points -> pixel (u, v) and depth along the optical axis (Gazebo camera: +x forward,
    image right = -y, image down = -z)."""
    cx, cy, cz, yaw, pitch = cam
    cyw, syw, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    R = np.array([[cyw, -syw, 0], [syw, cyw, 0], [0, 0, 1]]) @ \
        np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    pc = (pw - [cx, cy, cz]) @ R                                 # = R^T (p - c)
    f = (w / 2) / math.tan(hfov / 2)
    z = pc[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        u, v = w / 2 - f * pc[:, 1] / z, h / 2 - f * pc[:, 2] / z
    return u, v, z


def label(meta, human, mesh, gt):
    hx, hy, hyaw = meta['human_pose']
    u, v, z = project(to_world(mesh, hx, hy, hyaw), meta['cam'], W, H, RGB_HFOV)
    front = z > 0.05
    if front.sum() < 10:
        return None
    x1, x2 = np.clip([u[front].min(), u[front].max()], 0, W)
    y1, y2 = np.clip([v[front].min(), v[front].max()], 0, H)
    if y2 - y1 < 10 or x2 - x1 < 4:
        return None
    ku, kv, kz = project(to_world(np.array(human['kp']), hx, hy, hyaw), meta['cam'], W, H,
                         RGB_HFOV)
    gh, gw = gt.shape
    kps = []
    for a, b, c in zip(ku, kv, kz):
        if not (c > 0.05 and 0 <= a < W and 0 <= b < H):
            kps.append((0.0, 0.0, 0))
            continue
        g = float(gt[min(int(b * gh / H), gh - 1), min(int(a * gw / W), gw - 1)])
        kps.append((a / W, b / H, 2 if not g < c - 0.15 else 1))
    if not any(k[2] == 2 for k in kps):
        return None
    row = [CLASSES[meta['label']], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W,
           (y2 - y1) / H] + [x for k in kps for x in k]
    return ' '.join(f'{x:.6f}' if isinstance(x, float) else str(x) for x in row)


def depth_channel(depths, rng):
    sensor = 'oakd' if rng.random() < 0.5 else 'd435i'
    key, hfov = DEPTH[sensor]
    d = depths[key]
    if sensor == 'd435i':
        d = sm.d435i_depth_noise(d, hfov, rng)
    a = sm.align_depth(d, hfov, (W, H), RGB_HFOV)
    a = np.nan_to_num(np.asarray(a, np.float32), nan=FAR, posinf=FAR, neginf=FAR)
    return (np.clip((a - NEAR) / (FAR - NEAR), 0, 1) * 255).astype(np.uint8), sensor


def main():
    humans = {h['name']: h for h in json.load(open(f'{HERE}/models/humans.json'))}
    meshes = {}
    stats = {}
    for kind in ('rgb', 'rgbd'):
        for split in ('train', 'val'):
            for sub in ('images', 'labels'):
                os.makedirs(f'{OUT}/{kind}/{sub}/{split}', exist_ok=True)
    for f in sorted(glob.glob(f'{HERE}/data/captures/*/*.npz')):
        z = np.load(f)
        meta = json.loads(str(z['meta']))
        split = 'val' if meta['world'].startswith('val') else 'train'
        stem = f"{meta['world']}_{meta['i']:05d}"
        rng = np.random.default_rng([sum(map(ord, meta['world'])), meta['i']])
        depths = {k: z[k].astype(np.float32) for k in ('depth_oakd', 'depth_d435i', 'gt_depth')}
        rgb = cv2.imdecode(z['rgb'], cv2.IMREAD_COLOR)
        row = None
        if meta['human']:
            h = humans[meta['human']]
            if h['name'] not in meshes:
                meshes[h['name']] = load_obj(f"{HERE}/models/{h['name']}/meshes/human.obj")
            row = label(meta, h, meshes[h['name']], depths['gt_depth'])
        fog = None
        if not meta['world'].endswith('_fog') and rng.random() < 0.35:
            fog = float(np.exp(rng.uniform(np.log(4), np.log(60))))
            z_rgb = sm.align_depth(depths['gt_depth'], GT_HFOV, (rgb.shape[1], rgb.shape[0]),
                                   RGB_HFOV)
            rgb = sm.fog_rgb(rgb, z_rgb, RGB_HFOV, fog)
            for s, (k, hf) in DEPTH.items():
                depths[k] = sm.fog_depth(depths[k], hf, fog)
        img = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        dch, sensor = depth_channel(depths, rng)
        cv2.imwrite(f'{OUT}/rgb/images/{split}/{stem}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(f'{OUT}/rgbd/images/{split}/{stem}.tiff', np.dstack([img, dch]))
        for kind in ('rgb', 'rgbd'):
            open(f'{OUT}/{kind}/labels/{split}/{stem}.txt', 'w').write(row + '\n' if row else '')
        key = (split, meta['label'] if row else 'none')
        stats[key] = stats.get(key, 0) + 1
    for kind, ch in (('rgb', 3), ('rgbd', 4)):
        open(f'{OUT}/{kind}.yaml', 'w').write(
            f'path: {OUT}/{kind}\ntrain: images/train\nval: images/val\nchannels: {ch}\n'
            f'kpt_shape: [17, 3]\nflip_idx: {FLIP_IDX}\n'
            'names:\n  0: LEFT\n  1: RIGHT\n  2: NEUTRAL\n')
    print(json.dumps({f'{s}/{l}': n for (s, l), n in sorted(stats.items())}, indent=1))


if __name__ == '__main__':
    main()
