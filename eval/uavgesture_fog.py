#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Algorithm 1 (gesture rules + line 13b) with the stock yolo11n-pose on the full UAV-GESTURE
dataset, in clear air and in Koschmieder fog, RGB and depth mode. No training.

The fog is applied to each frame with the same model as the synthetic test set (sensor_model.fog_rgb,
fog_depth), using the depth estimated from the clear frame (Depth-Anything-V2); depth mode runs on
the fogged estimated depth. Particle fog cannot be applied to recorded videos.
    python3 uavgesture_fog.py run --device 0 [--mirror]   -> data/results_uavgesture_fog/frames[_mirror].csv
--trained: YOLOv11-L_total instead, RGB-D (fogged RGB + fogged estimated depth), its classes and
Algorithm 1 on its keypoints: T4 (synthetic-trained) on all 11 people, T4R (real-trained) on its
4 held-out people only            -> data/results_uavgesture_fog/trained[_mirror].csv
--fused LAMBDA: depth-weighted pose fusion, paper Eq. (1)-(2) (drone_control.fusion), RGB + estimated
depth, Algorithm 1 decides; its temporal history runs over the processed frames of each video,
separately per condition       -> data/results_uavgesture_fog/fused[_mirror].csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio

import uavgesture as ug
from uavgesture_heldout import EVAL, HFOV, STEP, SWAP, TEST, TRAIN

OUT = f'{EVAL}/data/results_uavgesture_fog'
SUBJECTS = (1, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15)
VIS = (50, 20, 10, 5)
MODES = ('rgb', 'depth')


def pipelines(device, trained, fused=None):
    """name -> (pipeline, mode, subjects); fused: one pipeline per condition (own history)."""
    from ultralytics import YOLO

    from drone_control.pipeline import DEFAULT_WEIGHTS, GesturePipeline
    if fused is not None:
        from drone_control.fusion import FusedPipeline
        m = YOLO(DEFAULT_WEIGHTS)
        return {f'fused-{c}': (FusedPipeline(m, lam=fused, device=device, rgb_hfov=HFOV, depth_hfov=HFOV),
                               'fused', SUBJECTS) for c in ('clear',) + VIS}
    if not trained:
        return {f'alg1-{m}': (GesturePipeline.published(YOLO(DEFAULT_WEIGHTS), mode=m, device=device,
                                                        both_arms_hover=True, face_camera=True),
                              m, SUBJECTS) for m in MODES}
    pipes = {}
    for stage, subjects in (('T4', SUBJECTS), ('T4R', TEST)):
        w = f'{TRAIN}/runs/{stage}/weights/best.pt'
        pipes[f'{stage}-classes'] = (GesturePipeline.trained(YOLO(w, task='pose'), mode='rgbd', device=device,
                                                             rgb_hfov=HFOV, depth_hfov=HFOV), 'rgbd', subjects)
        pipes[f'{stage}-alg1'] = (GesturePipeline.published(YOLO(w, task='pose'), mode='rgbd', device=device,
                                                            rgb_hfov=HFOV, depth_hfov=HFOV, both_arms_hover=True,
                                                            face_camera=True), 'rgbd', subjects)
    return pipes


def run(device, mirror, trained, fused=None):
    import cv2

    from build_real import DepthEstimator, H, W
    from drone_control import sensor_model as sm
    est = DepthEstimator(f'cuda:{device}')
    pipes = pipelines(device, trained, fused)
    rows = []
    vs = [v for v in ug.videos() if v['subject'] in SUBJECTS]
    for i, v in enumerate(vs):
        lab, _, _ = ug.frame_labels(sio.loadmat(v['mat'])['pos_img'])
        cap = cv2.VideoCapture(v['path'])
        for p, _, _ in pipes.values():
            p.reset()                                   # new video: no temporal history
        f = 0
        while cap.grab():
            if f % STEP == 0:
                _, img = cap.retrieve()
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                depth = est(img).astype(np.float32)
                label = lab[f]
                if mirror:
                    img, depth, label = (np.ascontiguousarray(img[:, ::-1]),
                                         np.ascontiguousarray(depth[:, ::-1]), SWAP[label])
                base = dict(video=v['video'], cls=v['cls'], subject=v['subject'], frame=f, label=label)
                for cond in ('clear',) + VIS:
                    if cond == 'clear':
                        rgb, dep = img, depth
                    else:
                        rgb = sm.fog_rgb(img, depth, HFOV, cond)
                        dep = sm.fog_depth(depth, HFOV, cond).astype(np.float32)
                    for name, (p, m, subjects) in pipes.items():
                        if v['subject'] not in subjects or (fused is not None and name != f'fused-{cond}'):
                            continue
                        _, people, raw, best = p.process(None if m == 'depth' else rgb,
                                                         None if m == 'rgb' else dep)
                        rows.append(dict(base, cond='clear' if cond == 'clear' else f'phys{cond}',
                                         pipeline=name, pred=raw, detected=int(bool(people)),
                                         conf=best.conf if best else 0.0))
            f += 1
        print(f'{i + 1}/{len(vs)} {v["video"]}', flush=True)
    os.makedirs(OUT, exist_ok=True)
    kind = 'fused' if fused is not None else 'trained' if trained else 'frames'
    out = f"{OUT}/{kind}{'_mirror' if mirror else ''}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'-> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('run',))
    ap.add_argument('--device', default='0')
    ap.add_argument('--mirror', action='store_true')
    ap.add_argument('--trained', action='store_true')
    ap.add_argument('--fused', type=float, default=None, metavar='LAMBDA')
    a = ap.parse_args()
    run(a.device, a.mirror, a.trained, a.fused)


if __name__ == '__main__':
    main()
