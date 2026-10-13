#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Choose lambda of Eq. (2) for the depth-weighted fusion (drone_control.fusion) on the
validation worlds val_0, val_1 of the training captures (never the test worlds or real people),
train/PROTOCOL.md "depth-weighted pose fusion".

Frames with a person (LEFT / RIGHT / NEUTRAL label); conditions as in the test set: clear and
Koschmieder V = 50, 20, 10, 5 m on the clear worlds, particle fog in the _fog worlds; OakD depth.
Each frame is independent (temporal history reset). YOLO runs once per frame and condition; the
fusion is repeated for every lambda on the same detections.

    python3 tune_fusion.py --device 0   -> data/fusion_tuning/val.csv, REPORT.md
"""
import argparse
import glob
import hashlib
import os

import numpy as np
import pandas as pd
from ultralytics import YOLO

from drone_control.fusion import FusedPipeline
from drone_control.pipeline import DEFAULT_WEIGHTS, GesturePipeline

import evaluate as ev
import simtools as st

LAMBDAS = (0, 10, 30, 100, 300, 1000, 3000)
OUT = f'{st.EVAL}/data/fusion_tuning'
CAPTURES = os.path.join(st.EVAL, '..', 'train', 'data', 'captures')


class Memo:
    """Shared YOLO results for identical input images (same frame and condition)."""

    def __init__(self, pipe):
        self.pipe, self.cache = pipe, {}

    def __call__(self, img):
        key = (img.shape, hashlib.blake2b(np.ascontiguousarray(img).data, digest_size=16).digest())
        if key not in self.cache:
            self.cache[key] = GesturePipeline.detect(self.pipe, img)
        return self.cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='0')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    model = YOLO(DEFAULT_WEIGHTS)
    hfov = ev.DEPTH['oakd'][1]
    kw = dict(device=a.device, rgb_hfov=ev.RGB_HFOV, depth_hfov=hfov)
    fused = {lam: FusedPipeline(model, lam=lam, **kw) for lam in LAMBDAS}
    memo = Memo(fused[LAMBDAS[0]])
    for p in fused.values():
        p.detect = memo
    alg1 = {m: GesturePipeline.published(model, mode=m, both_arms_hover=True, face_camera=True, **kw)
            for m in ('rgb', 'depth')}
    files = sorted(glob.glob(f'{CAPTURES}/val_*/*.npz'))
    files = files[:a.limit] if a.limit else files
    rows = []
    for i, path in enumerate(files):
        meta, rgb, depths = ev.load(path)
        if meta['label'] not in ('LEFT', 'RIGHT', 'NEUTRAL'):
            continue
        meta = dict(meta, k=meta['i'], gesture=meta['label'].lower(), frame=0)
        for cond, img, dd in ev.conditions(meta, rgb, depths, (50, 20, 10, 5), 0):
            memo.cache.clear()
            base = dict(world=meta['world'], i=meta['i'], d=meta['d'], cond=cond, truth=meta['label'])
            for m, p in alg1.items():
                raw = p.process(img if m == 'rgb' else None, dd['oakd'])[2]
                rows.append(dict(base, pipeline=f'alg1-{m}', pred=raw))
            for lam, p in fused.items():
                p.reset()
                raw = p.process(img, dd['oakd'])[2]
                rows.append(dict(base, pipeline=f'fused-{lam:g}', pred=raw))
        if i % 50 == 0:
            print(f'{i + 1}/{len(files)}', flush=True)
    os.makedirs(OUT, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/val.csv', index=False)
    d['ok'] = d.pred == d.truth
    acc = d.pivot_table(index='pipeline', columns='cond', values='ok', aggfunc='mean')
    acc['all'] = d.groupby('pipeline').ok.mean()
    fused_acc = acc.loc[[f'fused-{lam:g}' for lam in LAMBDAS], 'all']
    best = fused_acc.idxmax()
    lines = ['# lambda of Eq. (2) on the validation worlds (val_0, val_1; OakD)', '',
             f'{d.groupby("pipeline").size().iloc[0]} frame-conditions per pipeline. Direction accuracy:', '',
             '```', (acc * 100).round(1).to_string(), '```', '',
             f'Chosen: **{best}** (highest overall accuracy among the fused pipelines; ties -> smaller lambda).']
    open(f'{OUT}/REPORT.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
