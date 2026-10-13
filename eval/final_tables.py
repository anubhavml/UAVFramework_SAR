#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Extended-evaluation tables (paper Section 7) from the logged per-frame results.

Synthetic test set (captures_test, OakD): Algorithm 1 (gesture rules + line 13b) on RGB and on
depth, Algorithm 1 on the depth-weighted fused pose (Eq. (1)-(2)), YOLOv11-L_total (T4) classes.
Real people (UAV-GESTURE, every 3rd frame, labelled frames only; original and mirrored): the same
on all 11 people, and on the 4 held-out people with YOLOv11-L_total (real, T4R).

    python3 final_tables.py [--fused-lambda L]   -> <workspace>/train/results/FINAL_TABLES.md
"""
import argparse
import os

import pandas as pd

EVAL = os.path.dirname(os.path.abspath(__file__))
D = f'{EVAL}/data'
OUT = os.path.join(EVAL, '..', 'train', 'results', 'FINAL_TABLES.md')
CONDS = ['clear', 'particle', 'phys50', 'phys20', 'phys10', 'phys5']
HELDOUT = (1, 3, 11, 14)
LR = ('LEFT', 'RIGHT')


def metrics(d, truth):
    t, p = d[truth], d.pred
    neutral = t == 'NEUTRAL'
    lr = t.isin(LR)
    return dict(n=len(d), det=d.detected.mean() * 100, conf=d.conf.mean(),
                acc=(p == t).mean() * 100, lr=(p[lr] == t[lr]).mean() * 100 if lr.any() else float('nan'),
                fc=p[neutral].isin(LR).mean() * 100 if neutral.any() else float('nan'))


def table(rows, conds, truth, keys=('acc', 'lr', 'fc', 'det', 'conf')):
    """rows: [(name, frame)] -> markdown, one block per metric."""
    out = []
    for k in keys:
        out += [f'**{k}**', '', '| pipeline | ' + ' | '.join(conds) + ' |', '|---' * (len(conds) + 1) + '|']
        for name, d in rows:
            vals = []
            for c in conds:
                q = d[d.cond == c]
                if not len(q):
                    vals.append('–')
                    continue
                v = metrics(q, truth)[k]
                vals.append(f'{v:.2f}' if k == 'conf' else f'{v:.1f}')
            out.append(f'| {name} | ' + ' | '.join(vals) + ' |')
        out.append('')
    ns = {name: {c: int((d.cond == c).sum()) for c in conds} for name, d in rows}
    out.append(f'frames per condition: {ns[rows[0][0]]}')
    return out


def synthetic(lam):
    uv = pd.read_csv(f'{D}/results_test_uavview/frames.csv')
    uv = uv[uv.sensor == 'oakd']
    t4 = pd.read_csv(os.path.join(EVAL, '..', 'train', 'results', 'frames_T4.csv'))
    t4 = t4[t4.sensor == 'oakd']
    rows = [('Alg. 1, RGB', uv[uv.pipeline == 'rulesfix-rgb']),
            ('Alg. 1, depth', uv[uv.pipeline == 'rulesfix-depth'])]
    f = f'{D}/results_test_fused/frames.csv'
    if os.path.exists(f):
        fu = pd.read_csv(f)
        fu = fu[(fu.sensor == 'oakd') & (fu.pipeline == f'fused{lam:g}-fused')]
        rows.append((f'Alg. 1, fused pose Eq. (1)-(2), lambda {lam:g}', fu))
    rows.append(('YOLOv11-L_total (synthetic-trained), classes', t4[t4.pipeline == 'trained-T4-rgbd']))
    lines = ['## Synthetic test set (City, Forest, Wall; OakD)', ''] + table(rows, CONDS, 'truth')
    times = {name: d.proc_s.median() for name, d in rows}
    lines += ['', 'median processing time per frame (s): ' +
              ', '.join(f'{k}: {v * 1000:.0f} ms' for k, v in times.items()), '']
    return lines


def real(lam):
    fr = {}
    for mirror in ('', '_mirror'):
        parts = [pd.read_csv(f'{D}/results_uavgesture_fog/frames{mirror}.csv')]
        for kind in ('fused', 'trained'):
            f = f'{D}/results_uavgesture_fog/{kind}{mirror}.csv'
            if os.path.exists(f):
                parts.append(pd.read_csv(f))
        d = pd.concat(parts, ignore_index=True)
        fr[mirror] = d[d.label.notna()]
    names = {'alg1-rgb': 'Alg. 1, RGB', 'alg1-depth': 'Alg. 1, depth (est.)',
             'fused': f'Alg. 1, fused pose Eq. (1)-(2), lambda {lam:g}',
             'T4-classes': 'YOLOv11-L_total (synthetic-trained), classes',
             'T4-alg1': 'Alg. 1 on YOLOv11-L_total (synthetic-trained) keypoints',
             'T4R-classes': 'YOLOv11-L_total (real-trained), classes',
             'T4R-alg1': 'Alg. 1 on YOLOv11-L_total (real-trained) keypoints'}
    conds = [c for c in CONDS if c != 'particle']
    lines = []
    for title, subj, keys in (('all 11 people', None, ('alg1-rgb', 'alg1-depth', 'fused', 'T4-classes', 'T4-alg1')),
                              ('4 held-out people (S1, S3, S11, S14)', HELDOUT,
                               ('alg1-rgb', 'alg1-depth', 'fused', 'T4-classes', 'T4R-classes', 'T4R-alg1'))):
        for mirror in ('', '_mirror'):
            d = fr[mirror]
            if subj:
                d = d[d.subject.isin(subj)]
            rows = []
            for k in keys:
                q = d[d.pipeline.str.startswith('fused')] if k == 'fused' else d[d.pipeline == k]
                if len(q):
                    rows.append((names[k], q))
            lines += [f'## Real people, {title}{", mirrored" if mirror else ""}', ''] + \
                table(rows, conds, 'label') + ['']
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fused-lambda', type=float, default=3000.0)
    a = ap.parse_args()
    lines = ['# Extended evaluation: final tables (no retraining unless stated)', '',
             'acc = direction accuracy; lr = accuracy on LEFT/RIGHT frames; fc = false commands '
             '(NEUTRAL read as LEFT/RIGHT); det = person detected; conf = mean detection confidence '
             '(0 if none). Algorithm 1 = with the gesture rules and line 13b, stock yolo11n-pose.', '']
    lines += synthetic(a.fused_lambda) + real(a.fused_lambda)
    open(OUT, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
