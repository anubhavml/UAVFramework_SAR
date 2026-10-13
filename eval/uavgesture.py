#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Real-data test on UAV-GESTURE (train/PROTOCOL.md, "Real-data test: UAV-GESTURE").

    python3 uavgesture.py labels            # label distribution only (no model)
    python3 uavgesture.py run --device 0    # run the pipelines -> data/results_uavgesture/frames.csv
    python3 uavgesture.py report            # -> data/results_uavgesture/REPORT.md

Per frame (every STEP-th frame of each video) the label comes from the authors' joint
annotations (OpenPose COCO-18) with the protocol's gesture definition in the image plane; the
pipelines see only the video frame (RGB, no depth).
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import scipy.io as sio

EVAL = os.path.dirname(os.path.abspath(__file__))
from demo_common import DATA, videos  # noqa: E402,F401  (dataset location, video list)
OUT = f'{EVAL}/data/results_uavgesture'
STEP = 3
NECK, RSH, REL, RWR, LSH, LEL, LWR, RHIP, LHIP = 1, 2, 3, 4, 5, 6, 7, 8, 11
PIPES = ('published-rgb', 'trained-T1-rgb')
NAMES = {'published-rgb': 'Alg. 1 RGB (stock)', 'trained-T1-rgb': 'T1 (trained, RGB)'}




def angle(a, b):
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def frame_labels(p):
    """p: (2, 18, n) joint pixels, (0, 0) = missing -> per frame label (or None) and angles."""
    j = p.transpose(2, 1, 0)                                         # (n, 18, 2)
    need = [NECK, RSH, REL, RWR, LSH, LEL, LWR, RHIP, LHIP]
    ok = (np.abs(j[:, need]).sum(-1) > 0).all(1)
    down = (j[:, RHIP] + j[:, LHIP]) / 2 - j[:, NECK]
    out = []
    for side, (s, e, w) in (('L', (LSH, LEL, LWR)), ('R', (RSH, REL, RWR))):
        upper, fore = j[:, e] - j[:, s], j[:, w] - j[:, e]
        out.append((angle(upper, down), angle(upper, fore)))
    (la, lb), (ra, rb) = out
    raised = lambda a, b: (a >= 60) & (a <= 110) & (b <= 40)          # noqa: E731
    lab = np.full(len(j), None, object)
    lab[raised(la, lb) & (ra < 35)] = 'LEFT'
    lab[raised(ra, rb) & (la < 35)] = 'RIGHT'
    lab[((la < 35) & (ra < 35)) | ((la >= 60) & (ra >= 60))] = 'NEUTRAL'
    lab[~ok] = None
    return lab, la, ra


def frame_labels_v2(p):
    """Labels v2 (PROTOCOL.md; gesture rule 4): as frame_labels,
    but one arm raised + the other arm up (>= 60 deg, not raised) -> the raised arm's direction."""
    lab, la, ra = frame_labels(p)
    j = p.transpose(2, 1, 0)
    bend = {}
    for side, (s, e, w) in (('L', (LSH, LEL, LWR)), ('R', (RSH, REL, RWR))):
        bend[side] = angle(j[:, e] - j[:, s], j[:, w] - j[:, e])
    raised_l = (la >= 60) & (la <= 110) & (bend['L'] <= 40)
    raised_r = (ra >= 60) & (ra <= 110) & (bend['R'] <= 40)
    lab = lab.copy()
    known = np.array([x is not None for x in lab])
    lab[known & raised_l & ~raised_r & (ra >= 60)] = 'LEFT'
    lab[known & raised_r & ~raised_l & (la >= 60)] = 'RIGHT'
    return lab, la, ra


def labels_table():
    rows = []
    for v in videos():
        lab, la, ra = frame_labels(sio.loadmat(v['mat'])['pos_img'])
        for f in range(0, len(lab), STEP):
            rows.append(dict(video=v['video'], cls=v['cls'], subject=v['subject'], frame=f,
                             label=lab[f], abd_left=la[f], abd_right=ra[f]))
    return pd.DataFrame(rows)


def run(device):
    import cv2
    from ultralytics import YOLO

    from drone_control.pipeline import DEFAULT_WEIGHTS, GesturePipeline
    pipes = {'published-rgb': GesturePipeline.published(YOLO(DEFAULT_WEIGHTS), mode='rgb', device=device),
             'trained-T1-rgb': GesturePipeline.trained(
                 YOLO(f'{EVAL}/../train/runs/T1/weights/best.pt', task='pose'), mode='rgb', device=device)}
    lab = labels_table().set_index(['video', 'frame'])
    rows = []
    vs = videos()
    for i, v in enumerate(vs):
        cap = cv2.VideoCapture(v['path'])
        f = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if f % STEP == 0:
                _, img = cap.retrieve()
                base = dict(video=v['video'], cls=v['cls'], subject=v['subject'], frame=f,
                            label=lab.loc[(v['video'], f), 'label'])
                for name, p in pipes.items():
                    _, people, raw, best = p.process(img, None)
                    rows.append(dict(base, pipeline=name, pred=raw, detected=int(bool(people)),
                                     n_people=len(people), conf=best.conf if best else 0.0))
            f += 1
        print(f'{i + 1}/{len(vs)} {v["video"]}', flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/frames.csv', index=False)
    print(f'-> {OUT}/frames.csv')


def filtered(d):
    """5-of-7 vote filter (the gesture node's) over each video's frame sequence."""
    from drone_control.pdd import TemporalFilter
    out = []
    for (_, _), g in d.sort_values('frame').groupby(['pipeline', 'video'], sort=False):
        tf = TemporalFilter(7, 5)
        out.append(g.assign(cmd=[tf.update(r) for r in g.pred]))
    return pd.concat(out)


def report():
    from scipy.stats import binomtest
    d = filtered(pd.read_csv(f'{OUT}/frames.csv'))
    lab = d[d.label.notna()].copy()
    lab['ok'] = lab.pred == lab.label
    pct = lambda x: '–' if pd.isna(x) else f'{100 * x:.0f}%'          # noqa: E731
    n_frames = d.groupby(['video', 'frame']).ngroups
    L = ['# Real-data test: UAV-GESTURE', '',
         f'{d.video.nunique()} videos, {d.subject.nunique()} people, {n_frames} frames (every {STEP}rd); '
         f'{lab.groupby(["video", "frame"]).ngroups} frames labelled from the annotated joints '
         f'(protocol definition): ' + ', '.join(f'{k} {v}' for k, v in
                                                lab[lab.pipeline == PIPES[0]].label.value_counts().items()) +
         '. RGB only (no depth). See train/PROTOCOL.md and uavgesture.py.', '',
         '## 1. Overview', '',
         '| Pipeline | Detection (all frames) | Mean confidence | Accuracy (labelled) | Acc. LEFT/RIGHT frames | '
         'False commands (NEUTRAL → L/R) | Swaps L↔R | Filtered: wrong L/R command active |',
         '|---|---|---|---|---|---|---|---|']
    for p in PIPES:
        a, g = d[d.pipeline == p], lab[lab.pipeline == p]
        lr, ne = g[g.label != 'NEUTRAL'], g[g.label == 'NEUTRAL']
        swap = lr[lr.pred.isin(['LEFT', 'RIGHT']) & (lr.pred != lr.label)]
        wrong_cmd = g[g.cmd.isin(['LEFT', 'RIGHT']) & (g.cmd != g.label)]
        L.append(f'| {NAMES[p]} | {pct(a.detected.mean())} | {a.conf.mean():.2f} | {pct(g.ok.mean())} | '
                 f'{pct(lr.ok.mean()) if len(lr) else "–"} (n={len(lr)}) | '
                 f'{pct(ne.pred.isin(["LEFT", "RIGHT"]).mean())} | {pct(len(swap) / max(len(lr), 1))} | '
                 f'{pct(len(wrong_cmd) / max(len(g), 1))} |')
    # paired: Alg. 1 vs T1 on labelled frames
    w = lab.pivot_table(index=['video', 'frame', 'subject'], columns='pipeline', values='ok').dropna()
    b = int(((w[PIPES[1]] == 1) & (w[PIPES[0]] == 0)).sum())
    c = int(((w[PIPES[1]] == 0) & (w[PIPES[0]] == 1)).sum())
    p_mc = binomtest(min(b, c), b + c).pvalue if b + c else 1.0
    rng = np.random.default_rng(0)
    subj = w.index.get_level_values('subject').unique()
    by = {s: w.xs(s, level='subject') for s in subj}
    boot = []
    for _ in range(10000):
        g = pd.concat([by[s] for s in rng.choice(subj, len(subj))])
        boot.append(g[PIPES[1]].mean() - g[PIPES[0]].mean())
    diff = w[PIPES[1]].mean() - w[PIPES[0]].mean()
    L += ['', '## 2. Paired comparison (same labelled frames)', '',
          '| A → B | n | Acc. A | Acc. B | B − A | 95% CI (people as clusters) | B better / A better | McNemar p |',
          '|---|---|---|---|---|---|---|---|',
          f'| {NAMES[PIPES[0]]} → {NAMES[PIPES[1]]} | {len(w)} | {pct(w[PIPES[0]].mean())} | {pct(w[PIPES[1]].mean())} | '
          f'{100 * diff:+.0f} pp | [{100 * np.percentile(boot, 2.5):+.0f}, {100 * np.percentile(boot, 97.5):+.0f}] | '
          f'{b} / {c} | {p_mc:.2g} |']
    L += ['', '## 3. By UAV-GESTURE class (labelled frames: accuracy; all frames: share of frames read as LEFT/RIGHT)', '',
          '| Class | Labelled frames (L / R / N) | ' + ' | '.join(f'{NAMES[p]} acc.' for p in PIPES) + ' | ' +
          ' | '.join(f'{NAMES[p]} L/R read' for p in PIPES) + ' |', '|---|---|' + '---|' * (2 * len(PIPES))]
    for cls, g in d.groupby('cls'):
        gl = lab[(lab.cls == cls) & (lab.pipeline == PIPES[0])].label.value_counts()
        cnt = f"{gl.get('LEFT', 0)} / {gl.get('RIGHT', 0)} / {gl.get('NEUTRAL', 0)}"
        acc = [pct(lab[(lab.cls == cls) & (lab.pipeline == p)].ok.mean()) for p in PIPES]
        lrr = [pct(g[g.pipeline == p].pred.isin(['LEFT', 'RIGHT']).mean()) for p in PIPES]
        L.append(f'| {cls} | {cnt} | ' + ' | '.join(acc + lrr) + ' |')
    L += ['', '## 4. Confusion (labelled frames, rows = label from the joints)', '']
    for p in PIPES:
        cm = pd.crosstab(lab[lab.pipeline == p].label, lab[lab.pipeline == p].pred)
        L += [f'**{NAMES[p]}**', '', '| label \\ pred | ' + ' | '.join(cm.columns) + ' |',
              '|---|' + '---|' * len(cm.columns)]
        L += [f'| {i} | ' + ' | '.join(str(v) for v in r) + ' |' for i, r in cm.iterrows()] + ['']
    open(f'{OUT}/REPORT.md', 'w').write('\n'.join(L) + '\n')
    print(f'-> {OUT}/REPORT.md')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('labels', 'run', 'report'))
    ap.add_argument('--device', default='0')
    a = ap.parse_args()
    if a.step == 'labels':
        t = labels_table()
        print(len(t), 'frames;', t.label.value_counts(dropna=False).to_dict())
        print(t.groupby('cls').label.value_counts(dropna=False).unstack(fill_value=0))
    elif a.step == 'run':
        run(a.device)
    else:
        report()


if __name__ == '__main__':
    main()
