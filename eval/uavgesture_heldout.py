#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Held-out real-people test for T1R–T4R (train/PROTOCOL.md, "Training with real images").

    python3 uavgesture_heldout.py run --device 0 [--mirror]   -> data/results_uavgesture_heldout/frames[_mirror].csv
    python3 uavgesture_heldout.py report                      -> data/results_uavgesture_heldout/REPORT.md
    --full (run and report): stock + synthetic-trained T1-T4 on all 11 people, all videos
                              -> data/results_uavgesture_full/ (PROTOCOL.md, "full UAV-GESTURE dataset")

Test people S1, S3, S11, S14 (never used for training or choices), every 3rd frame, labels from
the authors' joints (uavgesture.frame_labels). Frames resized to 1280x720 as in training; RGB-D
models get the estimated metric depth (build_real.DepthEstimator), aligned to the image.
--mirror: frames mirrored and LEFT/RIGHT swapped (the only real LEFT test).
"""
import argparse
import os
import sys
import zlib

import numpy as np
import pandas as pd
import scipy.io as sio

import uavgesture as ug

EVAL = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(EVAL, '..', 'train')
sys.path.insert(0, TRAIN)
OUT = f'{EVAL}/data/results_uavgesture_heldout'
TEST = (1, 3, 11, 14)
STEP = 3
SWAP = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT', 'NEUTRAL': 'NEUTRAL', None: None}
MODELS = [('Alg. 1 RGB (stock)', None, 'rgb'),
          ('T1R (real-trained)', 'T1R', 'rgb'), ('T2R (real-trained)', 'T2R', 'rgbd'),
          ('T3R (real-trained)', 'T3R', 'rgbd'), ('T4R (real-trained)', 'T4R', 'rgbd'),
          ('T1 (synthetic-trained)', 'T1', 'rgb'), ('T2 (synthetic-trained)', 'T2', 'rgbd'),
          ('T3 (synthetic-trained)', 'T3', 'rgbd'), ('T4 (synthetic-trained)', 'T4', 'rgbd')]
SECTIONS = [('Real study: stock Alg. 1 vs models trained on real people only', 'real-trained'),
            ('Cross-domain: models trained on synthetic data only, on the same real people', 'synthetic-trained')]
HFOV = 1.204                      # any value: depth and image share the camera (no re-projection)


def use_full():
    """All people and videos, stock + synthetic-trained models only."""
    global TEST, OUT, MODELS, SECTIONS, REPORTS, TITLE
    TEST = (1, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15)
    OUT = f'{EVAL}/data/results_uavgesture_full'
    MODELS = [m for m in MODELS if m[1] is None or not m[1].endswith('R')]
    SECTIONS = [('Sim-to-real: models trained on synthetic data only, on all UAV-GESTURE people', 'synthetic-trained')]
    REPORTS = (f'{OUT}/REPORT.md', f'{TRAIN}/results/UAVGESTURE_FULL.md')
    TITLE = ['# Full UAV-GESTURE dataset (all 11 people, 119 videos): synthetic-trained models', '',
             'No split: none of these people was used to train or choose T1–T4 (synthetic data only). '
             "Every 3rd frame; labels from the authors' joint annotations; RGB-D models get estimated depth "
             '(Depth-Anything-V2 metric, not a depth sensor). See train/PROTOCOL.md, "synthetic-trained '
             'models on the full UAV-GESTURE dataset".']


def use_rules():
    """Algorithm 1 without vs with the gesture rules (both_arms_hover), RGB."""
    global MODELS, SECTIONS, REPORTS, TITLE, OUT
    use_full()
    OUT = f'{EVAL}/data/results_uavgesture_rules'
    MODELS = [('Alg. 1 RGB (stock)', None, 'rgb'), ('Alg. 1 RGB + gesture rules (stock)', 'rules', 'rgb')]
    SECTIONS = [('Algorithm 1 without vs with the gesture rules (both arms -> hover), all UAV-GESTURE people',
                 'gesture rules')]
    REPORTS = (f'{OUT}/REPORT.md', f'{TRAIN}/results/UAVGESTURE_RULES.md')
    TITLE = ['# Algorithm 1 with the gesture rules on the full UAV-GESTURE dataset (all 11 people, 119 videos)', '',
             'Stock yolo11n-pose, RGB, no training. Gesture rules (PROTOCOL.md): both arms '
             'stretched out -> hover; exactly one arm stretched out with the other arm seen -> that '
             "direction; anything else -> hover. Every 3rd frame; labels from the authors' joints."]


def use_depth():
    """Algorithm 1 RGB and depth mode, without and with the gesture rules, all people."""
    global MODELS, SECTIONS, REPORTS, TITLE, OUT
    use_full()
    OUT = f'{EVAL}/data/results_uavgesture_depth'
    MODELS = [('Alg. 1 RGB (stock)', None, 'rgb'), ('Alg. 1 RGB + gesture rules (stock)', 'rules', 'rgb'),
              ('Alg. 1 depth (stock)', 'depth', 'depth'),
              ('Alg. 1 depth + gesture rules (stock)', 'depth-rules', 'depth'),
              ('Alg. 1 depth, D435i noise (stock)', 'depth-d435i', 'depth'),
              ('Alg. 1 depth, D435i noise + gesture rules (stock)', 'depth-d435i-rules', 'depth')]
    SECTIONS = [('Algorithm 1, RGB and depth mode, without and with the gesture rules, all UAV-GESTURE people',
                 'Alg. 1')]
    REPORTS = (f'{OUT}/REPORT.md', f'{TRAIN}/results/UAVGESTURE_DEPTH.md')
    TITLE = ['# Algorithm 1 depth mode on the full UAV-GESTURE dataset (all 11 people, 119 videos)', '',
             'Stock yolo11n-pose, no training. Depth mode: estimated metric depth (Depth-Anything-V2, not a '
             'depth sensor) -> Algorithm 1 step 4 (clip 0.2-12 m, JET colour map) -> pose on the colour map; '
             '"D435i noise": the same estimate with the D435i stereo noise model (sigma ~ z^2, no return outside '
             '0.28-10 m) as in the synthetic study. '
             'Gesture rules (PROTOCOL.md): both arms stretched out -> hover; exactly one arm '
             "stretched out with the other arm seen -> that direction; anything else -> hover. Every 3rd frame; "
             "labels from the authors' joints."]


def use_uavview():
    """Algorithm 1 + gesture rules, without / with line 13b (sides as the UAV sees them)."""
    global MODELS, SECTIONS, REPORTS, TITLE, OUT
    use_full()
    OUT = f'{EVAL}/data/results_uavgesture_uavview'
    MODELS = [('Alg. 1 RGB + rules (stock)', 'rules', 'rgb'),
              ('Alg. 1 RGB + rules + 13b (stock)', 'rules-fix', 'rgb'),
              ('Alg. 1 depth + rules (stock)', 'depth-rules', 'depth'),
              ('Alg. 1 depth + rules + 13b (stock)', 'depth-rules-fix', 'depth'),
              ('Alg. 1 depth, D435i noise + rules (stock)', 'depth-d435i-rules', 'depth'),
              ('Alg. 1 depth, D435i noise + rules + 13b (stock)', 'depth-d435i-rules-fix', 'depth')]
    SECTIONS = [('Algorithm 1 with the gesture rules, without / with line 13b, all UAV-GESTURE people', 'Alg. 1')]
    REPORTS = (f'{OUT}/REPORT.md', f'{TRAIN}/results/UAVGESTURE_UAVVIEW.md')
    TITLE = ['# Algorithm 1 line 13b (sides as the UAV sees them) on the full UAV-GESTURE dataset', '',
             'Stock yolo11n-pose, no training, gesture rules on. Line 13b: if the left-shoulder keypoint lies '
             "on the image's left, left/right keypoints are swapped (pdd.face_camera). Depth = estimated "
             '(Depth-Anything-V2), optionally with the D435i noise model. Every 3rd frame; labels from the '
             "authors' joints. Comparisons are against the first row."]


def use_alg1trained():
    """Algorithm 1 + gesture rules + line 13b deciding the direction from the keypoints of the
    synthetic-trained T4 (YOLOv11-L_total), against the same Algorithm 1 on the stock model."""
    global MODELS, SECTIONS, REPORTS, TITLE, OUT
    use_full()
    OUT = f'{EVAL}/data/results_uavgesture_alg1trained'
    MODELS = [('Alg. 1 RGB + rules + 13b (stock)', 'rules-fix', 'rgb'),
              ('Alg. 1 depth + rules + 13b (stock)', 'depth-rules-fix', 'depth'),
              ('T4 classes (synthetic-trained)', 'T4', 'rgbd'),
              ('Alg. 1 + rules + 13b on T4 keypoints (synthetic-trained)', 'T4-alg1', 'rgbd')]
    SECTIONS = [('Algorithm 1 on stock vs synthetic-trained keypoints, all UAV-GESTURE people', 'synthetic-trained'),
                ('Algorithm 1 RGB vs depth mode (stock)', 'depth')]
    REPORTS = (f'{OUT}/REPORT.md', f'{TRAIN}/results/UAVGESTURE_ALG1_TRAINED.md')
    TITLE = ['# Algorithm 1 on the keypoints of the synthetic-trained model, full UAV-GESTURE dataset', '',
             'All 11 people, every 3rd frame, labels v1 from the authors\' joints. Algorithm 1 = gesture rules '
             '+ line 13b. "on T4 keypoints": T4 (YOLOv11-L_total, synthetic data only, RGB + estimated depth) '
             'gives the keypoints and Algorithm 1 decides; "T4 classes": T4 decides with its own classes. '
             'Estimated depth (Depth-Anything-V2). PROTOCOL.md, "Algorithm 1 on trained keypoints".']


def run(device, mirror):
    import cv2
    from ultralytics import YOLO

    from build_real import DepthEstimator, H, W
    from drone_control.pipeline import DEFAULT_WEIGHTS, GesturePipeline
    est = DepthEstimator(f'cuda:{device}') if any(m[2] in ('rgbd', 'depth') for m in MODELS) else None
    pipes = {}
    for name, stage, mode in MODELS:
        if stage is None:
            pipes[name] = (GesturePipeline.published(YOLO(DEFAULT_WEIGHTS), mode='rgb', device=device), mode)
        elif not stage.startswith('T'):               # Algorithm 1 variants: rules / fix / depth
            pipes[name] = (GesturePipeline.published(YOLO(DEFAULT_WEIGHTS), mode=mode, device=device,
                                                     both_arms_hover='rules' in stage,
                                                     face_camera='fix' in stage), mode)
        elif stage.endswith('-alg1'):                  # trained keypoints, Algorithm 1 decides
            m = YOLO(f'{TRAIN}/runs/{stage[:-5]}/weights/best.pt', task='pose')
            pipes[name] = (GesturePipeline.published(m, mode=mode, device=device, rgb_hfov=HFOV,
                                                     depth_hfov=HFOV, both_arms_hover=True,
                                                     face_camera=True), mode)
        elif os.path.exists(f'{TRAIN}/runs/{stage}/weights/best.pt'):
            m = YOLO(f'{TRAIN}/runs/{stage}/weights/best.pt', task='pose')
            pipes[name] = (GesturePipeline.trained(m, mode=mode, device=device, rgb_hfov=HFOV,
                                                   depth_hfov=HFOV), mode)
    rows = []
    vs = [v for v in ug.videos() if v['subject'] in TEST]
    for i, v in enumerate(vs):
        lab, _, _ = ug.frame_labels(sio.loadmat(v['mat'])['pos_img'])
        cap = cv2.VideoCapture(v['path'])
        f = 0
        while cap.grab():
            if f % STEP == 0:
                _, img = cap.retrieve()
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                depth = est(img) if est else np.zeros(img.shape[:2], np.float32)
                noisy = None
                if any(m[1] and 'd435i' in m[1] for m in MODELS):
                    from drone_control.sensor_model import d435i_depth_noise
                    seed = (zlib.crc32(v['video'].encode()) * 100003 + f) % 2 ** 32
                    noisy = d435i_depth_noise(depth.astype(np.float32), rng=np.random.default_rng(seed))
                label = lab[f]
                if mirror:
                    img, depth, label = np.ascontiguousarray(img[:, ::-1]), np.ascontiguousarray(depth[:, ::-1]), SWAP[label]
                    noisy = None if noisy is None else np.ascontiguousarray(noisy[:, ::-1])
                base = dict(video=v['video'], cls=v['cls'], subject=v['subject'], frame=f, label=label)
                for name, (p, mode) in pipes.items():
                    dep = noisy if 'D435i' in name else depth
                    _, people, raw, best = p.process(
                        None if mode == 'depth' else img,
                        dep.astype(np.float32) if mode in ('rgbd', 'depth') else None)
                    rows.append(dict(base, pipeline=name, pred=raw, detected=int(bool(people)),
                                     conf=best.conf if best else 0.0))
            f += 1
        print(f'{i + 1}/{len(vs)} {v["video"]}', flush=True)
    os.makedirs(OUT, exist_ok=True)
    out = f"{OUT}/frames{'_mirror' if mirror else ''}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'-> {out}')


def summary(d, group):
    from scipy.stats import binomtest
    d = d[(d.pipeline == MODELS[0][0]) | d.pipeline.str.contains(group, regex=False)]
    d = ug.filtered(d)
    lab = d[d.label.notna()].assign(ok=lambda x: x.pred == x.label)
    pct = lambda x: '–' if pd.isna(x) else f'{100 * x:.0f}%'          # noqa: E731
    order = [m[0] for m in MODELS if m[0] in set(d.pipeline)]
    L = ['| Model | Detection | Mean conf. | Accuracy | Acc. LEFT/RIGHT (n) | False cmds (N → L/R) | '
         'Swaps | Filtered: wrong L/R active | vs Alg. 1: B − A [95% CI, people], McNemar p |',
         '|---|---|---|---|---|---|---|---|---|']
    w = lab.pivot_table(index=['video', 'frame', 'subject'], columns='pipeline', values='ok')
    subj = w.index.get_level_values('subject').unique()
    rng = np.random.default_rng(0)
    draws = [rng.choice(subj, len(subj)) for _ in range(5000)]
    by = {s: w.xs(s, level='subject') for s in subj}
    for p in order:
        a, g = d[d.pipeline == p], lab[lab.pipeline == p]
        lr, ne = g[g.label != 'NEUTRAL'], g[g.label == 'NEUTRAL']
        swap = lr[lr.pred.isin(['LEFT', 'RIGHT']) & (lr.pred != lr.label)]
        wrong = g[g.cmd.isin(['LEFT', 'RIGHT']) & (g.cmd != g.label)]
        cmp_ = '–'
        if p != order[0] and order[0] == MODELS[0][0]:
            x = w[[order[0], p]].dropna()
            b_, c_ = int(((x[p] == 1) & (x[order[0]] == 0)).sum()), int(((x[p] == 0) & (x[order[0]] == 1)).sum())
            boot = []
            for dr in draws:
                gg = pd.concat([by[s] for s in dr])[[order[0], p]].dropna()
                boot.append(gg[p].mean() - gg[order[0]].mean())
            cmp_ = (f'{100 * (x[p].mean() - x[order[0]].mean()):+.0f} pp '
                    f'[{100 * np.percentile(boot, 2.5):+.0f}, {100 * np.percentile(boot, 97.5):+.0f}], '
                    f'{binomtest(min(b_, c_), b_ + c_).pvalue if b_ + c_ else 1.0:.2g}')
        L.append(f'| {p} | {pct(a.detected.mean())} | {a.conf.mean():.2f} | {pct(g.ok.mean())} | '
                 f'{pct(lr.ok.mean())} ({len(lr)}) | {pct(ne.pred.isin(["LEFT", "RIGHT"]).mean())} | '
                 f'{pct(len(swap) / max(len(lr), 1))} | {pct(len(wrong) / max(len(g), 1))} | {cmp_} |')
    counts = lab[lab.pipeline == order[0]].label.value_counts().to_dict()
    return L, counts


REPORTS = (f'{OUT}/REPORT.md', f'{TRAIN}/results_real/REPORT.md')
TITLE = ['# Held-out real people (UAV-GESTURE S1, S3, S11, S14)', '',
         'Test people never used for training or any choice; every 3rd frame; labels from the '
         "authors' joint annotations; RGB-D models get estimated depth (Depth-Anything-V2 metric, "
         'not a depth sensor). See train/PROTOCOL.md, "Training with real images".']


def report():
    L = list(TITLE)
    for suffix, title in (('', 'Original frames'), ('_mirror', 'Mirrored frames (LEFT ↔ RIGHT swapped)')):
        f = f'{OUT}/frames{suffix}.csv'
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        for sec, group in SECTIONS:
            if not d.pipeline.str.contains(group, regex=False).any():
                continue
            rows, counts = summary(d, group)
            L += ['', f'## {sec} — {title.lower()}', '', f'Labelled frames: {counts}', ''] + rows
    for out in REPORTS:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, 'w').write('\n'.join(L) + '\n')
        print(f'-> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('run', 'report'))
    ap.add_argument('--device', default='0')
    ap.add_argument('--mirror', action='store_true')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--rules', action='store_true', help='Algorithm 1 vs + gesture rules, all people')
    ap.add_argument('--depth', action='store_true', help='Algorithm 1 RGB + depth mode, +/- rules, all people')
    ap.add_argument('--uavview', action='store_true', help='Algorithm 1 + rules, without / with line 13b')
    ap.add_argument('--alg1trained', action='store_true', help='Algorithm 1 on synthetic-trained keypoints')
    a = ap.parse_args()
    if a.alg1trained:
        use_alg1trained()
    elif a.uavview:
        use_uavview()
    elif a.depth:
        use_depth()
    elif a.rules:
        use_rules()
    elif a.full:
        use_full()
    run(a.device, a.mirror) if a.step == 'run' else report()


if __name__ == '__main__':
    main()
