#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Run Unified-PDD (published and improved) on the captured frames -> data/results/frames.csv.

Conditions
  clear        clear worlds as rendered
  particle     *_fog worlds (Gazebo particle-emitter fog, as in RUN_SAR.md)
  phys<V>      clear-world frames + Koschmieder fog at visibility V m (drone_control.sensor_model),
               RGB from gt_depth, depth sensors lose returns beyond V
Sensors        oakd   OakD-Lite depth (640x480, 73 deg, as on x500_depth)
               d435i  RealSense D435i depth (848x480, 87 deg) + Intel z^2 noise model
Pipelines      published-{rgb,depth}  Algorithm 1 without the gesture rules (GesturePipeline.published)
               improved-{rgb,depth}   with the robustness additions (GesturePipeline defaults)
               fused<lambda>-fused    depth-weighted pose fusion, paper Eq. (1)-(2), with
                                      Algorithm 1 + gesture rules + line 13b (drone_control.fusion;
                                      --fused LAMBDA; --only-fused skips the other pipelines;
                                      every frame independent: temporal history reset)
               trained-<name>         a trained gesture model (GesturePipeline.trained,
                                      --trained name=weights.pt:rgb|rgbd; --only-trained
                                      skips the stock pipelines)
Each pipeline runs once per depth sensor (RGB pipelines use it for the distance and for picking
the nearest person).
"""
import argparse
import csv
import glob
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

from drone_control import sensor_model as sm
from drone_control.fusion import FusedPipeline
from drone_control.pipeline import DEFAULT_WEIGHTS
from drone_control.pipeline import GesturePipeline

import simtools as st

RGB_HFOV = 1.204
DEPTH = {'oakd': ('depth_oakd', 1.274), 'd435i': ('depth_d435i', sm.D435I_DEPTH_HFOV)}
GT_HFOV = 1.204
FIELDS = ['world', 'cond', 'sensor', 'pipeline', 'd', 'k', 'gesture', 'frame', 'truth', 'pred',
          'detected', 'n_people', 'conf', 'dist', 'proc_s']


def load(path):
    z = np.load(path)
    meta = json.loads(str(z['meta']))
    rgb = cv2.imdecode(z['rgb'], cv2.IMREAD_COLOR)
    return meta, rgb, {k: z[k].astype(np.float32) for k in ('depth_oakd', 'depth_d435i', 'gt_depth')}


def conditions(meta, rgb, depths, vis_list, seed):
    """Yield (cond, rgb, {sensor: depth}) for one captured frame."""
    rng = np.random.default_rng([seed, meta['k'], int(meta['d'] * 10), meta['frame'],
                                 st.GESTURES.index(meta['gesture'])])
    def sensors(dd):
        return {'oakd': dd['depth_oakd'],
                'd435i': sm.d435i_depth_noise(dd['depth_d435i'], DEPTH['d435i'][1], rng)}
    if meta['world'].endswith('_fog'):
        yield 'particle', rgb, sensors(depths)
        return
    yield 'clear', rgb, sensors(depths)
    z_rgb = sm.align_depth(depths['gt_depth'], GT_HFOV, (rgb.shape[1], rgb.shape[0]), RGB_HFOV)
    for v in vis_list:
        fogged = {k: sm.fog_depth(depths[k], DEPTH[s][1], v) for s, (k, _) in DEPTH.items()}
        yield f'phys{v:g}', sm.fog_rgb(rgb, z_rgb, RGB_HFOV, v), sensors(dict(depths, **fogged))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captures', default=f'{st.EVAL}/data/captures')
    ap.add_argument('--worlds', nargs='+', help='capture folders to evaluate (default: all)')
    ap.add_argument('--out', default=None,
                    help='CSV (default data/results/frames_<worlds>.csv)')
    ap.add_argument('--visibility', type=float, nargs='+', default=[50, 20, 10, 5])
    ap.add_argument('--device', default='0', help="'0' GPU (FP32) or 'cpu'")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='frames per world (0 = all), for tests')
    ap.add_argument('--trained', nargs='+', default=[], metavar='NAME=WEIGHTS:MODE')
    ap.add_argument('--only-trained', action='store_true')
    ap.add_argument('--trained-alg1', nargs='+', default=[], metavar='NAME=WEIGHTS:MODE',
                    help="a trained model's keypoints with Algorithm 1 + gesture rules + line 13b "
                         "deciding the direction (not its classes), as 'alg1-<name>'; PROTOCOL.md")
    ap.add_argument('--fused', type=float, nargs='+', default=[], metavar='LAMBDA',
                    help='depth-weighted fusion (Eq. (1)-(2)) with this lambda, as fused<lambda>-fused')
    ap.add_argument('--only-fused', action='store_true')
    ap.add_argument('--fix', action='store_true',
                    help="with --rules: also Algorithm 1 + gesture rules + line 13b (sides as the UAV sees "
                         "them, pdd.face_camera) as 'rulesfix-*'; PROTOCOL.md")
    ap.add_argument('--rules', action='store_true',
                    help="only Algorithm 1 without and with the gesture rules "
                         "(both_arms_hover, PROTOCOL.md), as 'published-*' / 'rules-*'")
    a = ap.parse_args()

    pipes = {}
    if a.fused:
        fmodel = YOLO(DEFAULT_WEIGHTS)
        for lam in a.fused:
            for s, (_, hfov) in DEPTH.items():
                pipes[(f'fused{lam:g}', 'fused', s)] = FusedPipeline(
                    fmodel, lam=lam, device=a.device, rgb_hfov=RGB_HFOV, depth_hfov=hfov)
    if not (a.only_trained or a.only_fused):
        model = YOLO(DEFAULT_WEIGHTS)
        for mode in ('rgb', 'depth'):
            for s, (_, hfov) in DEPTH.items():
                kw = dict(device=a.device, rgb_hfov=RGB_HFOV, depth_hfov=hfov)
                pipes[('published', mode, s)] = GesturePipeline.published(model, mode=mode, **kw)
                if a.rules:
                    pipes[('rules', mode, s)] = GesturePipeline.published(
                        model, mode=mode, both_arms_hover=True, **kw)
                    if a.fix:
                        pipes[('rulesfix', mode, s)] = GesturePipeline.published(
                            model, mode=mode, both_arms_hover=True, face_camera=True, **kw)
                else:
                    pipes[('improved', mode, s)] = GesturePipeline(model, mode=mode, **kw)
    for spec in a.trained:
        name, rest = spec.split('=', 1)
        weights, mode = rest.rsplit(':', 1)
        tm = YOLO(weights, task='pose')
        for s, (_, hfov) in DEPTH.items():
            pipes[(f'trained-{name}', mode, s)] = GesturePipeline.trained(
                tm, mode=mode, device=a.device, rgb_hfov=RGB_HFOV, depth_hfov=hfov)
    for spec in a.trained_alg1:
        name, rest = spec.split('=', 1)
        weights, mode = rest.rsplit(':', 1)
        tm = YOLO(weights, task='pose')
        for s, (_, hfov) in DEPTH.items():
            pipes[(f'alg1-{name}', mode, s)] = GesturePipeline.published(
                tm, mode=mode, device=a.device, rgb_hfov=RGB_HFOV, depth_hfov=hfov,
                both_arms_hover=True, face_camera=True)
    worlds = a.worlds or sorted(os.listdir(a.captures))
    a.out = a.out or f"{st.EVAL}/data/results/frames_{'+'.join(worlds)}.csv"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    files = []
    for w in worlds:
        fs = sorted(glob.glob(f'{a.captures}/{w}/*.npz'))
        files += fs[:a.limit] if a.limit else fs
    t_start = time.time()
    with open(a.out, 'w', newline='') as f:
        out = csv.DictWriter(f, FIELDS)
        out.writeheader()
        for i, path in enumerate(files):
            meta, rgb, depths = load(path)
            base = dict(world=meta['world'].replace('_fog', ''), d=meta['d'], k=meta['k'],
                        gesture=meta['gesture'], frame=meta['frame'],
                        truth=st.LABEL[meta['gesture']])
            for cond, img, dd in conditions(meta, rgb, depths, a.visibility, a.seed):
                for (variant, mode, s), pipe in pipes.items():   # the sensor also gives distance
                    t0 = time.perf_counter()
                    pipe.reset()                          # frames are independent samples
                    _, people, raw, best = pipe.process(
                        img if mode != 'depth' else None, dd[s])
                    out.writerow(dict(base, cond=cond, sensor=s,
                                      pipeline=f'{variant}-{mode}', pred=raw,
                                      detected=int(bool(people)), n_people=len(people),
                                      conf=best.conf if best else 0.0,
                                      dist=best.distance if best else float('nan'),
                                      proc_s=time.perf_counter() - t0))
            if i % 25 == 0:
                el = time.time() - t_start
                print(f'{i + 1}/{len(files)} frames  {el / 60:.1f} min  '
                      f'(~{el / (i + 1) * (len(files) - i - 1) / 60:.0f} min left)', flush=True)
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
