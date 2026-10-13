#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Procedurally generated training / validation worlds (never the paper's Forest, City, Wall).

    python3 make_train_worlds.py            -> worlds/train_<i>{,_fog}.sdf, worlds/val_<i>{,_fog}.sdf

Each world: random ground and sky colour and sun direction; random mix of the paper worlds'
asset types (oak and pine trees, school, collapsed building, playground, coloured walls) laid out
around a free area at the origin where the person stands; the training humans of its split
(parked below the ground) and the evaluation camera rig. Fog variants add the particle-emitter
fog of sar_sim with a random density. Obstacles go to worlds/<name>.json for placement checks.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'sar_sim', 'scripts'))
import make_worlds as mw  # noqa: E402

OUT = os.path.join(HERE, 'worlds')
PARK_Z = -30.0
# Fuel asset footprints (half extents, m) for placement checks
BUILDINGS = {'school': (16.5, 8.8), 'collapsed_industrial': (11.5, 11.5), 'playground': (5.8, 3.4)}


def grey(rng, lo, hi, tint=0.08):
    g = rng.uniform(lo, hi)
    return ' '.join(f'{np.clip(g + rng.uniform(-tint, tint), 0, 1):.3f}' for _ in range(3))


def world(name, rng, humans, fogged):
    body = mw.ground(grey(rng, 0.3, 0.92), size=200)
    obstacles = []
    for i in range(rng.integers(0, 22)):                        # trees, some close to the person
        r, a = rng.uniform(1.8, 28), rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a), r * math.sin(a)
        kind = 'oak_tree' if rng.random() < 0.55 else 'pine_tree'
        body += mw.include(kind, f'tree_{i}', x, y, round(rng.uniform(0, 6.28), 2))
        obstacles.append(dict(kind='tree', x=x, y=y, r=1.0))
    for i in range(rng.integers(0, 3)):                         # buildings, outside the stage
        kind = list(BUILDINGS)[rng.integers(0, 3)]
        r, a, yaw = rng.uniform(28, 42), rng.uniform(0, 2 * math.pi), rng.uniform(0, 6.28)
        x, y = r * math.cos(a), r * math.sin(a)
        body += mw.include(kind, f'building_{i}', x, y, round(yaw, 2))
        hx, hy = BUILDINGS[kind]
        obstacles.append(dict(kind='box', x=x, y=y, hx=hx, hy=hy, yaw=yaw))
    for i in range(rng.integers(0, 6)):                         # walls in random colours
        r, a, yaw = rng.uniform(6, 26), rng.uniform(0, 2 * math.pi), rng.uniform(0, 3.14)
        x, y = r * math.cos(a), r * math.sin(a)
        sx, sy, sz = rng.uniform(0.2, 0.6), rng.uniform(4, 25), rng.uniform(2.5, 8)
        body += mw.box(f'wall_{i}', round(x, 2), round(y, 2), round(sx, 2), round(sy, 2),
                       round(sz, 2), grey(rng, 0.05, 0.85, 0.15), round(yaw, 2))
        obstacles.append(dict(kind='box', x=x, y=y, hx=sx / 2, hy=sy / 2, yaw=yaw))
    for i, h in enumerate(humans):
        body += mw.include(h, h, 0.0, 3.0 * i, z=PARK_Z)
    body += mw.include('eval_rig', 'rig', -5.0, 0.0, z=1.9)
    sun = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1.0])
    sun /= np.linalg.norm(sun)
    hdr = dict(bg=grey(rng, 0.3, 0.95, 0.05), sun_dir=' '.join(f'{v:.3f}' for v in sun),
               gui_cam='-8 0 4 0 0.3 0')
    sdf = mw.HEADER.format(name=name, extra_plugins=mw.FOG_PLUGIN if fogged else '', **hdr)
    if fogged:
        body += mw.fog(0.0, 0.0, 26, int(rng.integers(6, 17)))
    return sdf + body + mw.FOOTER, obstacles


def main():
    os.makedirs(OUT, exist_ok=True)
    humans = json.load(open(os.path.join(HERE, 'models', 'humans.json')))
    split = {'train': [h['name'] for h in humans if h['split'] == 'train'],
             'val': [h['name'] for h in humans if h['split'] == 'val']}
    mw.SHIFT_X = 0.0
    for kind, n, seed in (('train', 10, 100), ('val', 2, 200)):
        for i in range(n):
            rng = np.random.default_rng([seed, i])
            layout_state = rng.bit_generator.state
            for fogged in (False, True):
                rng.bit_generator.state = layout_state        # fog variant: same layout
                name = f'{kind}_{i}' + ('_fog' if fogged else '')
                sdf, obstacles = world(name, rng, split[kind], fogged)
                open(os.path.join(OUT, f'{name}.sdf'), 'w').write(sdf)
                json.dump(dict(obstacles=obstacles, humans=split[kind]),
                          open(os.path.join(OUT, f'{name}.json'), 'w'))
                print('wrote', name, len(obstacles), 'obstacles')


if __name__ == '__main__':
    main()
