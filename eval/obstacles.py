#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Obstacle points of a project world near flight height (demonstrations: react:=fly safety).

points(world) -> (N, 2) array of world x, y of every mesh vertex / box sample of the world's
static objects (trees, playground, school, buildings, walls) between 0.3 and 4 m above ground,
cached in data/obstacles_<world>.npy.  safe_distance() gives how far the UAV can fly from a
point along a heading with `clearance` m to all of them.
"""
import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, '..', 'src', 'sar_sim', 'models')
WORLDS = os.path.join(HERE, '..', 'src', 'sar_sim', 'worlds')
Z_BAND = (0.3, 4.0)


def _dae_vertices(path):
    """all vertices of a COLLADA file in metres, model frame (Z up), node transforms applied."""
    s = open(path).read()
    s = re.sub(r'\sxmlns="[^"]+"', '', s, count=1)
    root = ET.fromstring(s)
    unit = root.find('asset/unit')
    scale = float(unit.get('meter')) if unit is not None else 1.0
    up = (root.findtext('asset/up_axis') or 'Z_UP').strip()
    geo = {}
    for g in root.iter('geometry'):
        pts = []
        for fa in g.iter('float_array'):
            if 'position' in fa.get('id', '').lower():
                pts.append(np.array(fa.text.split(), float).reshape(-1, 3))
        if pts:
            geo[g.get('id')] = np.concatenate(pts)
    out = []

    def walk(node, M):
        m = node.find('matrix')
        if m is not None:
            M = M @ np.array(m.text.split(), float).reshape(4, 4)
        for t in node.findall('translate'):
            T = np.eye(4); T[:3, 3] = np.array(t.text.split(), float); M = M @ T
        for sc in node.findall('scale'):
            M = M @ np.diag([*np.array(sc.text.split(), float), 1.0])
        for ig in node.findall('instance_geometry'):
            v = geo.get(ig.get('url', '').lstrip('#'))
            if v is not None:
                out.append((np.c_[v, np.ones(len(v))] @ M.T)[:, :3])
        for ch in node.findall('node'):
            walk(ch, M)

    for vs in root.iter('visual_scene'):
        for n in vs.findall('node'):
            walk(n, np.eye(4))
    if not out:                                   # no scene graph: raw geometry
        out = list(geo.values())
    v = np.concatenate(out) * scale
    if up == 'Y_UP':
        v = v[:, [0, 2, 1]] * [1, -1, 1]
    return v


def _world_points(world):
    s = open(f'{WORLDS}/{world}.sdf').read()
    pts = []
    for uri, pose in re.findall(r'<include>\s*<uri>model://(\w+)</uri>\s*<name>[^<]*</name>\s*'
                                r'<pose>([^<]*)</pose>', s):
        if uri.startswith('sar_human') or uri == 'fog_generator':
            continue
        msdf = open(f'{MODELS}/{uri}/model.sdf').read()
        mesh = re.search(r'<uri>model://\w+/meshes/([^<]+\.dae)</uri>', msdf)
        if not mesh:
            continue
        sc = re.search(r'<scale>([^<]+)</scale>', msdf)
        v = _dae_vertices(f'{MODELS}/{uri}/meshes/{mesh.group(1)}')
        if sc:
            v = v * np.array(sc.group(1).split(), float)
        x, y, z, _, _, yaw = map(float, pose.split())
        c, sn = math.cos(yaw), math.sin(yaw)
        w = np.c_[x + c * v[:, 0] - sn * v[:, 1], y + sn * v[:, 0] + c * v[:, 1], z + v[:, 2]]
        pts.append(w[(w[:, 2] > Z_BAND[0]) & (w[:, 2] < Z_BAND[1]), :2])
    for pose, size in re.findall(r'<model name="wall\w*">\s*<static>true</static>\s*<pose>([^<]*)</pose>'
                                 r'.*?<box><size>([^<]*)</size>', s, re.S):
        x, y, _, _, _, yaw = map(float, pose.split())
        sx, sy, _ = map(float, size.split())
        u, v = np.meshgrid(np.arange(-sx / 2, sx / 2 + 0.1, 0.2), np.arange(-sy / 2, sy / 2 + 0.1, 0.2))
        u, v = u.ravel(), v.ravel()
        c, sn = math.cos(yaw), math.sin(yaw)
        pts.append(np.c_[x + c * u - sn * v, y + sn * u + c * v])
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def points(world):
    world = world.replace('_fog', '')
    cache = f'{HERE}/data/obstacles_{world}.npy'
    if os.path.exists(cache):
        return np.load(cache)
    p = _world_points(world)
    p = np.unique(np.round(p, 1), axis=0)           # 10 cm grid is plenty
    np.save(cache, p)
    return p


def safe_distance(pts, x, y, heading, want, clearance=1.5, step=0.1):
    """largest d <= want such that the segment from (x, y) along heading (world yaw, rad) to d
    keeps `clearance` m from every obstacle point (0 if the start itself is too close)."""
    if len(pts) == 0:
        return want
    near = pts[np.hypot(pts[:, 0] - x, pts[:, 1] - y) < want + clearance + 1]
    d = 0.0
    while d + step <= want + 1e-9:
        px, py = x + (d + step) * math.cos(heading), y + (d + step) * math.sin(heading)
        if len(near) and np.hypot(near[:, 0] - px, near[:, 1] - py).min() < clearance:
            break
        d += step
    return round(d, 1)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, HERE)
    import simtools as st
    for w in ('city', 'forest', 'wall'):
        p = points(w)
        _, (hx, hy, hyaw), _ = st.world_info(w)
        print(f'{w}: {len(p)} obstacle points; person at ({hx}, {hy})')
        for d in (10, 7, 5, 3):
            x, y = hx - d * math.cos(hyaw + math.pi), hy - d * math.sin(hyaw + math.pi)
            face = hyaw + math.pi                    # UAV faces the person
            l = safe_distance(p, x, y, face + math.pi / 2, 8)
            r = safe_distance(p, x, y, face - math.pi / 2, 8)
            print(f'   {d:2d} m from the person: free to the left {l:.1f} m, to the right {r:.1f} m')
