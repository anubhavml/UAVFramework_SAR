#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Training humans: the rigged Gazebo actor (as in sar_sim/scripts/make_human.py) posed with
sampled gesture parameters, body proportions and colours, plus the 17 COCO keypoints.

    ../external/px4-venv/bin/python make_humans.py      (pycollada lives in px4-venv)

Writes models/thuman_{train,val}_<i>/{model.sdf, model.config, meshes/human.obj} and
models/humans.json: [{name, split, label, params, kp: 17 x (x, y, z) in the model frame
(x forward, y left, z up)}].  Training (seed 0) and validation (seed 1) humans are distinct.

Labels follow PROTOCOL.md: LEFT / RIGHT = that arm raised sideways (abduction 60-110 deg, elbow
bent <= 40 deg), other arm < 35 deg; NEUTRAL = both arms < 35 deg, or both raised (two-arm
signal). 35-60 deg is never generated.  The test meshes (sar_human_*) are not reproduced: their
exact pose (abduction 82 deg, straight elbow, no forward angle) is excluded.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'sar_sim', 'scripts'))
import make_human as mh  # noqa: E402

OUT = os.path.join(HERE, 'models')
# COCO keypoint -> actor joint (body) ; head keypoints come from mesh vertices (see head_vertices)
JOINTS = {5: 'LeftArm', 6: 'RightArm', 7: 'LeftForeArm', 8: 'RightForeArm', 9: 'LeftHand',
          10: 'RightHand', 11: 'LeftUpLeg', 12: 'RightUpLeg', 13: 'LeftLeg', 14: 'RightLeg',
          15: 'LeftFoot', 16: 'RightFoot'}


def sample(label, rng):
    """Gesture parameters (degrees) for one human."""
    down = lambda: dict(abduct=rng.uniform(3, 32), fwd=rng.uniform(-20, 30),  # noqa: E731
                        elbow=rng.uniform(0, 45))
    raised = lambda: dict(abduct=rng.uniform(60, 110), fwd=rng.uniform(-25, 30),  # noqa: E731
                          elbow=rng.uniform(0, 40))
    if label == 'NEUTRAL':
        both_up = rng.random() < 0.25
        arms = {'Left': raised() if both_up else down(), 'Right': raised() if both_up else down()}
    else:
        side = 'Left' if label == 'LEFT' else 'Right'
        other = 'Right' if side == 'Left' else 'Left'
        arms = {side: raised(), other: down()}
        a = arms[side]
        if abs(a['abduct'] - 82) < 2 and a['elbow'] < 2 and abs(a['fwd']) < 2:   # test pose
            a['fwd'] = 5.0
    return dict(arms=arms, height=rng.uniform(1.55, 1.95), width=rng.uniform(0.88, 1.15),
                colour=colour(rng))


def colour(rng):
    """Mostly varied clothing colours; some white like the paper's (untextured) human."""
    if rng.random() < 0.2:
        return [1.0, 1.0, 1.0]
    return list(np.round(rng.uniform(0.05, 0.95, 3), 3))


def pose(doc, p):
    """-> mesh vertices, faces, joint world positions, all in the model frame."""
    parent, local, root_world = mh.joint_tree(doc)
    local.update({j: m for j, m in mh.first_keyframes(doc).items() if j in local})
    W = mh.world_matrices(parent, local, root_world)
    up = mh.pos(W, 'Head') - mh.pos(W, 'Hips')
    up /= np.linalg.norm(up)
    right = mh.pos(W, 'RightArm') - mh.pos(W, 'LeftArm')
    right -= up * np.dot(right, up)
    right /= np.linalg.norm(right)
    fwd = np.cross(up, right)
    for side, a in p['arms'].items():
        out = right if side == 'Right' else -right
        ab, fw, el = np.radians([a['abduct'], a['fwd'], a['elbow']])
        upper = np.cos(ab) * -up + np.sin(ab) * (np.cos(fw) * out + np.sin(fw) * fwd)
        n = fwd + up - upper * np.dot(fwd + up, upper)          # elbow bends forward/up
        n /= np.linalg.norm(n)
        fore = np.cos(el) * upper + np.sin(el) * n
        for j, child, d in ((f'{side}Arm', f'{side}ForeArm', upper),
                            (f'{side}ForeArm', f'{side}Hand', fore),
                            (f'{side}Hand', f'{side}HandIndex1', fore)):
            cur = mh.pos(W, child) - mh.pos(W, j)
            if np.linalg.norm(cur) > 1e-6:
                mh.rotate_subtree(W, parent, j, mh.rot_between(cur, d), mh.pos(W, j))
    V, F = skin(doc, W)
    J = {j: mh.pos(W, j) for j in W}
    Rm = np.vstack([fwd, -right, up])
    V, J = V @ Rm.T, {j: Rm @ x for j, x in J.items()}
    hips = (J['LeftUpLeg'] + J['RightUpLeg']) / 2       # centre on the hips, not the (arm-
    off = np.array([hips[0], hips[1], V[:, 2].min()])     # dependent) vertex mean
    s = p['height'] / (V[:, 2].max() - off[2])
    scale = np.array([s * p['width'], s * p['width'], s])
    return (V - off) * scale, F, {j: (x - off) * scale for j, x in J.items()}


def skin(doc, W):
    ctrl = doc.controllers[0]
    names = [str(n) for n in ctrl.weight_joints.data.ravel()]
    ibm = np.asarray(ctrl.joint_matrices).reshape(-1, 4, 4) \
        if not isinstance(ctrl.joint_matrices, dict) \
        else np.array([ctrl.joint_matrices[n] for n in names])
    skin_m = np.array([W[n] @ ibm[i] for i, n in enumerate(names)])
    geom = ctrl.geometry
    V = geom.primitives[0].vertex
    Vh = np.c_[V, np.ones(len(V))] @ ctrl.bind_shape_matrix.T
    weights = ctrl.weights.data.ravel()
    out = np.zeros((len(V), 3))
    for vi, infl in enumerate(ctrl.index):
        acc = np.zeros(4)
        for jidx, widx in infl:
            acc += weights[widx] * (skin_m[jidx] @ Vh[vi])
        out[vi] = acc[:3]
    tris = [t if isinstance(t, mh.collada.triangleset.TriangleSet) else t.triangleset()
            for t in geom.primitives]
    return out, np.vstack([t.vertex_index.reshape(-1, 3) for t in tris])


def head_vertices(V, J):
    """Vertex indices for nose, eyes and ears, picked once on a neutral pose: nose = most forward
    head vertex at eye level, eyes 3.2 cm to each side of it and 3.5 cm up, ears = most lateral
    head vertices at that height."""
    head = J['Head']
    near = np.linalg.norm(V - head, axis=1) < 0.16
    lvl = near & (np.abs(V[:, 2] - (head[2] + 0.06)) < 0.03)
    idx = np.nonzero(lvl)[0]
    nose = idx[np.argmax(V[idx, 0])]
    pick = lambda target: int(np.argmin(np.linalg.norm(V - target, axis=1) + 10 * ~near))  # noqa
    n = V[nose]
    eyes = [pick(n + [-0.02, 0.032, 0.035]), pick(n + [-0.02, -0.032, 0.035])]   # left = +y
    ears = [int(idx[np.argmax(V[idx, 1])]), int(idx[np.argmin(V[idx, 1])])]
    return [int(nose), eyes[0], eyes[1], ears[0], ears[1]]


SDF = """<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><mesh><uri>model://{name}/meshes/human.obj</uri></mesh></geometry>
        <material>
          <ambient>{c} 1</ambient>
          <diffuse>{c} 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <emissive>{e} 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
CONFIG = """<?xml version="1.0"?>
<model><name>{name}</name><version>1.0</version><sdf version="1.9">model.sdf</sdf>
  <description>Training human ({label}), generated by train/make_humans.py</description>
</model>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=40, help='training humans per class')
    ap.add_argument('--n-val', type=int, default=10, help='validation humans per class')
    a = ap.parse_args()
    doc = mh.fetch_actor()
    V0, _, J0 = pose(doc, dict(arms={}, height=1.75, width=1.0, colour=[1, 1, 1]))
    hv = head_vertices(V0, J0)
    humans = []
    for split, n, seed in (('train', a.n_train, 0), ('val', a.n_val, 1)):
        rng = np.random.default_rng(seed)
        for i, label in enumerate(['LEFT', 'RIGHT', 'NEUTRAL'] * n):
            p = sample(label, rng)
            V, F, J = pose(doc, p)
            kp = [V[v] for v in hv] + [J[JOINTS[k]] for k in range(5, 17)]
            name = f'thuman_{split}_{i:03d}'
            d = os.path.join(OUT, name)
            os.makedirs(os.path.join(d, 'meshes'), exist_ok=True)
            mh.write_obj(os.path.join(d, 'meshes', 'human.obj'), V, F)
            c = ' '.join(f'{x:.3f}' for x in p['colour'])
            e = ' '.join(f'{0.25 * x:.3f}' for x in p['colour'])
            open(os.path.join(d, 'model.sdf'), 'w').write(SDF.format(name=name, c=c, e=e))
            open(os.path.join(d, 'model.config'), 'w').write(CONFIG.format(name=name, label=label))
            humans.append(dict(name=name, split=split, label=label, params=p,
                               kp=[[round(float(x), 4) for x in k] for k in kp]))
        print(split, len([h for h in humans if h['split'] == split]), 'humans', flush=True)
    json.dump(humans, open(os.path.join(OUT, 'humans.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
