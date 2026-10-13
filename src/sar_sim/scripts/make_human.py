#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Generate the untextured SAR human models (white, pointing) used in the paper's worlds.

The paper's original human was a custom Blender mesh (Section 5.1) that was never published,
so this script builds an equivalent one from the rigged Gazebo Fuel "actor" (stand.dae):
it evaluates the standing pose, swings one arm straight out to the side (elbow straight),
CPU-skins the mesh and writes a static OBJ -- the pose the Unified-PDD algorithm classifies
as LEFT / RIGHT (shoulder 65-106 deg, elbow 100-180 deg).

Outputs  models/sar_human_{right,left,neutral}/{model.sdf,model.config,meshes/human.obj}
         (+ the demonstration poses in EXTRA, only when named: they are used for the gesture
         videos, never for the evaluation)
Requires pycollada  (pip install pycollada  -- installed in <workspace>/external/px4-venv)
Usage    px4-venv && python3 make_human.py                 # the three evaluation humans
         px4-venv && python3 make_human.py both_out ...    # named poses from DESC / EXTRA
"""
import io
import os
import urllib.request
import zipfile

import collada
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(HERE, '..', 'models'))
ACTOR_URL = 'https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/actor.zip'
ACTOR_DAE = os.path.join(HERE, 'actor_stand.dae')
NS = '{http://www.collada.org/2005/11/COLLADASchema}'

ARM_DROP_DEG = 8.0    # extended arm is this many degrees below horizontal
ARM_UP_DEG = 25.0     # a raised ("up") arm is this many degrees out from vertical
HEIGHT = 1.75         # final model height in metres


def fetch_actor():
    if not os.path.exists(ACTOR_DAE):
        with urllib.request.urlopen(ACTOR_URL) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        with open(ACTOR_DAE, 'wb') as fh:
            fh.write(z.read('meshes/stand.dae'))
    return collada.Collada(ACTOR_DAE)


def first_keyframes(doc):
    """joint id -> local 4x4 matrix at the first animation keyframe."""
    out = {}
    for anim in doc.xmlnode.getroot().iter(NS + 'animation'):
        srcs = {s.get('id'): s for s in anim.iter(NS + 'source')}
        samplers = {s.get('id'): s for s in anim.iter(NS + 'sampler')}
        for ch in anim.findall(NS + 'channel'):
            joint = ch.get('target').split('/')[0]
            sampler = samplers[ch.get('source').lstrip('#')]
            for inp in sampler.findall(NS + 'input'):
                if inp.get('semantic') == 'OUTPUT':
                    arr = srcs[inp.get('source').lstrip('#')].find(NS + 'float_array')
                    vals = np.array(arr.text.split(), float)
                    out[joint] = vals[:16].reshape(4, 4)
    return out


def joint_tree(doc):
    """Return (parent map, local rest matrices, root world matrix) for the skeleton nodes."""
    parent, local = {}, {}

    def walk(node, par):
        for ch in node.children:
            if isinstance(ch, collada.scene.Node):
                if ch.xmlnode.get('type') == 'JOINT':
                    parent[ch.id] = par
                    local[ch.id] = ch.matrix.copy()
                    walk(ch, ch.id)
                else:
                    walk(ch, par)

    root_world = np.eye(4)
    for n in doc.scene.nodes:
        if any(isinstance(c, collada.scene.Node) and c.xmlnode.get('type') == 'JOINT'
               for c in n.children):
            root_world = n.matrix.copy()
        walk(n, None)
    return parent, local, root_world


def world_matrices(parent, local, root_world):
    W = {}

    def get(j):
        if j not in W:
            p = parent[j]
            W[j] = (get(p) if p else root_world) @ local[j]
        return W[j]

    for j in local:
        get(j)
    return W


def rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def descendants(parent, j):
    kids = [k for k, p in parent.items() if p == j]
    out = [j]
    for k in kids:
        out += descendants(parent, k)
    return out


def rotate_subtree(W, parent, joint, R, pivot):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pivot - R @ pivot
    for j in descendants(parent, joint):
        W[j] = T @ W[j]


def pos(W, j):
    return W[j][:3, 3]


def pose_and_skin(doc, pointing, arms=None, scale=None):
    """pointing: 'right' | 'left' | other (evaluation humans); arms: {'Right'|'Left': 'out'|'up'}
    for the demonstration poses (overrides pointing)."""
    parent, local = joint_tree(doc)[:2]
    root_world = joint_tree(doc)[2]
    local.update({j: m for j, m in first_keyframes(doc).items() if j in local})
    W = world_matrices(parent, local, root_world)

    # Body frame in skeleton world coordinates
    up = pos(W, 'Head') - pos(W, 'Hips')
    up /= np.linalg.norm(up)
    right = pos(W, 'RightArm') - pos(W, 'LeftArm')
    right -= up * np.dot(right, up)
    right /= np.linalg.norm(right)
    fwd = np.cross(up, right)   # right-handed: forward = up x right

    if arms is None:
        arms = {'Right': 'out'} if pointing == 'right' else {'Left': 'out'} if pointing == 'left' else {}
    for side, how in arms.items():
        out = right if side == 'Right' else -right
        if how == 'up':
            target = np.cos(np.radians(ARM_UP_DEG)) * up + np.sin(np.radians(ARM_UP_DEG)) * out
        else:
            target = np.cos(np.radians(ARM_DROP_DEG)) * out - np.sin(np.radians(ARM_DROP_DEG)) * up
        # swing the whole arm around the shoulder, then straighten elbow and wrist
        for j, child in ((f'{side}Arm', f'{side}ForeArm'),
                         (f'{side}ForeArm', f'{side}Hand'),
                         (f'{side}Hand', f'{side}HandIndex1')):
            cur = pos(W, child) - pos(W, j)
            if np.linalg.norm(cur) > 1e-6:
                rotate_subtree(W, parent, j, rot_between(cur, target), pos(W, j))

    ctrl = doc.controllers[0]
    names = [str(n) for n in ctrl.weight_joints.data.ravel()]
    ibm = np.asarray(ctrl.joint_matrices).reshape(-1, 4, 4) if not isinstance(ctrl.joint_matrices, dict) \
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
    tris = [p if isinstance(p, collada.triangleset.TriangleSet) else p.triangleset()
            for p in geom.primitives]
    faces = np.vstack([t.vertex_index.reshape(-1, 3) for t in tris])

    # to model frame: facing +X, left +Y, up +Z, feet at z=0, scaled to HEIGHT
    Rm = np.vstack([fwd, -right, up])
    out = out @ Rm.T
    out -= [out[:, 0].mean(), out[:, 1].mean(), out[:, 2].min()]
    if scale is None:
        scale = HEIGHT / out[:, 2].max()
    return out * scale, faces


def body_scale(doc):
    """Scale of the standing body (for poses whose raised hand, not the head, is the top)."""
    V, _ = pose_and_skin(doc, 'neutral', scale=1.0)
    return HEIGHT / V[:, 2].max()


def write_obj(path, V, F):
    N = np.zeros_like(V)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    with open(path, 'w') as fh:
        fh.write('# SAR human generated by sar_sim/scripts/make_human.py from the Fuel actor\n')
        fh.writelines(f'v {x:.5f} {y:.5f} {z:.5f}\n' for x, y, z in V)
        fh.writelines(f'vn {x:.5f} {y:.5f} {z:.5f}\n' for x, y, z in N)
        fh.writelines(f'f {a}//{a} {b}//{b} {c}//{c}\n' for a, b, c in F + 1)


SDF = """<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <pose>0 0 0.875 0 0 0</pose>
        <geometry><box><size>0.3 0.5 1.75</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><mesh><uri>model://{name}/meshes/human.obj</uri></mesh></geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <emissive>0.25 0.25 0.25 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

CONFIG = """<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>Untextured SAR human, {desc}. Generated by sar_sim/scripts/make_human.py from the Gazebo Fuel actor (Mingfei/actor, stand.dae).</description>
</model>
"""

DESC = {'right': 'right arm extended (points RIGHT)',
        'left': 'left arm extended (points LEFT)',
        'neutral': 'standing, arms relaxed (NEUTRAL)'}

# demonstration poses (gesture videos only): name -> (arms, description)
EXTRA = {'both_out': ({'Right': 'out', 'Left': 'out'}, 'both arms extended sideways (HOVER)'),
         'both_up': ({'Right': 'up', 'Left': 'up'}, 'both arms raised overhead (HOVER)'),
         'rightout_leftup': ({'Right': 'out', 'Left': 'up'}, 'right arm extended, left arm raised (RIGHT)'),
         'leftout_rightup': ({'Left': 'out', 'Right': 'up'}, 'left arm extended, right arm raised (LEFT)')}

if __name__ == '__main__':
    import sys
    doc = fetch_actor()
    names = sys.argv[1:] or list(DESC)
    for pose in names:
        arms, desc = EXTRA[pose] if pose in EXTRA else (None, DESC[pose])
        name = f'sar_human_{pose}'
        d = os.path.join(MODELS_DIR, name)
        os.makedirs(os.path.join(d, 'meshes'), exist_ok=True)
        up_arm = arms is not None and 'up' in arms.values()
        V, F = pose_and_skin(doc, pose, arms, scale=body_scale(doc) if up_arm else None)
        write_obj(os.path.join(d, 'meshes', 'human.obj'), V, F)
        with open(os.path.join(d, 'model.sdf'), 'w') as fh:
            fh.write(SDF.format(name=name))
        with open(os.path.join(d, 'model.config'), 'w') as fh:
            fh.write(CONFIG.format(name=name, desc=desc))
        ext = V.max(0) - V.min(0)
        print(f'wrote {d}  size x={ext[0]:.2f} y={ext[1]:.2f} z={ext[2]:.2f} m')
