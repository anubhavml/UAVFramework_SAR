# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Unified-PDD core (Pose estimation, Distance calculation, Direction classification).

Implements Algorithm 1 of "A ROS2 based Human-In-The-Loop Framework for UAV Search and
Rescue Operations", with the logic of the paper's original scripts:
  depth pre-processing  (32FC1, NaN -> far, clip [near, far], JET colormap)
  joint angles / labels (shoulder in [65, 106] deg and elbow in [100, 180] deg)
Kept free of ROS so it can be unit-tested on its own.
"""
from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np

NEAR, FAR = 0.2, 12.0     # depth clip range (m), as in depth.py / the paper

# COCO keypoint indices (paper Table 2)
KP = {'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
      'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
      'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
      'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16}

# left/right keypoint pairs (eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

SHOULDER_RANGE = (65.0, 106.0)
ELBOW_RANGE = (100.0, 180.0)

LEFT, RIGHT, NEUTRAL, NONE = 'LEFT', 'RIGHT', 'NEUTRAL', 'NONE'


def preprocess_depth(depth, near=NEAR, far=FAR):
    """32FC1 depth -> (clipped metric depth, pseudo-RGB colormap image)."""
    d = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=far, posinf=far, neginf=far)
    d = np.clip(d, near, far)
    norm = ((d - near) / (far - near) * 255).astype(np.uint8)
    return d, cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def calculate_angle(a, b, c):
    """Angle ABC in degrees (same formula as pose.py)."""
    ba, bc = np.asarray(a, float) - b, np.asarray(c, float) - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return float('nan')
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def joint_angles(kp):
    """kp: (17, 2) pixel keypoints -> dict of the four angles used for direction."""
    P = lambda name: np.asarray(kp[KP[name]], float)  # noqa: E731
    return {
        'right_shoulder': calculate_angle(P('right_hip'), P('right_shoulder'), P('right_elbow')),
        'right_elbow': calculate_angle(P('right_shoulder'), P('right_elbow'), P('right_wrist')),
        'left_shoulder': calculate_angle(P('left_hip'), P('left_shoulder'), P('left_elbow')),
        'left_elbow': calculate_angle(P('left_shoulder'), P('left_elbow'), P('left_wrist')),
    }


def face_camera(kp, kc=None):
    """The operator faces the UAV while gesturing, so their right shoulder appears on the image's
    left.  YOLO infers front/back view from facial detail; with a featureless or shaded face it
    can assume a back view and swap every left/right label, turning a RIGHT point into LEFT.
    Restore the front-view labelling.  kp: (17, 2), kc: (17,) keypoint confidences or None."""
    l, r = kp[KP['left_shoulder']], kp[KP['right_shoulder']]
    if not (np.any(l) and np.any(r)) or l[0] >= r[0]:   # missing (0, 0) or already front view
        return kp, kc
    kp = np.array(kp, copy=True)
    kc = None if kc is None else np.array(kc, copy=True)
    for a, b in LR_PAIRS:
        kp[[a, b]] = kp[[b, a]]
        if kc is not None:
            kc[[a, b]] = kc[[b, a]]
    return kp, kc


def side_confidence(kc, direction):
    """Lowest keypoint confidence among the joints that define a LEFT/RIGHT point."""
    side = 'right' if direction == RIGHT else 'left'
    return float(min(kc[KP[f'{side}_{j}']] for j in ('shoulder', 'elbow', 'wrist', 'hip')))


def crop_around(box, shape, scale=1.4):
    """Square crop (x1, y1, x2, y2) centred on a person box, `scale` x its longer side, so that
    a pointing arm the first pass missed (it can stick out ~half a body height) is inside."""
    x1, y1, x2, y2 = box
    half = scale * max(x2 - x1, y2 - y1) / 2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = shape[:2]
    return (int(max(0, cx - half)), int(max(0, cy - half)),
            int(min(w, cx + half)), int(min(h, cy + half)))


def enhance_contrast(img, clip=3.0):
    """CLAHE on the lightness channel: restores local contrast that fog washes out."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def is_hazy(img, max_std=20.0):
    """Fog flattens the whole image: grey-level std is < 20 in the fog worlds, >= 22 in clear ones."""
    small = cv2.resize(img, (320, 320 * img.shape[0] // img.shape[1]), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).std()) < max_std


def _in(v, rng):
    return rng[0] <= v <= rng[1]


def _extended(angles, side):
    return (_in(angles[f'{side}_shoulder'], SHOULDER_RANGE)
            and _in(angles[f'{side}_elbow'], ELBOW_RANGE))


def classify_direction(angles, one_arm=False, both_arms_hover=False, seen=None):
    """Paper Algorithm 1, step 15 (RIGHT is checked first, as in pose.py).

    one_arm: a command needs exactly one extended arm.  Algorithm 1 never looks at the other arm,
    so two-arm signals (e.g. "hover", arms out to both sides) read as RIGHT; on the real
    UAV-GESTURE joint sequences this rule removes most of those false commands.
    both_arms_hover: the gesture rules added to Algorithm 1.  "Stretched out" =
    the paper's extended-arm thresholds; RIGHT / LEFT = the operator's arm (the control node
    turns the UAV left / right, towards where they point):
      both arms stretched out                                -> NEUTRAL (hover)
      one arm stretched out, the other arm seen down or up   -> that arm's direction
      anything else (no arm stretched out, or the other arm
      not seen -- it may be hidden by fog)                   -> NEUTRAL (hover)
    seen: {'left': bool, 'right': bool} from arm_seen; None = both arms assumed seen."""
    right, left = _extended(angles, 'right'), _extended(angles, 'left')
    if both_arms_hover:
        if right == left:                                  # both stretched out, or neither
            return NEUTRAL
        other = 'left' if right else 'right'
        if not np.isfinite(angles[f'{other}_shoulder']) or (seen is not None and not seen[other]):
            return NEUTRAL                                 # other arm not seen
        return RIGHT if right else LEFT
    if one_arm and right and left:
        return NEUTRAL
    if right:
        return RIGHT
    if left:
        return LEFT
    return NEUTRAL


def arm_seen(kc, min_conf=0.5):
    """Per side: shoulder and elbow keypoints located with confidence >= min_conf (Ultralytics'
    visibility threshold).  kc: (17,) keypoint confidences or None (-> both seen)."""
    if kc is None:
        return None
    return {s: bool(kc[KP[f'{s}_shoulder']] >= min_conf and kc[KP[f'{s}_elbow']] >= min_conf)
            for s in ('left', 'right')}


def median_distance(depth_m, box, far=FAR):
    """Median valid depth inside box (x1, y1, x2, y2) -- Algorithm 1, step 10."""
    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
    roi = depth_m[y1:y2, x1:x2]
    valid = roi[roi < far]
    return float(np.median(valid)) if valid.size else float(far)


def torso_box(kp, box, kc=None, min_conf=0.5):
    """ROI for the distance median: the torso quad (shoulders + hips) when those keypoints
    are present, else the full box.  A thin silhouette fills well under half of its bounding
    box, so the full-box median tends to return the background (e.g. the wall behind).
    With keypoint confidences kc, only confidently located keypoints count: a person cut off
    by the image edge (at 1 m the camera sees only the head) gets guessed torso joints on the
    border, so then the head (nose, eyes, ears; from the box top down to the lowest of them)
    is used.  If the quad collapses to a line anyway, keep the informative dimension (the
    torso's columns over the box height, or its rows over the box width)."""
    kp = np.asarray(kp, float) if kp is not None else np.zeros((17, 2))
    found = np.any(kp != 0, axis=1)                     # YOLO reports undetected keypoints as (0, 0)
    if kc is not None:
        found &= np.asarray(kc, float) >= min_conf
    torso = [KP[n] for n in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')]
    head = [KP[n] for n in ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear')]
    if not found[torso].all():
        hd = [i for i in head if found[i]]
        if kc is not None and len(hd) >= 2:
            (x1, _), (x2, y2) = kp[hd].min(0), kp[hd].max(0)
            if x2 - x1 > 2 and y2 - box[1] > 2:
                return (x1, box[1], x2, y2)
        return box
    (x1, y1), (x2, y2) = kp[torso].min(0), kp[torso].max(0)
    wide, tall = (x2 - x1) > 2, (y2 - y1) > 2
    if wide and tall:
        return (x1, y1, x2, y2)
    if wide:
        return (x1, box[1], x2, box[3])
    if tall:
        return (box[0], y1, box[2], y2)
    return box


def map_box(box, src_wh, src_hfov, dst_wh, dst_hfov):
    """Map a pixel box between two co-located pinhole cameras with different FOV/resolution
    (OakD-Lite RGB 1920x1080 @ 1.204 rad  ->  stereo depth 640x480 @ 1.274 rad)."""
    (sw, sh), (dw, dh) = src_wh, dst_wh
    fs = (sw / 2) / np.tan(src_hfov / 2)
    fd = (dw / 2) / np.tan(dst_hfov / 2)
    x1, y1, x2, y2 = box
    conv = lambda u, c_s, c_d: c_d + (u - c_s) * fd / fs  # noqa: E731
    return (conv(x1, sw / 2, dw / 2), conv(y1, sh / 2, dh / 2),
            conv(x2, sw / 2, dw / 2), conv(y2, sh / 2, dh / 2))


@dataclass
class Person:
    box: tuple
    conf: float
    keypoints: np.ndarray
    angles: dict
    direction: str
    distance: float


def lateral_direction(kp, direction):
    """Check a LEFT/RIGHT label against where the extended wrist is.  With the operator facing
    the camera, a sideways point puts the wrist outside both shoulders on the pointing side:
    image-left = operator's RIGHT, image-right = LEFT.  YOLO sometimes attaches the extended arm
    to the wrong shoulder (crossed skeleton), which flips the label; the wrist position does not
    depend on that.  Wrist between the shoulders (not a sideways point) -> NEUTRAL."""
    if direction not in (LEFT, RIGHT):
        return direction
    side = 'right' if direction == RIGHT else 'left'
    wrist = np.asarray(kp[KP[f'{side}_wrist']], float)
    sh = np.asarray([kp[KP['left_shoulder']], kp[KP['right_shoulder']]], float)
    if not wrist.any() or not sh.all():
        return direction                               # cannot check: keep Algorithm 1's label
    if wrist[0] < sh[:, 0].min():
        return RIGHT
    if wrist[0] > sh[:, 0].max():
        return LEFT
    return NEUTRAL


def select_command(people, nearest=False, tol=0.75):
    """Frame-level command from all detected persons (Algorithm 1 loops over every person):
    the most confident non-NEUTRAL person wins; otherwise NEUTRAL; NONE if nobody is seen.
    YOLO often returns both a body-only box and a wider box that includes the pointing arm,
    so looking only at detection [0] (as pose.py does) can miss the gesture.

    nearest: only the operator -- the nearest person -- may command the UAV.  Detections within
    `tol` m of the nearest distance (duplicate boxes of the same person) are considered; without
    depth the tallest box is the operator.  Rejects "people" YOLO finds in the background
    (tree trunks, fog) that would otherwise out-vote a NEUTRAL operator."""
    if not people:
        return NONE, None
    if nearest:
        dists = [p.distance for p in people if np.isfinite(p.distance) and p.distance < FAR]
        if dists:
            people = [p for p in people if p.distance <= min(dists) + tol]
        else:
            tallest = max(people, key=lambda p: p.box[3] - p.box[1])
            people = [p for p in people if _iou(p.box, tallest.box) > 0.3]
    pointing = [p for p in people if p.direction in (LEFT, RIGHT)]
    best = max(pointing or people, key=lambda p: p.conf)
    return (best.direction if pointing else NEUTRAL), best


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class TemporalFilter:
    """Temporal consistency (paper Sec. 4.1.2 / 4.2.2): a label is emitted only when it wins
    at least `min_votes` of the last `window` frames; otherwise hold (NEUTRAL)."""

    def __init__(self, window=5, min_votes=3):
        self.hist = deque(maxlen=window)
        self.min_votes = min_votes

    def update(self, label):
        self.hist.append(label)
        top, votes = Counter(self.hist).most_common(1)[0]
        if votes >= self.min_votes:
            return top
        return NEUTRAL if any(h != NONE for h in self.hist) else NONE
