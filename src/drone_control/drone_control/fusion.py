# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Depth-weighted human pose estimation, paper Eq. (1)-(2) (input_mode = fused).

For every person and COCO joint j, four 3D estimates are fused (Eq. 1)

    P_j = alpha D_j + beta RGB_j + gamma T_j + delta C_j

    RGB_j  the joint found by YOLO pose on the RGB frame, back-projected with the depth
    D_j    the joint found by YOLO pose on the colour-mapped depth frame, back-projected
    T_j    the person's joint over the last frames of the same stream (temporal): the mean of the
           observation-based estimates (RGB_j, D_j, C_j fused without T_j), so that T_j never
           feeds on its own output
    C_j    body-plane prior: the observed (RGB / depth) joint moved onto the person's torso distance
           (the operator stands upright, facing the UAV, arms near the torso plane)

with weights from the uncertainty of each estimate (Eq. 2)

    alpha = exp(-lambda sigma_D^2) / sum_i exp(-lambda sigma_i^2),  likewise beta, gamma, delta

over the estimates available for that joint. The fused joints are projected back to RGB pixels,
and Algorithm 1 (with the gesture rules and line 13b) decides the direction from them.
All distances in metres, in the RGB camera frame (pinhole, co-located depth camera).
"""
import warnings
from collections import deque

import numpy as np

from drone_control import pdd
from drone_control import sensor_model as sm
from drone_control.pipeline import GesturePipeline

MIN_KC = 0.05         # keypoint confidence below this (or (0, 0)) = joint not found
BAND = 0.6            # depth samples within this of the torso distance belong to the person (m)
Z_DEFAULT = 5.0       # back-projection depth when the person has no depth at all (m)
PX_640 = 4.0          # keypoint localisation noise at 640 px input, at confidence 1 (px)
SIGMA_T0 = 0.05       # floor of the temporal estimate's standard deviation (m)
SIGMA_C = 0.25        # standard deviation of the body-plane prior (m)
WINDOW = 5            # frames kept for T_j
TRACK_GATE = 0.75     # torso moved less than this between frames = same person (m)
MATCH_IOU = 0.3       # RGB / depth detections of the same person


def weights(sigma2, lam):
    """Eq. (2): softmax(-lambda sigma^2) over axis 0; NaN sigma^2 = estimate not available."""
    ok = np.isfinite(sigma2)
    a = np.where(ok, -lam * np.where(ok, sigma2, 0.0), -np.inf)
    top = a.max(axis=0, keepdims=True)
    e = np.where(ok, np.exp(a - np.where(np.isfinite(top), top, 0.0)), 0.0)
    tot = e.sum(axis=0, keepdims=True)
    return np.divide(e, tot, out=np.zeros_like(e), where=tot > 0)


def fuse(est, sigma2, lam):
    """Eq. (1): est (M, 17, 3) estimates, sigma2 (M, 17) -> fused (17, 3), weights (M, 17).
    Joints with no available estimate are NaN."""
    w = weights(sigma2, lam)
    p = (np.where(np.isfinite(est), est, 0.0) * w[..., None]).sum(axis=0)
    p[w.sum(axis=0) == 0] = np.nan
    return p, w


class Camera:
    """Pinhole camera of the RGB frame: square pixels, principal point at the centre."""

    def __init__(self, w, h, hfov):
        self.w, self.h = w, h
        self.f = (w / 2) / np.tan(hfov / 2)

    def back(self, uv, z):
        """(N, 2) pixels, (N,) depths -> (N, 3) points."""
        uv = np.asarray(uv, float)
        return np.c_[(uv[:, 0] - self.w / 2) * z / self.f, (uv[:, 1] - self.h / 2) * z / self.f, z]

    def project(self, p):
        """(N, 3) points -> (N, 2) pixels; NaN points -> (0, 0) = not found."""
        uv = np.c_[self.w / 2 + self.f * p[:, 0] / p[:, 2], self.h / 2 + self.f * p[:, 1] / p[:, 2]]
        return np.where(np.isfinite(uv), uv, 0.0)


def depth_to_rgb_px(kp, depth_wh, depth_hfov, rgb_wh, rgb_hfov):
    """Depth-image keypoints -> RGB pixels of the co-located camera; (0, 0) stays not found."""
    (dw, dh), (w, h) = depth_wh, rgb_wh
    s = ((w / 2) / np.tan(rgb_hfov / 2)) / ((dw / 2) / np.tan(depth_hfov / 2))
    out = np.c_[w / 2 + (kp[:, 0] - dw / 2) * s, h / 2 + (kp[:, 1] - dh / 2) * s]
    return np.where(np.any(kp != 0, axis=1, keepdims=True), out, 0.0)


class FusedPipeline(GesturePipeline):
    """Algorithm 1 (gesture rules + line 13b) on the depth-weighted fused pose, Eq. (1)-(2).
    lam: lambda of Eq. (2); 3000 was chosen on the validation worlds (train/PROTOCOL.md)."""

    def __init__(self, model, lam=3000.0, **kw):
        kw = dict(dict(imgsz=640, refine=False, face_camera=True, kp_conf=0.0, dehaze=False,
                       one_arm=False, lateral_check=False, nearest_operator=False,
                       both_arms_hover=True), **kw)
        kw.pop('mode', None)
        super().__init__(model, mode='fused', **kw)
        self.lam = lam
        self.reset()

    def reset(self):
        """Forget the temporal history (new video / independent frame)."""
        self.track = deque(maxlen=WINDOW)          # observation-based (17, 3) skeletons, tracked person
        self.track_torso = None
        self.last_weights = None                   # (4, 17) weights of the selected person

    def process(self, rgb, depth_raw):
        if rgb is None:
            return None, [], pdd.NONE, None
        h, w = rgb.shape[:2]
        cam = Camera(w, h, self.rgb_hfov)
        if depth_raw is not None:
            depth_m, colored = pdd.preprocess_depth(depth_raw)
            z = sm.align_depth(depth_raw, self.depth_hfov, (w, h), self.rgb_hfov)
            z = np.where(np.isfinite(z) & (z > 0) & (z < pdd.FAR), z, np.nan).astype(np.float32)
        else:
            depth_m = colored = None
            z = np.full((h, w), np.nan, np.float32)
        obs_rgb = [self._observation(*d) for d in self.detect(rgb)]
        obs_d = []
        if colored is not None:
            dwh = (colored.shape[1], colored.shape[0])
            for box, conf, kp, kc, _ in self.detect(colored):
                kp, kc = pdd.face_camera(kp, kc)
                box = pdd.map_box(box, dwh, self.depth_hfov, (w, h), self.rgb_hfov)
                obs_d.append((np.asarray(box, float), float(conf),
                              depth_to_rgb_px(kp, dwh, self.depth_hfov, (w, h), self.rgb_hfov), kc))
        pairs = self._match(obs_rgb, obs_d)
        people, obs_all = [], []
        for o_rgb, o_d in pairs:
            person, p_obs = self._fuse_person(cam, z, o_rgb, o_d)
            people.append(person)
            obs_all.append(p_obs)
        if depth_m is not None:
            dh, dw = depth_m.shape[:2]
            for i, p in enumerate(people):
                p.distance = pdd.median_distance(depth_m, pdd.map_box(
                    pdd.torso_box(p.keypoints, p.box, p.kc), (w, h), self.rgb_hfov,
                    (dw, dh), self.depth_hfov))
        raw, best = pdd.select_command(people, nearest=self.nearest)
        if best is not None:
            i = people.index(best)
            self.last_weights = best.weights
            self._update_track(obs_all[i], best.torso)
        return rgb, people, raw, best

    def _observation(self, box, conf, kp, kc, _cls=0):
        kp, kc = pdd.face_camera(kp, kc)
        return np.asarray(box, float), float(conf), np.asarray(kp, float), np.asarray(kc, float)

    @staticmethod
    def _match(obs_rgb, obs_d):
        """Greedy IoU matching -> [(rgb or None, depth or None)]."""
        pairs, used = [], set()
        cand = sorted(((pdd._iou(a[0], b[0]), i, j) for i, a in enumerate(obs_rgb)
                       for j, b in enumerate(obs_d)), reverse=True)
        taken_r = set()
        for iou, i, j in cand:
            if iou <= MATCH_IOU or i in taken_r or j in used:
                continue
            pairs.append((obs_rgb[i], obs_d[j]))
            taken_r.add(i)
            used.add(j)
        pairs += [(o, None) for i, o in enumerate(obs_rgb) if i not in taken_r]
        pairs += [(None, o) for j, o in enumerate(obs_d) if j not in used]
        return pairs

    def _joint_depth(self, z, uv, torso_d, r):
        """Per keypoint: (depth, variance of the samples) from the person's depth near it."""
        h, w = z.shape
        zs, vs = np.full(len(uv), np.nan), np.zeros(len(uv))
        for j, (u, v) in enumerate(uv):
            if not (u or v):
                continue
            x0, x1 = int(max(0, u - r)), int(min(w, u + r + 1))
            y0, y1 = int(max(0, v - r)), int(min(h, v + r + 1))
            s = z[y0:y1, x0:x1]
            s = s[np.isfinite(s)]
            if np.isfinite(torso_d):
                s = s[np.abs(s - torso_d) <= BAND]
            if s.size:
                zs[j], vs[j] = np.median(s), np.var(s)
            elif np.isfinite(torso_d):
                zs[j] = torso_d
        return zs, vs

    def _fuse_person(self, cam, z, o_rgb, o_d):
        with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)          # NaN means: joint not seen
            return self._fuse(cam, z, o_rgb, o_d)

    def _fuse(self, cam, z, o_rgb, o_d):
        obs = [o for o in (o_d, o_rgb) if o is not None]         # order: D, RGB
        box = o_rgb[0] if o_rgb is not None else o_d[0]
        conf = max(o[1] for o in obs)
        # torso distance: median person depth inside the torso quad of the best-located keypoints
        kp0, kc0 = (o_rgb[2], o_rgb[3]) if o_rgb is not None else (o_d[2], o_d[3])
        x1, y1, x2, y2 = (int(round(v)) for v in pdd.torso_box(kp0, box, kc0))
        roi = z[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        roi = roi[np.isfinite(roi)]
        torso_d = float(np.median(roi)) if roi.size else np.nan
        r = max(2, round(cam.w / 256))
        s_px = PX_640 * cam.w / 640
        est, sig, kcs = np.full((4, 17, 3), np.nan), np.full((4, 17), np.nan), np.zeros((2, 17))
        for m, o in ((0, o_d), (1, o_rgb)):                     # 0 = D_j, 1 = RGB_j
            if o is None:
                continue
            kp, kc = o[2], o[3]
            ok = np.any(kp != 0, axis=1) & (kc >= MIN_KC)
            zj, vj = self._joint_depth(z, np.where(ok[:, None], kp, 0.0), torso_d, r)
            zj = np.where(np.isfinite(zj), zj, Z_DEFAULT if not np.isfinite(torso_d) else torso_d)
            p = cam.back(kp, zj)
            est[m][ok] = p[ok]
            sig[m][ok] = ((s_px * zj / cam.f) ** 2 / np.maximum(kc, MIN_KC) + vj)[ok]
            kcs[m] = np.where(ok, kc, 0.0)
        seen = np.isfinite(sig[:2]).any(axis=0)
        torso3 = np.nanmean(est[:2, [5, 6, 11, 12]].reshape(-1, 3), axis=0) \
            if np.isfinite(est[:2, [5, 6, 11, 12]]).any() else np.full(3, np.nan)
        # T_j: the tracked person's recent observation-based joints, if this is that person
        if self.track and self.track_torso is not None and np.all(np.isfinite(torso3)) \
                and np.linalg.norm(torso3 - self.track_torso) < TRACK_GATE:
            hist = np.stack(self.track)                           # (n, 17, 3)
            n = np.isfinite(hist).all(axis=2).sum(axis=0)
            t_mean = np.nanmean(np.where(np.isfinite(hist), hist, np.nan), axis=0) \
                if n.any() else np.full((17, 3), np.nan)
            t_var = np.nansum(np.nanvar(hist, axis=0), axis=1) if n.any() else np.full(17, np.nan)
            ok = seen & (n > 0)
            est[2][ok] = t_mean[ok]
            sig[2][ok] = t_var[ok] + SIGMA_T0 ** 2
        # C_j: the observations' mean moved onto the torso plane
        if np.isfinite(torso_d):
            obs_mean = np.nanmean(np.where(np.isfinite(est[:2]), est[:2], np.nan), axis=0) \
                if seen.any() else np.full((17, 3), np.nan)
            c = obs_mean * (torso_d / obs_mean[:, 2:3])       # same pixel ray, z = torso distance
            ok = seen & np.isfinite(c).all(axis=1)
            est[3][ok] = c[ok]
            sig[3][ok] = SIGMA_C ** 2
        p3, wts = fuse(est, sig, self.lam)
        no_t = sig.copy()
        no_t[2] = np.nan
        p_obs, _ = fuse(est, no_t, self.lam)                      # history for T_j (no feedback)
        kp = cam.project(p3)
        wd = wts[:2].sum(axis=0)
        kc = np.divide((wts[:2] * kcs).sum(axis=0), wd, out=np.zeros(17), where=wd > 0)
        d = self.direction(kp, kc)
        person = pdd.Person(tuple(float(v) for v in box), conf, kp, pdd.joint_angles(kp), d, pdd.FAR)
        person.kc, person.weights, person.torso = kc, wts, torso3
        return person, p_obs

    def _update_track(self, p_obs, torso3):
        if not np.all(np.isfinite(torso3)):
            return
        if self.track_torso is not None and np.linalg.norm(torso3 - self.track_torso) >= TRACK_GATE:
            self.track.clear()                                    # another person: start over
        self.track.append(p_obs)
        self.track_torso = torso3
