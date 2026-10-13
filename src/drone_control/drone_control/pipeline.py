# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Unified-PDD per-frame pipeline (Algorithm 1 + optional robustness additions), free of ROS.

Shared by gesture_node (live) and the offline evaluation in <workspace>/eval, so both run
exactly the same code.

    pipe = GesturePipeline(YOLO(weights), mode='rgb')          # robustness additions on
    pipe = GesturePipeline.published(YOLO(weights), mode='rgb') # Algorithm 1, no additions
    pipe = GesturePipeline.trained(YOLO(trained_pt), mode='rgbd')  # gesture classes from the model
    pipe = fusion.FusedPipeline(YOLO(weights))                  # depth-weighted fusion, Eq. (1)-(2)
    frame, people, raw, best = pipe.process(rgb_bgr8, depth_32fc1)
"""
import os

import numpy as np

from drone_control import pdd
from drone_control import sensor_model as sm


# stock COCO yolo11n-pose (kept here, not in gesture_node, so offline tools need no ROS):
# $SAR_POSE_WEIGHTS, else <workspace>/external/models/yolo11n-pose.pt (install_workspace.sh downloads
# it); the workspace is $ROSWS (px4_env.sh)
_HERE = os.path.abspath(__file__)          # <workspace>/install/drone_control/lib/.../pipeline.py
_WS = os.environ.get('ROSWS') or (_HERE.split(os.sep + 'install' + os.sep)[0] if os.sep + 'install' + os.sep in _HERE
                                  else os.path.normpath(os.path.join(os.path.dirname(_HERE), '..', '..', '..')))
DEFAULT_WEIGHTS = os.environ.get('SAR_POSE_WEIGHTS') or os.path.join(_WS, 'external/models/yolo11n-pose.pt')


class GesturePipeline:
    """Additions (all optional, see gesture_node docstring / RUN_SAR.md): refine, face_camera,
    kp_conf, dehaze (rgb only), one_arm, lateral_check, nearest_operator;
    imgsz 0 = 640 rgb / 1280 depth."""

    def __init__(self, model, mode='rgb', conf=0.25, imgsz=0, device='cpu', half=False,
                 refine=True, face_camera=True, kp_conf=0.5, dehaze=True, haze_std=20.0,
                 one_arm=True, lateral_check=True, nearest_operator=True, both_arms_hover=False,
                 rgb_hfov=1.204, depth_hfov=1.274, threads=0):
        self.model, self.mode, self.conf = model, mode, conf
        # the small depth frame is upscaled so a person at ~8 m is still detected
        self.imgsz = imgsz or (1280 if mode == 'depth' else 640)
        self.device, self.half = device, half
        self.refine, self.face, self.kp_conf, self.one_arm = refine, face_camera, kp_conf, one_arm
        self.both_arms_hover = both_arms_hover
        self.lateral, self.nearest = lateral_check, nearest_operator
        self.dehaze = dehaze and mode != 'depth'      # CLAHE means nothing on a colormap
        self.haze_std = haze_std
        self.rgb_hfov, self.depth_hfov = rgb_hfov, depth_hfov
        self.threads = threads      # CPU threads; Ultralytics resets torch to min(8, cores-1)
        self.nan_keypoints = 0      # keypoints dropped because the model returned NaN

    @classmethod
    def published(cls, model, mode='rgb', **kw):
        """Algorithm 1 without additions, 640 px input in both modes. Algorithm 1 of the paper:
        both_arms_hover=True (both arms extended -> NEUTRAL) and face_camera=True (left / right
        as seen by the UAV)."""
        kw = dict(dict(imgsz=640, refine=False, face_camera=False, kp_conf=0.0, dehaze=False,
                       one_arm=False, lateral_check=False, nearest_operator=False), **kw)
        return cls(model, mode=mode, **kw)

    @classmethod
    def trained(cls, model, mode='rgb', **kw):
        """A gesture model trained with LEFT / RIGHT / NEUTRAL classes (<workspace>/train):
        the class of the detection is the direction (no joint angles, no additions); mode
        'rgbd' feeds the RGB image plus the depth aligned to it. Nearest person commands."""
        kw = dict(dict(imgsz=640, refine=False, face_camera=False, kp_conf=0.0, dehaze=False,
                       one_arm=False, lateral_check=False, nearest_operator=True), **kw)
        p = cls(model, mode=mode, **kw)
        p.classes = {i: n.upper() for i, n in model.names.items()}
        return p

    classes = None              # class id -> direction, for trained gesture models

    def reset(self):
        """Per-stream state (none here; fusion.FusedPipeline keeps a temporal history)."""

    def process(self, rgb, depth_raw):
        """rgb: HxWx3 bgr8 or None; depth_raw: 32FC1 metres or None.
        Returns (frame the pose ran on, [pdd.Person], raw frame label, selected person)."""
        depth_m = colored = None
        if depth_raw is not None:
            depth_m, colored = pdd.preprocess_depth(depth_raw)
        frame = colored if self.mode == 'depth' else rgb
        if frame is None:
            return None, [], pdd.NONE, None
        net_in = frame
        if self.mode == 'rgbd':
            from drone_control.rgbd_model import to_rgbd
            h, w = rgb.shape[:2]
            aligned = (sm.align_depth(depth_raw, self.depth_hfov, (w, h), self.rgb_hfov)
                       if depth_raw is not None else np.full((h, w), np.inf, np.float32))
            net_in = to_rgbd(rgb, aligned)
        dets = self.detect(net_in)
        if not dets and self.dehaze and pdd.is_hazy(frame, self.haze_std):
            dets = self.detect(pdd.enhance_contrast(frame))
        people = [self.person(frame, depth_m, *d) for d in dets]
        raw, best = pdd.select_command(people, nearest=self.nearest)
        return frame, people, raw, best

    def detect(self, img):
        """YOLO pose -> [(xyxy, conf, keypoints (17, 2), keypoint confidences (17,), class)]."""
        r = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, device=self.device,
                               half=self.half, verbose=False)[0]
        if self.threads:                                 # after Ultralytics' select_device()
            import torch
            if torch.get_num_threads() != self.threads:
                torch.set_num_threads(self.threads)
        if r.boxes is None or r.keypoints is None or not len(r.boxes):
            return []
        kc = r.keypoints.conf
        kc = kc.cpu().numpy() if kc is not None else np.ones(r.keypoints.xy.shape[:2])
        kp, boxes = r.keypoints.xy.cpu().numpy(), r.boxes.xyxy.cpu().numpy()
        # FP16 can yield NaN outputs on some GPUs: NaN keypoint = not found (0, 0)
        bad = np.isnan(kp).any(axis=2) | np.isnan(kc)
        self.nan_keypoints += int(bad.sum())
        kp, kc = np.where(bad[..., None], 0.0, kp), np.where(bad, 0.0, kc)
        ok = ~np.isnan(boxes).any(axis=1)
        return list(zip(boxes[ok], r.boxes.conf.cpu().numpy()[ok], kp[ok], kc[ok],
                        r.boxes.cls.cpu().numpy().astype(int)[ok]))

    def direction(self, kp, kc):
        d = pdd.classify_direction(pdd.joint_angles(kp), one_arm=self.one_arm,
                                   both_arms_hover=self.both_arms_hover,
                                   seen=pdd.arm_seen(kc) if self.both_arms_hover else None)
        if d in (pdd.LEFT, pdd.RIGHT) and pdd.side_confidence(kc, d) < self.kp_conf:
            return pdd.NEUTRAL
        return pdd.lateral_direction(kp, d) if self.lateral else d

    def person(self, frame, depth_m, box, conf, kp, kc, cls=0):
        if self.classes is not None:
            d = self.classes.get(int(cls), pdd.NEUTRAL)
        else:
            if self.face:
                kp, kc = pdd.face_camera(kp, kc)
            d = self.direction(kp, kc)
        if self.refine and d not in (pdd.LEFT, pdd.RIGHT):
            x1, y1, x2, y2 = pdd.crop_around(box, frame.shape)
            cx = (box[0] + box[2]) / 2 - x1              # same person: closest to the crop centre
            off = np.array([x1, y1], np.float32)
            for enhance in ((False, True) if self.dehaze else (False,)):
                crop = frame[y1:y2, x1:x2]
                sub = self.detect(pdd.enhance_contrast(crop) if enhance else crop)
                if not sub:
                    continue
                b2, c2, kp2, kc2, _ = min(sub, key=lambda s: abs((s[0][0] + s[0][2]) / 2 - cx))
                kp2 = np.where(kp2 > 0, kp2 + off, 0.0)  # keep (0, 0) = keypoint not found
                if self.face:
                    kp2, kc2 = pdd.face_camera(kp2, kc2)
                d2 = self.direction(kp2, kc2)
                if d2 in (pdd.LEFT, pdd.RIGHT):
                    box, conf, kp, kc, d = b2 + np.r_[off, off], max(conf, c2), kp2, kc2, d2
                    break
        xyxy = tuple(float(v) for v in box)
        if depth_m is None:
            dist = pdd.FAR
        elif self.mode == 'depth':
            dist = pdd.median_distance(depth_m, pdd.torso_box(kp, xyxy, kc))
        else:
            h, w = frame.shape[:2]
            dh, dw = depth_m.shape[:2]
            dist = pdd.median_distance(depth_m, pdd.map_box(
                pdd.torso_box(kp, xyxy, kc), (w, h), self.rgb_hfov, (dw, dh), self.depth_hfov))
        return pdd.Person(xyxy, float(conf), kp, pdd.joint_angles(kp), d, dist)
