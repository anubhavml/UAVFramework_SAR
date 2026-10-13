# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""YOLO11 pose with a parallel depth stream (paper Section 4.2.1 a) and optional cross-scale
attention (4.2.1 c), as Ultralytics model and trainer classes (used by <workspace>/train).

Input: 4-channel B, G, R, depth images (depth 0.1-20 m linear -> 0-255, no return = 255), as
written by `to_rgbd()` below (training images are built the same way). The RGB path is the unchanged
YOLO11-pose network, so COCO weights load as they are; the depth stream is a small CNN whose
stride 8 / 16 / 32 features are added to the backbone outputs P3 / P4 / P5 (layers 4, 6, 10)
through 1x1 convolutions initialised to zero: training starts exactly at the RGB model.
The attention block re-weights the three neck outputs from pooled descriptors of all scales
(channels) and from all scales' activation maps (space), with a zero-initialised gain.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.data import augment
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.nn.modules import SPPF, Conv
from ultralytics.nn.tasks import PoseModel

FUSE_AT = (4, 6, 10)        # P3, P4, P5 outputs of the YOLO11 backbone
NEAR, FAR = 0.1, 20.0


def to_rgbd(bgr, depth_m):
    """BGR image + metric depth already aligned to it -> 4-channel model input."""
    d = np.nan_to_num(np.asarray(depth_m, np.float32), nan=FAR, posinf=FAR, neginf=FAR)
    d = (np.clip((d - NEAR) / (FAR - NEAR), 0, 1) * 255).astype(np.uint8)
    if d.shape[:2] != bgr.shape[:2]:
        d = cv2.resize(d, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.dstack([bgr, d])


def yolo11_sppf(model):
    """Ultralytics 8.4 builds SPPF without the activation after cv1 (a YOLO26 change), but the
    YOLO11 COCO weights were trained with it: restore it so the loaded weights compute what they
    were trained for (checked: features then match the stock model exactly)."""
    for m in model.modules():
        if isinstance(m, SPPF) and isinstance(m.cv1.act, nn.Identity):
            m.cv1.act = nn.SiLU(inplace=True)
    return model


class GesturePoseModel(PoseModel):
    """Plain YOLO11 pose (RGB) with the SPPF fix: stage T1."""

    def __init__(self, cfg='yolo11n-pose.yaml', ch=3, nc=None, data_kpt_shape=(None, None),
                 verbose=True):
        super().__init__(cfg, ch=ch, nc=nc, data_kpt_shape=data_kpt_shape, verbose=verbose)
        yolo11_sppf(self)


class DepthStream(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.stem = nn.Sequential(Conv(1, 16, 3, 2), Conv(16, 32, 3, 2), Conv(32, 32, 3, 1))
        self.p3 = nn.Sequential(Conv(32, 64, 3, 2), Conv(64, 64, 3, 1))
        self.p4 = nn.Sequential(Conv(64, 96, 3, 2), Conv(96, 96, 3, 1))
        self.p5 = nn.Sequential(Conv(96, 128, 3, 2), Conv(128, 128, 3, 1))
        self.fuse = nn.ModuleList(nn.Conv2d(c_in, c, 1) for c_in, c in zip((64, 96, 128), chs))
        for f in self.fuse:
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)

    def forward(self, d):
        p3 = self.p3(self.stem(d))
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [f(p) for f, p in zip(self.fuse, (p3, p4, p5))]


class CrossScaleAttention(nn.Module):
    def __init__(self, chs, r=8):
        super().__init__()
        tot = sum(chs)
        self.chs = list(chs)
        self.mlp = nn.Sequential(nn.Linear(tot, tot // r), nn.SiLU(), nn.Linear(tot // r, tot))
        self.spatial = nn.ModuleList(nn.Conv2d(2 + len(chs) - 1, 1, 7, padding=3) for _ in chs)
        self.gain = nn.Parameter(torch.zeros(len(chs)))

    def forward(self, xs):
        g = torch.cat([x.mean((2, 3)) for x in xs], 1)
        w = torch.sigmoid(self.mlp(g)).split(self.chs, 1)
        maps = [x.mean(1, keepdim=True) for x in xs]
        out = []
        for i, x in enumerate(xs):
            others = [F.interpolate(m, size=x.shape[2:], mode='bilinear', align_corners=False)
                      for j, m in enumerate(maps) if j != i]
            s = torch.sigmoid(self.spatial[i](torch.cat(
                [maps[i], x.amax(1, keepdim=True)] + others, 1)))
            out.append(x + self.gain[i] * x * w[i][:, :, None, None] * s)
        return out


class RGBDPoseModel(PoseModel):
    attention = False

    def __init__(self, cfg='yolo11n-pose.yaml', ch=4, nc=None, data_kpt_shape=(None, None),
                 verbose=True):
        super().__init__(cfg, ch=3, nc=nc, data_kpt_shape=data_kpt_shape, verbose=verbose)
        yolo11_sppf(self)
        self.model.eval()                # probe pass: leave BatchNorm statistics untouched
        with torch.no_grad():
            chs, neck = self._channels()
        self.model.train()
        self.depth = DepthStream(chs)
        self.attn = CrossScaleAttention(neck) if self.attention else None
        self.yaml['channels'] = 4

    def _channels(self):
        y, x = [], torch.zeros(1, 3, 64, 64)
        for m in self.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        head = self.model[-1]
        return [y[i].shape[1] for i in FUSE_AT], [y[i].shape[1] for i in head.f]

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        feats = None
        if x.shape[1] == 4:
            feats = self.depth(x[:, 3:4])
            x = x[:, [2, 1, 0]]          # B, G, R -> R, G, B as the COCO weights expect
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if m is self.model[-1] and getattr(self, 'attn', None) is not None:
                x = self.attn(x)
            x = m(x)
            if feats is not None and m.i in FUSE_AT:
                x = x + feats[FUSE_AT.index(m.i)]
            y.append(x if m.i in self.save or m.i in FUSE_AT else None)
        return x


class RGBDAttnPoseModel(RGBDPoseModel):
    attention = True


class GesturePoseTrainer(PoseTrainer):
    model_cls = GesturePoseModel

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.model_cls(cfg, nc=self.data['nc'], data_kpt_shape=self.data['kpt_shape'],
                               verbose=verbose)
        if weights:
            model.load(weights)
        return model


class RGBDPoseTrainer(GesturePoseTrainer):
    model_cls = RGBDPoseModel


class RGBDAttnPoseTrainer(RGBDPoseTrainer):
    model_cls = RGBDAttnPoseModel


_hsv = augment.RandomHSV.__call__


def _hsv_rgb_channels(self, labels):
    """Ultralytics skips HSV augmentation for images that are not 3-channel; apply it to the
    colour channels of RGB-D images too (the depth channel is left alone)."""
    img = labels['img']
    if img.ndim == 3 and img.shape[-1] == 4:
        bgr = np.ascontiguousarray(img[..., :3])
        labels['img'] = bgr
        _hsv(self, labels)
        img[..., :3] = labels['img']
        labels['img'] = img
        return labels
    return _hsv(self, labels)


augment.RandomHSV.__call__ = _hsv_rgb_channels
