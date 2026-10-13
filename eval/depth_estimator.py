# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Metric monocular depth for real video frames (the demo's real-video depth mode and camera fog):
Depth-Anything-V2-Metric-Outdoor-Small (external/models, downloaded by install_workspace.sh)."""
import os
import sys

import cv2

WS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DEPTH_MODEL = os.path.join(WS, 'external/models/depth-anything-v2-metric-outdoor-small')
W, H = 1280, 720            # real frames are resized to this before the depth model and the gesture node


class DepthEstimator:
    """Metric monocular depth (m) for a BGR image, at the image's resolution."""

    def __init__(self, device='cuda:1'):
        sys.path.insert(0, os.path.join(WS, 'external/pydepth'))
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch, self.device = torch, device
        self.proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        self.net = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()

    def __call__(self, bgr):
        t = self.torch
        with t.no_grad():
            x = self.proc(images=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), return_tensors='pt').to(self.device)
            d = self.net(**x).predicted_depth[:, None]
            d = t.nn.functional.interpolate(d, size=bgr.shape[:2], mode='bicubic', align_corners=False)
        return d[0, 0].clamp(min=0).cpu().numpy()
