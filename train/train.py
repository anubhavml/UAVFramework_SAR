#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Train YOLOv11-L_total (stage T4) or one of its intermediate stages from the COCO yolo11n-pose
weights (train/PROTOCOL.md; paper Section 7.2).

    python3 train.py T1            # RGB, gesture classes
    python3 train.py T2            # + parallel depth stream (RGB-D)
    python3 train.py T3            # + cross-scale attention
    python3 train.py T4            # + depth and temporal loss terms = YOLOv11-L_total (needs build_t4.py)
    python3 train.py T1R ... T4R   # the same, trained only on real UAV-GESTURE images (build_real*.py)
    python3 train.py T4 --data <dataset>/rgbd.yaml    # any dataset in the format of README.md
    -> runs/<stage>/weights/best.pt (selected on the validation set by Ultralytics' fitness)

Ultralytics defaults except fliplr=0 (a mirrored image turns LEFT into RIGHT) and FP32 training.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'eval'))

from ultralytics import YOLO  # noqa: E402

from drone_control.pipeline import DEFAULT_WEIGHTS  # noqa: E402
from drone_control.rgbd_model import (GesturePoseTrainer, RGBDAttnPoseTrainer,  # noqa: E402
                                      RGBDPoseTrainer)
from drone_control.t4_model import T4PoseTrainer  # noqa: E402

STAGES = {'T1': ('rgb', GesturePoseTrainer), 'T2': ('rgbd', RGBDPoseTrainer),
          'T3': ('rgbd', RGBDAttnPoseTrainer), 'T4': ('rgbd', T4PoseTrainer),
          # the same stages trained only on real UAV-GESTURE images (build_real.py, build_real_t4.py)
          'T1R': ('real/rgb', GesturePoseTrainer), 'T2R': ('real/rgbd', RGBDPoseTrainer),
          'T3R': ('real/rgbd', RGBDAttnPoseTrainer), 'T4R': ('real/rgbd', T4PoseTrainer)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=STAGES)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--data', default=None, help='dataset yaml (default data/yolo/<kind>.yaml)')
    ap.add_argument('--name', default=None)
    ap.add_argument('--device', default='0', help="GPU index (Ultralytics sets CUDA_VISIBLE_DEVICES from it)")
    ap.add_argument('--fraction', type=float, default=1.0, help='of the training set (tests)')
    a = ap.parse_args()
    kind, trainer = STAGES[a.stage]
    model = YOLO(DEFAULT_WEIGHTS)
    data = (f'{HERE}/data/yolo_real/{kind[5:]}.yaml' if kind.startswith('real/')
            else f'{HERE}/data/yolo/{kind}.yaml')
    model.train(trainer=trainer, data=a.data or data,
                epochs=a.epochs, batch=a.batch, workers=a.workers, imgsz=640, device=a.device,
                fliplr=0.0, project=f'{HERE}/runs', name=a.name or a.stage, exist_ok=True,
                seed=0, deterministic=True, plots=True, amp=False, fraction=a.fraction)


if __name__ == '__main__':
    main()
