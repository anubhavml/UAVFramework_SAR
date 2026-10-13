# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Stage T4: the T3
network (RGB-D, depth-aware FPN, cross-scale attention) trained with the two extra terms of the
paper's Eq. 3:

  L_depth     a distance branch on the head's input features predicts, per anchor, the
              person's distance (m); L1 against the ground-truth distance of the person the
              Ultralytics assigner matches to that anchor.  The distances travel through the
              augmentations as a second column of the labels' "cls" (T4Dataset).
  L_temporal  every training step also runs CLIPS_PER_STEP clips x 4 consecutive frames
              (clips_train.npy next to the dataset folder: uint8 (clips, 4, 384, 640, 4)
              B, G, R, D frames, 1280x720 letterboxed to 640x384, no other augmentation;
              clips_train_boxes.npy: float32 (clips, 4, 4) person box cx, cy, w, h in pixels);
              per frame the person's gesture logits = mean class logits of the anchors whose
              centres lie in the central half of the labelled box; the term is the mean squared
              deviation of those logits from the clip mean.
Both weighted 1.0 and scaled by the batch size like Ultralytics' own terms.  At inference the
network is used like T3 (the distance branch output is only added to the raw head dict).
"""
import os

import numpy as np
import torch
import torch.nn as nn
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.modules import Conv
from ultralytics.utils.loss import v8PoseLoss
from ultralytics.utils.tal import make_anchors

from drone_control.rgbd_model import RGBDAttnPoseModel, RGBDAttnPoseTrainer

CLIPS_PER_STEP = 2
W_DEPTH, W_TEMPORAL = 1.0, 1.0
_STATE = {}          # training model and clip arrays (kept out of the model so checkpoints stay small)


class T4PoseModel(RGBDAttnPoseModel):
    def __init__(self, cfg='yolo11n-pose.yaml', ch=4, nc=None, data_kpt_shape=(None, None),
                 verbose=True):
        super().__init__(cfg, ch=ch, nc=nc, data_kpt_shape=data_kpt_shape, verbose=verbose)
        self.model.eval()                # probe pass: leave BatchNorm statistics untouched
        with torch.no_grad():
            neck = self._channels()[1]
        self.model.train()
        c = max(neck[0] // 4, 16)
        self.dist_head = nn.ModuleList(
            nn.Sequential(Conv(x, c, 3), Conv(c, c, 3), nn.Conv2d(c, 1, 1)) for x in neck)
        for h in self.dist_head:
            nn.init.zeros_(h[-1].weight)
            nn.init.constant_(h[-1].bias, 5.0)            # start at 5 m, mid of the 0.8-12 m range

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        feats = None
        if x.shape[1] == 4:
            feats = self.depth(x[:, 3:4])
            x = x[:, [2, 1, 0]]
        y, dist = [], None
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if m is self.model[-1]:          # (the stride probe in __init__ runs before attn exists)
                if getattr(self, 'attn', None) is not None:
                    x = self.attn(x)
                if getattr(self, 'dist_head', None) is not None:
                    dist = torch.cat([h(xi).flatten(2) for h, xi in zip(self.dist_head, x)], 2)[:, 0]
            x = m(x)
            if feats is not None and m.i in (4, 6, 10):
                x = x + feats[(4, 6, 10).index(m.i)]
            y.append(x if m.i in self.save or m.i in (4, 6, 10) else None)
        if isinstance(x, dict):
            x['dist'] = dist
        elif isinstance(x, tuple) and isinstance(x[1], dict):
            x[1]['dist'] = dist
        return x

    def init_criterion(self):
        return T4PoseLoss(self)


class T4PoseLoss(v8PoseLoss):
    def get_assigned_targets_and_loss(self, preds, batch):
        out = super().get_assigned_targets_and_loss(preds, batch)
        self._assigned = out[0]
        return out

    def loss(self, preds, batch):
        loss, items = super().loss(preds, batch)          # (box, pose, kobj, cls, dfl) x batch
        bs = preds['boxes'].shape[0]
        zero = torch.zeros((), device=self.device)
        l_depth = l_temp = zero
        if 'dist' in batch and preds.get('dist') is not None:
            fg, gidx = self._assigned[0], self._assigned[1]
            if fg.any():
                bi = batch['batch_idx'].view(-1).long().to(self.device)
                gd = batch['dist'].view(-1).float().to(self.device)
                pad = torch.zeros(bs, max(int(bi.bincount(minlength=bs).max()), 1), device=self.device)
                for j in range(bs):
                    v = gd[bi == j]
                    pad[j, :len(v)] = v
                target = pad.gather(1, gidx.clamp(max=pad.shape[1] - 1))
                use = fg & torch.isfinite(target)          # real images: no ground-truth distance (NaN)
                if use.any():
                    l_depth = (preds['dist'][use] - target[use]).abs().mean()
        model = _STATE.get('model')
        if torch.is_grad_enabled() and model is not None and model.training and 'clips' in _STATE:
            l_temp = temporal_loss(model, self.stride)
        extra = torch.stack([l_depth * W_DEPTH, l_temp * W_TEMPORAL]) * bs
        return torch.cat([loss, extra]), torch.cat([items, extra.detach() / bs])


def temporal_loss(model, stride):
    """The clip pass must not update the BatchNorm running statistics used at inference (in
    training mode it would, and the inference-time statistics would drift towards the clips)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momentum = [b.momentum for b in bns]
    for b in bns:
        b.momentum = 0.0
    try:
        return _temporal_loss(model, stride)
    finally:
        for b, m in zip(bns, momentum):
            b.momentum = m


def _temporal_loss(model, stride):
    clips, boxes = _STATE['clips'], _STATE['boxes']
    idx = _STATE['rng'].choice(len(clips), CLIPS_PER_STEP, replace=False)
    dev = next(model.parameters()).device
    x = torch.from_numpy(np.ascontiguousarray(clips[idx])).to(dev)        # (C, F, H, W, 4) uint8
    c, f = x.shape[:2]
    x = x.view(c * f, *x.shape[2:]).permute(0, 3, 1, 2).float() / 255
    preds = model(x)
    preds = preds[1] if isinstance(preds, tuple) else preds
    anchors, strides = make_anchors(preds['feats'], stride, 0.5)
    pts = anchors * strides                                                # (A, 2) pixels
    b = torch.from_numpy(boxes[idx].reshape(c * f, 4)).to(dev)              # cx, cy, w, h pixels
    inside = ((pts[None, :, 0] - b[:, None, 0]).abs() < b[:, None, 2] / 4) & \
             ((pts[None, :, 1] - b[:, None, 1]).abs() < b[:, None, 3] / 4)
    near = ((pts[None] - b[:, None, :2]) ** 2).sum(-1).argmin(1)            # tiny boxes
    inside[torch.arange(c * f, device=dev), near] = True
    scores = preds['scores']                                               # (N, nc, A) logits
    logit = (scores * inside[:, None]).sum(2) / inside.sum(1, keepdim=True)  # (N, nc)
    logit = logit.view(c, f, -1)
    return ((logit - logit.mean(1, keepdim=True)) ** 2).mean()


class T4Dataset(YOLODataset):
    """Training dataset whose labels carry the person distance as a second 'cls' column."""

    def get_labels(self):
        labels = super().get_labels()
        for lb in labels:        # <dataset>/images/<split>/<stem>.tiff -> <dataset>/dist/<split>/<stem>.txt
            root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(lb['im_file']))), 'dist')
            split = os.path.basename(os.path.dirname(lb['im_file']))
            stem = os.path.splitext(os.path.basename(lb['im_file']))[0]
            n = len(lb['cls'])
            d = np.loadtxt(f'{root}/{split}/{stem}.txt', ndmin=1).astype(np.float32) if n else \
                np.zeros(0, np.float32)
            assert len(d) == n, f'{stem}: {n} labels, {len(d)} distances'
            lb['cls'] = np.concatenate([lb['cls'].reshape(-1, 1), d.reshape(-1, 1)], 1)
        return labels

    @staticmethod
    def collate_fn(batch):
        for b in batch:                    # images without people: Format gives cls (0, 1)
            if b['cls'].shape[0] == 0 and b['cls'].shape[-1] != 2:
                b['cls'] = torch.zeros(0, 2)
        return YOLODataset.collate_fn(batch)


class T4PoseTrainer(RGBDAttnPoseTrainer):
    model_cls = T4PoseModel

    def build_dataset(self, img_path, mode='train', batch=None):
        if mode != 'train':
            return super().build_dataset(img_path, mode, batch)
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)
        return T4Dataset(img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=True,
                         hyp=self.args, rect=False, cache=self.args.cache or None,
                         single_cls=self.args.single_cls or False, stride=gs, pad=0.0,
                         prefix='train: ', task=self.args.task, classes=self.args.classes,
                         data=self.data, fraction=self.args.fraction)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if batch['cls'].shape[-1] == 2:
            batch['dist'] = batch['cls'][:, 1:2]
            batch['cls'] = batch['cls'][:, :1]
        return batch

    def plot_training_labels(self):
        for lb in self.train_loader.dataset.labels:
            lb['cls_full'] = lb['cls']
            lb['cls'] = lb['cls'][:, :1]
        try:
            super().plot_training_labels()
        finally:
            for lb in self.train_loader.dataset.labels:
                lb['cls'] = lb.pop('cls_full')

    def get_validator(self):
        v = super().get_validator()
        self.loss_names = (*self.loss_names, 'depth_loss', 'temp_loss')
        return v

    def _setup_train(self, *a, **kw):
        super()._setup_train(*a, **kw)
        d = os.path.join(os.path.dirname(str(self.data['path']).rstrip('/')), 'clips_train')
        _STATE['clips'] = np.load(f'{d}.npy', mmap_mode='r')
        _STATE['boxes'] = np.load(f'{d}_boxes.npy')
        _STATE['rng'] = np.random.default_rng(self.args.seed)
        _STATE['model'] = self.model
        print(f'T4: {len(_STATE["clips"])} training clips for the temporal loss', flush=True)
