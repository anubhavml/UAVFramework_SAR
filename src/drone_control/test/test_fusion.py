# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
import numpy as np

from drone_control import fusion, pdd
from test_pdd import skeleton

W, H, HFOV = 640, 480, 1.2


def test_weights_eq2():
    s2 = np.array([[0.01], [0.01], [np.nan], [0.04]])
    w = fusion.weights(s2, 100.0)
    assert abs(w.sum() - 1) < 1e-9 and w[2, 0] == 0
    assert abs(w[0, 0] - w[1, 0]) < 1e-12 and w[3, 0] < w[0, 0]
    assert np.allclose(fusion.weights(s2, 0.0)[[0, 1, 3], 0], 1 / 3)      # lambda 0: plain mean
    assert fusion.weights(np.full((4, 1), np.nan), 100.0).sum() == 0


def test_fuse_eq1():
    est = np.full((4, 17, 3), np.nan)
    est[0, 0], est[1, 0] = (0, 0, 4), (1, 0, 4)
    s2 = np.full((4, 17), np.nan)
    s2[0, 0] = s2[1, 0] = 0.01
    p, w = fusion.fuse(est, s2, 50.0)
    assert np.allclose(p[0], (0.5, 0, 4)) and np.isnan(p[1]).all()


def test_camera_round_trip():
    cam = fusion.Camera(W, H, HFOV)
    uv = np.array([[10.0, 20.0], [320.0, 240.0], [600.0, 470.0]])
    assert np.allclose(cam.project(cam.back(uv, np.array([2.0, 5.0, 9.0]))), uv)


class FakePipeline(fusion.FusedPipeline):
    """YOLO replaced by fixed detections: the RGB frame and the depth colormap give the same
    skeleton (rgb_arm / depth_arm), so the fusion and Algorithm 1 run on known keypoints."""

    def __init__(self, rgb_arm='out', depth_arm='out', **kw):
        super().__init__(model=None, rgb_hfov=HFOV, depth_hfov=HFOV, **kw)
        self.arms = {'rgb': rgb_arm, 'depth': depth_arm}

    def detect(self, img):
        src = 'depth' if img.dtype == np.uint8 and img[0, 0, 0] != 50 else 'rgb'
        arm = self.arms[src]
        if arm is None:
            return []
        kp = skeleton(right_arm=arm) + (200, 100)
        return [(np.array([200.0, 100, 340, 310]), 0.9, kp, np.full(17, 0.9), 0)]


def frame():
    return np.full((H, W, 3), 50, np.uint8)


def test_fused_pipeline_right_point():
    pipe = FakePipeline()
    depth = np.full((H, W), 5.0, np.float32)
    _, people, raw, best = pipe.process(frame(), depth)
    assert raw == pdd.RIGHT and len(people) == 1
    assert abs(best.distance - 5.0) < 1e-6
    assert np.allclose(best.weights[[0, 1]].sum(axis=0)[pdd.KP['right_wrist']], 1, atol=0.05)


def test_fused_pipeline_without_depth_and_with_depth_only():
    assert FakePipeline().process(frame(), None)[2] == pdd.RIGHT            # RGB only
    pipe = FakePipeline(rgb_arm=None)                                       # RGB lost (fog)
    assert pipe.process(frame(), np.full((H, W), 5.0, np.float32))[2] == pdd.RIGHT


def test_disagreement_and_neutral():
    pipe = FakePipeline(rgb_arm='down', depth_arm='down')
    assert pipe.process(frame(), np.full((H, W), 5.0, np.float32))[2] == pdd.NEUTRAL


def test_temporal_term_only_after_a_frame():
    pipe = FakePipeline()
    depth = np.full((H, W), 5.0, np.float32)
    pipe.process(frame(), depth)
    assert pipe.last_weights[2].sum() == 0                                  # no history yet
    pipe.process(frame(), depth)
    assert pipe.last_weights[2].sum() > 0                                   # T_j used
    pipe.reset()
    pipe.process(frame(), depth)
    assert pipe.last_weights[2].sum() == 0


def test_temporal_term_follows_the_observations():
    """T_j must not lock onto its own output: with noisy depth (RGB / D uncertain), an arm that
    goes from down to out is read as RIGHT within the temporal window."""
    pipe = FakePipeline(rgb_arm='down', depth_arm='down')
    depth = (5.0 + np.random.default_rng(0).uniform(-0.4, 0.4, (H, W))).astype(np.float32)
    for _ in range(5):
        assert pipe.process(frame(), depth)[2] == pdd.NEUTRAL
    pipe.arms = {'rgb': 'out', 'depth': 'out'}
    assert pdd.RIGHT in [pipe.process(frame(), depth)[2] for _ in range(fusion.WINDOW + 1)]
