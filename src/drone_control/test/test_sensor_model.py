# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
import numpy as np

from drone_control import sensor_model as sm


def test_transmission_at_visibility_is_two_percent():
    z = np.full((3, 3), 10.0, np.float32)                 # centre pixel: ray along the axis
    t = sm.transmission(z, np.radians(60), visibility=10.0)
    assert abs(t[1, 1] - 0.02) < 1e-3
    assert np.all(t[[0, 0, 2, 2], [0, 2, 0, 2]] < t[1, 1])  # off-axis rays are longer


def test_fog_rgb_limits():
    rgb = np.zeros((4, 4, 3), np.uint8)
    near = sm.fog_rgb(rgb, np.full((4, 4), 0.01, np.float32), 1.0, 50.0)
    sky = sm.fog_rgb(rgb, np.full((4, 4), np.inf, np.float32), 1.0, 50.0)
    assert near.max() <= 1 and sky.min() >= 216            # J at r ~ 0, airlight for the sky


def test_fog_depth_drops_beyond_visibility():
    z = np.array([[2.0, 30.0]], np.float32)
    out = sm.fog_depth(z, 0.1, visibility=10.0)
    assert out[0, 0] == 2.0 and np.isinf(out[0, 1])


def test_align_depth_same_camera_is_identity():
    z = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.array_equal(sm.align_depth(z, 1.0, (4, 3), 1.0), z)


def test_d435i_noise_grows_with_square_of_distance():
    rng = np.random.default_rng(0)
    err = {}
    for d in (2.0, 8.0):
        z = sm.d435i_depth_noise(np.full((200, 848), d, np.float32), rng=rng)
        err[d] = np.std(z[np.isfinite(z)] - d)
    f = sm.focal_px(848, sm.D435I_DEPTH_HFOV)
    assert abs(err[2.0] - 4 * sm.D435I_SUBPIXEL / (f * sm.D435I_BASELINE)) < 1e-3
    assert 12 < err[8.0] / err[2.0] < 20                   # ~ (8/2)^2 = 16
    assert np.isinf(sm.d435i_depth_noise(np.full((4, 848), 12.0, np.float32), rng=rng)).all()
