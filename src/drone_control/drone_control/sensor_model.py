# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Sensor models applied on top of the ideal Gazebo images (ROS-free, used live and offline).

Physically based fog -- Koschmieder's law (atmospheric scattering model), as used for synthetic
fog on real images (e.g. Foggy Cityscapes, RESIDE):
    I(x) = J(x) t(x) + A (1 - t(x)),   t(x) = exp(-beta r(x)),   beta = -ln(0.02) / V = 3.912 / V
J: clear image, A: airlight, r: path length along the pixel's ray, V: meteorological visibility
(distance at which contrast falls to 2 %).  Stereo depth needs image contrast, so depth pixels
beyond the visibility (t < 2 %) are returned invalid.

RealSense D435i depth noise -- Intel's stereo RMS error model,
    sigma(z) = z^2 * subpixel / (focal_px * baseline),  baseline 50 mm, subpixel 0.08,
valid range [0.28 m, 10 m] (min-Z at 848x480; beyond 10 m returned invalid).
"""
from functools import lru_cache

import cv2
import numpy as np

KOSCHMIEDER = -np.log(0.02)      # 3.912
FAR_SKY = 1e3                    # path length used for pixels with no depth return (sky)

# Intel RealSense D435i (datasheet): RGB 69 x 42 deg, depth 87 x 58 deg, baseline 50 mm
D435I_RGB_HFOV = np.radians(69.4)
D435I_DEPTH_HFOV = np.radians(87.0)
D435I_BASELINE = 0.050
D435I_SUBPIXEL = 0.08
D435I_RANGE = (0.28, 10.0)


def focal_px(width, hfov):
    return (width / 2) / np.tan(hfov / 2)


@lru_cache(maxsize=8)
def _remap_tables(src_wh, src_hfov, dst_wh, dst_hfov):
    """Pixel maps that sample a co-located pinhole camera (src) at every dst pixel."""
    (sw, sh), (dw, dh) = src_wh, dst_wh
    fs, fd = focal_px(sw, src_hfov), focal_px(dw, dst_hfov)
    # pixel centres: dst pixel i is at (i + 0.5 - dw/2) / fd on the image plane
    u = (np.arange(dw, dtype=np.float32) + 0.5 - dw / 2) * fs / fd + sw / 2 - 0.5
    v = (np.arange(dh, dtype=np.float32) + 0.5 - dh / 2) * fs / fd + sh / 2 - 0.5
    return np.meshgrid(u, v)


@lru_cache(maxsize=8)
def _ray_factor(wh, hfov):
    """r / z for every pixel: path length along the ray per metre of optical-axis depth."""
    w, h = wh
    f = focal_px(w, hfov)
    x = (np.arange(w, dtype=np.float32) + 0.5 - w / 2) / f     # pixel centres
    y = (np.arange(h, dtype=np.float32) + 0.5 - h / 2) / f
    xx, yy = np.meshgrid(x, y)
    return np.sqrt(1 + xx ** 2 + yy ** 2).astype(np.float32)


def align_depth(depth, depth_hfov, dst_wh, dst_hfov):
    """Resample a z-depth image onto another co-located camera's pixel grid (nearest)."""
    d = np.asarray(depth, np.float32)
    mx, my = _remap_tables((d.shape[1], d.shape[0]), float(depth_hfov),
                           tuple(dst_wh), float(dst_hfov))
    return cv2.remap(d, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=float('inf'))


def transmission(z, hfov, visibility):
    """t = exp(-beta r) for a z-depth image; invalid / out-of-range depth = sky (t ~ 0)."""
    z = np.where(np.isfinite(z) & (z > 0), z, FAR_SKY).astype(np.float32)
    r = z * _ray_factor((z.shape[1], z.shape[0]), float(hfov))
    return np.exp(-(KOSCHMIEDER / visibility) * r)


def fog_rgb(rgb, z_rgb, rgb_hfov, visibility, airlight=217.0):
    """Koschmieder fog on a bgr8 image; z_rgb is z-depth aligned to the RGB pixels."""
    t = transmission(z_rgb, rgb_hfov, visibility)[..., None]
    out = rgb.astype(np.float32) * t + airlight * (1 - t)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def fog_depth(depth, depth_hfov, visibility):
    """Stereo depth in fog: no return where contrast is below 2 % (path longer than V)."""
    t = transmission(depth, depth_hfov, visibility)
    out = np.array(depth, np.float32, copy=True)
    out[t < 0.02] = np.inf
    return out


def d435i_depth_noise(depth, depth_hfov=D435I_DEPTH_HFOV, rng=None):
    """Add D435i-like stereo noise (sigma ~ z^2) and its valid-range limits to an ideal z-depth."""
    rng = np.random.default_rng() if rng is None else rng
    true = np.asarray(depth, np.float32)
    f = focal_px(true.shape[1], depth_hfov)
    valid = np.isfinite(true) & (true >= D435I_RANGE[0]) & (true <= D435I_RANGE[1])
    z = np.full(true.shape, np.inf, np.float32)
    sigma = true[valid] ** 2 * D435I_SUBPIXEL / (f * D435I_BASELINE)
    z[valid] = true[valid] + rng.standard_normal(int(valid.sum())).astype(np.float32) * sigma
    return z


def rgb_noise(rgb, sigma=1.8, rng=None):
    """Gaussian sensor noise (8-bit levels) -- same as gz <noise> stddev 0.007 on [0, 1]."""
    rng = np.random.default_rng() if rng is None else rng
    n = rng.standard_normal(rgb.shape).astype(np.float32) * sigma
    return np.clip(rgb.astype(np.float32) + n + 0.5, 0, 255).astype(np.uint8)
