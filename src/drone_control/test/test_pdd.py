# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
import numpy as np

from drone_control import pdd


def skeleton(right_arm='down', left_arm='down'):
    """Image-space COCO keypoints of a person facing the camera (their right = image left)."""
    kp = np.zeros((17, 2))
    kp[pdd.KP['right_shoulder']] = (90, 100)
    kp[pdd.KP['left_shoulder']] = (130, 100)
    kp[pdd.KP['right_hip']] = (95, 200)
    kp[pdd.KP['left_hip']] = (125, 200)
    arms = {'down': [(0, 50), (0, 100)], 'out': [(-50, 0), (-100, 0)],
            'bent_up': [(-40, 0), (-40, -40)]}
    for side, pose, sign in (('right', right_arm, 1), ('left', left_arm, -1)):
        sx, sy = kp[pdd.KP[f'{side}_shoulder']]
        (ex, ey), (wx, wy) = arms[pose]
        kp[pdd.KP[f'{side}_elbow']] = (sx + sign * ex, sy + ey)
        kp[pdd.KP[f'{side}_wrist']] = (sx + sign * wx, sy + wy)
    return kp


def test_calculate_angle():
    assert abs(pdd.calculate_angle((1, 0), (0, 0), (0, 1)) - 90) < 1e-6
    assert abs(pdd.calculate_angle((1, 0), (0, 0), (-1, 0)) - 180) < 1e-6


def test_classify_direction():
    assert pdd.classify_direction(pdd.joint_angles(skeleton('out', 'down'))) == pdd.RIGHT
    assert pdd.classify_direction(pdd.joint_angles(skeleton('down', 'out'))) == pdd.LEFT
    assert pdd.classify_direction(pdd.joint_angles(skeleton('down', 'down'))) == pdd.NEUTRAL
    # bent elbow (~90 deg) is outside [100, 180] -> not a pointing gesture
    assert pdd.classify_direction(pdd.joint_angles(skeleton('bent_up', 'down'))) == pdd.NEUTRAL


def test_preprocess_depth_and_distance():
    d = np.full((48, 64), 5.0, np.float32)
    d[0, 0], d[0, 1] = np.nan, np.inf
    m, col = pdd.preprocess_depth(d)
    assert m[0, 0] == pdd.FAR and m[0, 1] == pdd.FAR and col.shape == (48, 64, 3)
    assert abs(pdd.median_distance(m, (0, 0, 64, 48)) - 5.0) < 1e-6
    assert pdd.median_distance(np.full((10, 10), pdd.FAR), (0, 0, 10, 10)) == pdd.FAR


def test_map_box_centre_is_preserved():
    x1, y1, x2, y2 = pdd.map_box((950, 530, 970, 550), (1920, 1080), 1.204, (640, 480), 1.274)
    assert abs((x1 + x2) / 2 - 320) < 1e-6 and abs((y1 + y2) / 2 - 240) < 1e-6


def test_select_command_prefers_confident_pointing():
    p = lambda d, c: pdd.Person((0, 0, 1, 1), c, None, {}, d, 5.0)  # noqa: E731
    assert pdd.select_command([])[0] == pdd.NONE
    assert pdd.select_command([p(pdd.NEUTRAL, 0.8), p(pdd.RIGHT, 0.5)])[0] == pdd.RIGHT
    assert pdd.select_command([p(pdd.NEUTRAL, 0.8)])[0] == pdd.NEUTRAL


def test_temporal_filter():
    f = pdd.TemporalFilter(window=5, min_votes=3)
    assert f.update(pdd.RIGHT) == pdd.NEUTRAL
    assert f.update(pdd.RIGHT) == pdd.NEUTRAL
    assert f.update(pdd.RIGHT) == pdd.RIGHT
    assert f.update(pdd.LEFT) == pdd.RIGHT        # one outlier does not flip the command


def test_torso_box_and_fallback():
    kp = skeleton('out', 'down')
    x1, y1, x2, y2 = pdd.torso_box(kp, (0, 0, 300, 300))
    assert (x1, y1, x2, y2) == (90, 100, 130, 200)
    kp[pdd.KP['left_hip']] = (0, 0)                      # missing keypoint -> full box
    assert pdd.torso_box(kp, (0, 0, 300, 300)) == (0, 0, 300, 300)


def back_view(kp):
    """Same image, but labelled as YOLO does when it assumes the person faces away."""
    kp = kp.copy()
    for a, b in pdd.LR_PAIRS:
        kp[[a, b]] = kp[[b, a]]
    return kp


def test_face_camera_fixes_back_view_swap():
    kp = skeleton('out', 'down')                                # operator points with right arm
    swapped = back_view(kp)
    assert pdd.classify_direction(pdd.joint_angles(swapped)) == pdd.LEFT   # the bug
    fixed, kc = pdd.face_camera(swapped, np.arange(17.0))
    assert pdd.classify_direction(pdd.joint_angles(fixed)) == pdd.RIGHT
    assert kc[pdd.KP['right_wrist']] == pdd.KP['left_wrist']    # confidences follow the swap
    same, _ = pdd.face_camera(kp)                               # front view: unchanged
    assert np.array_equal(same, kp)


def test_face_camera_ignores_missing_shoulder():
    kp = back_view(skeleton('out', 'down'))
    kp[pdd.KP['left_shoulder']] = 0
    assert np.array_equal(pdd.face_camera(kp)[0], kp)


def test_side_confidence():
    kc = np.ones(17)
    kc[pdd.KP['right_wrist']] = 0.2
    assert pdd.side_confidence(kc, pdd.RIGHT) == 0.2
    assert pdd.side_confidence(kc, pdd.LEFT) == 1.0


def test_crop_around_is_square_and_clipped():
    x1, y1, x2, y2 = pdd.crop_around((900, 600, 1000, 900), (1080, 1920, 3))
    assert (x2 - x1, y2 - y1) == (420, 420)                     # 1.4 x the 300 px height
    assert pdd.crop_around((0, 900, 50, 1080), (1080, 1920, 3))[0] == 0
    assert pdd.crop_around((0, 900, 50, 1080), (1080, 1920, 3))[3] == 1080


def test_is_hazy_and_enhance_contrast():
    rng = np.random.default_rng(0)
    clear = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    fog = np.clip(200 + (clear.astype(int) - 128) // 20, 0, 255).astype(np.uint8)
    assert not pdd.is_hazy(clear) and pdd.is_hazy(fog)
    assert pdd.enhance_contrast(fog).std() > fog.std()


def test_one_arm_rule():
    both = pdd.joint_angles(skeleton('out', 'out'))
    assert pdd.classify_direction(both) == pdd.RIGHT                 # Algorithm 1: right first
    assert pdd.classify_direction(both, one_arm=True) == pdd.NEUTRAL  # two arms: not a command
    one = pdd.joint_angles(skeleton('down', 'out'))
    assert pdd.classify_direction(one, one_arm=True) == pdd.LEFT


def test_lateral_direction_fixes_crossed_arm():
    kp = skeleton('out', 'down')                                 # right arm out to image-left
    assert pdd.lateral_direction(kp, pdd.RIGHT) == pdd.RIGHT
    crossed = kp.copy()                                          # YOLO puts the extended arm on
    for j in ('elbow', 'wrist'):                                 # the left-shoulder labels
        a, b = pdd.KP[f'left_{j}'], pdd.KP[f'right_{j}']
        crossed[[a, b]] = crossed[[b, a]]
    assert pdd.classify_direction(pdd.joint_angles(crossed)) == pdd.LEFT   # the failure
    assert pdd.lateral_direction(crossed, pdd.LEFT) == pdd.RIGHT
    front = kp.copy()
    front[pdd.KP['right_wrist']] = (110, 120)                    # wrist between the shoulders
    assert pdd.lateral_direction(front, pdd.RIGHT) == pdd.NEUTRAL


def person(direction, conf, dist, box=(0, 0, 10, 40)):
    return pdd.Person(box, conf, None, {}, direction, dist)


def test_nearest_operator_ignores_background_people():
    operator = person(pdd.NEUTRAL, 0.67, 6.9)
    phantom = person(pdd.RIGHT, 0.71, 9.4)                       # tree trunk 2.5 m behind
    assert pdd.select_command([operator, phantom])[0] == pdd.RIGHT
    assert pdd.select_command([operator, phantom], nearest=True)[0] == pdd.NEUTRAL
    duplicate = person(pdd.RIGHT, 0.5, 7.2)                      # same person, wider box
    assert pdd.select_command([operator, duplicate, phantom], nearest=True)[0] == pdd.RIGHT
    far = pdd.FAR                                                # no depth: tallest box wins
    tall = person(pdd.NEUTRAL, 0.6, far, (0, 0, 10, 80))
    small = person(pdd.LEFT, 0.9, far, (50, 0, 60, 40))          # elsewhere in the image
    assert pdd.select_command([tall, small], nearest=True)[0] == pdd.NEUTRAL
