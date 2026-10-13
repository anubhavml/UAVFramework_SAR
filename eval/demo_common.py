# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Shared by the desktop demo (demo_flight.py) and the gesture videos: the gestures the person can
show, the UAV-GESTURE clips used for each of them, and drawing helpers."""
import glob
import os
import re

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
# UAV-GESTURE dataset (videos + joint_positions); default <workspace>/UAVGesture
DATA = os.environ.get('UAVGESTURE_DIR') or os.path.join(HERE, '..', 'UAVGesture')
# gesture -> (file name, what the person does, Algorithm 1 label expected, what the UAV should do)
GESTURES = {
    'right': ('01_right_arm_out__uav_turns_left', 'Right arm stretched out, left arm down', 'RIGHT',
              'turn LEFT (towards the pointed side)'),
    'left': ('02_left_arm_out__uav_turns_right', 'Left arm stretched out, right arm down', 'LEFT',
             'turn RIGHT (towards the pointed side)'),
    'neutral': ('03_arms_down__uav_hovers', 'Both arms down', 'NEUTRAL', 'hover'),
    'both_out': ('04_both_arms_out__uav_hovers', 'Both arms stretched out', 'NEUTRAL', 'hover'),
    'both_up': ('05_both_arms_up__uav_hovers', 'Both arms raised overhead', 'NEUTRAL', 'hover'),
    'rightout_leftup': ('06_right_arm_out_left_arm_up__uav_turns_left',
                        'Right arm stretched out, left arm raised', 'RIGHT', 'turn LEFT (towards the pointed side)'),
    'leftout_rightup': ('07_left_arm_out_right_arm_up__uav_turns_right',
                        'Left arm stretched out, right arm raised', 'LEFT', 'turn RIGHT (towards the pointed side)'),
}
# real clips per gesture: (UAV-GESTURE video, first frame, mirrored). Right / left: an arm signal in
# which the arm stays inside the range of Algorithm 1 (stretched out and lowered about once a
# second); left = a right-arm video mirrored (the dataset has no left-arm-only pointing)
REAL = {'right': ('S6_allClear_HD', 5, False), 'left': ('S6_notClear_HD', 5, True),
        'neutral': ('S6_land_HD', 0, False), 'both_out': ('S11_hover_HD', 0, False),
        'both_up': ('S12_waveOff_HD', 90, False), 'rightout_leftup': ('S6_moveToLeft_HD', 103, False),
        'leftout_rightup': ('S6_moveToRight_HD', 1, False)}
CLIP = 125                    # frames per clip (5 s at 25 fps)
GREEN, RED, CYAN, GREY, WHITE = (120, 230, 120), (110, 110, 255), (255, 220, 120), (190, 190, 190), (255, 255, 255)


def put(img, s, org, scale, colour, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def ok_colour(got, want):
    return GREEN if got == want else RED


def videos():
    out = []
    for v in sorted(glob.glob(f'{DATA}/*/*.mp4')):
        stem = os.path.basename(v)[:-4]
        m = glob.glob(f'{DATA}/joint_positions/*/{stem}/{stem}.mat')
        out.append(dict(video=stem, path=v, mat=m[0], cls=os.path.basename(os.path.dirname(v)),
                        subject=int(re.match(r'S(\d+)_', stem).group(1))))
    return out
