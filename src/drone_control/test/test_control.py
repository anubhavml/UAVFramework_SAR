# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
import math

from drone_control.control_node import wrap


def test_wrap():
    assert abs(wrap(0.0)) < 1e-12
    assert abs(wrap(2 * math.pi + 0.1) - 0.1) < 1e-9
    assert abs(wrap(-math.pi - 0.1) - (math.pi - 0.1)) < 1e-9
    assert abs(wrap(math.radians(350) - math.radians(10)) - math.radians(-20)) < 1e-9
