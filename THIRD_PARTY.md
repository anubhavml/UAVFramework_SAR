# Third-party content in this repository

The code of this repository is licensed under AGPL-3.0-only (`LICENSE`). The files below are
copies or adaptations of third-party content and keep their own licences.

| Files | Source | Author | Licence |
|---|---|---|---|
| `src/sar_sim/models/oak_tree` | Gazebo Fuel, [OpenRobotics/Oak tree](https://app.gazebosim.org/OpenRobotics/fuel/models/Oak%20tree) | Louise Poubel (Open Robotics) | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `src/sar_sim/models/pine_tree` | Gazebo Fuel, [OpenRobotics/Pine Tree](https://app.gazebosim.org/OpenRobotics/fuel/models/Pine%20Tree) | Louise Poubel (Open Robotics) | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `src/sar_sim/models/playground` | Gazebo Fuel, [OpenRobotics/Playground](https://app.gazebosim.org/OpenRobotics/fuel/models/Playground) | Nate Koenig (Open Robotics) | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `src/sar_sim/models/school` | Gazebo Fuel, [OpenRobotics/School](https://app.gazebosim.org/OpenRobotics/fuel/models/School) | Nate Koenig (Open Robotics) | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `src/sar_sim/models/collapsed_industrial` | Gazebo Fuel, [OpenRobotics/Collapsed industrial building](https://app.gazebosim.org/OpenRobotics/fuel/models/Collapsed%20industrial%20building) | Nate Koenig (Open Robotics) | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `src/sar_sim/models/fog_generator` | Gazebo Fuel, [OpenRobotics/Fog Generator](https://app.gazebosim.org/OpenRobotics/fuel/models/Fog%20Generator) | Nate Koenig (Open Robotics) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) |
| `src/sar_sim/models/sar_human_*/meshes/human.obj` | posed by `src/sar_sim/scripts/make_human.py` from Gazebo Fuel, [Mingfei/actor](https://app.gazebosim.org/Mingfei/fuel/models/actor) (downloaded by the script, not included) | Mingfei | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/); the posing is a modification |
| `src/sar_sim/models/x500_d435i` | adapted from the `x500_depth` model of [PX4-Autopilot](https://github.com/PX4/PX4-Autopilot) (`Tools/simulation/gz/models`) | PX4 Development Team | [BSD-3-Clause](https://github.com/PX4/PX4-Autopilot/blob/main/LICENSE) |
| `media/real_dataset_demo.mp4`, `.jpg` | excerpts of the [UAV-GESTURE dataset](https://asankagp.github.io/uavgesture/) (A. G. Perera, Y. W. Law, J. Chahl, "UAV-GESTURE: A dataset for UAV control and gesture recognition", ECCV Workshops, 2018), with the output of this framework drawn on them | the dataset authors | dataset terms: academic research; shown for illustration only, the dataset itself is not included |

Generated training humans (`train/models/`, made by `train/make_humans.py`, not included) are also
posed from the Fuel actor and fall under the same CC-BY-4.0 attribution.

Downloaded at installation, not included: PX4-Autopilot, px4_msgs, px4_ros_com (BSD-3-Clause),
Micro XRCE-DDS Agent (Apache-2.0), Ultralytics YOLO11 and the YOLO11n-pose weights (AGPL-3.0),
Depth-Anything-V2-Metric-Outdoor-Small (see its Hugging Face model card), and optionally
QGroundControl (Apache-2.0 / GPL-3.0) and VirtualGL.
