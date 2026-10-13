# PX4 + ROS 2 + Gazebo Harmonic + YOLOv11 — SAR Human-in-the-Loop: running it by hand

Manual start of every component, as in the paper *"A ROS2 based Human-In-The-Loop Framework for
UAV Search and Rescue Operations"*, on **Ubuntu 22.04 · ROS 2 Humble · PX4 v1.15.4 · Gazebo
Harmonic**. Installation and the one-command start (`./run_simulation.sh`, which opens these
terminals for you, with options for camera, pipeline, fog and real videos): see `README.md`. Paths below are relative to
the workspace folder (`$ROSWS` after `source px4_env.sh`).

| Path | What |
|---|---|
| `external/PX4-Autopilot` | PX4 v1.15.4 (SITL built) |
| `external/px4-venv` | venv with pycollada for the human-model generator (`px4-venv` alias) |
| `external/QGroundControl-4.4.4.AppImage` | QGroundControl (optional) |
| `src/px4_msgs`, `src/px4_ros_com` | PX4 v1.15 ROS 2 messages + examples |
| `src/Micro-XRCE-DDS-Agent` | uXRCE-DDS agent v2.4.2 (built by colcon) |
| `src/sar_sim` | Forest / City / Wall worlds (+ `_fog`), SAR human models, `x500_d435i` / `RealSense_D435i` |
| `src/drone_control` | Paper's Gesture Recognition node (Unified-PDD) + Control node + sensor simulation (physical fog, D435i noise) |
| `eval/` | Simulation start window, control panel, demo driver (`run_simulation.sh`); evaluation scripts |
| `px4_env.sh` | **source this in every terminal** |

Rebuild after changes: `cd $ROSWS && colcon build && source install/setup.bash`

---

## A. Paper pipeline — gesture-controlled UAV (world: `forest`, `city`, `wall`, `*_fog`)

Every terminal: `source <workspace>/px4_env.sh`

**Terminal 1 — Gazebo world**
```bash
gz sim -r $SAR_WORLDS/city.sdf
```

**Terminal 2 — PX4 SITL (spawns the level-camera x500_depth into the running world)**
```bash
cd $PX4_DIR
PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4002 PX4_SIM_MODEL=gz_x500_depth PX4_GZ_WORLD=city ./build/px4_sitl_default/bin/px4
```
(`PX4_GZ_WORLD` must match the world name: forest, city, wall, forest_fog, city_fog, wall_fog.)

**Terminal 3 — QGroundControl (optional)**
```bash
$ROSWS/external/QGroundControl-4.4.4.AppImage
```

**Terminal 4 — Micro XRCE-DDS Agent**
```bash
MicroXRCEAgent udp4 -p 8888
```

**Terminal 5 — bridge + gesture node + control node (drone takes off and follows gestures)**
```bash
ros2 launch drone_control sar_hitl.launch.py                    # Algorithm 1 of the paper, pose on RGB (default)
ros2 launch drone_control sar_hitl.launch.py input_mode:=depth  # pose on depth pseudo-RGB (forest / fog)
ros2 launch drone_control sar_hitl.launch.py input_mode:=fused  # depth-weighted fused RGB-D pose, Eq. (1)-(2)
ros2 launch drone_control sar_hitl.launch.py input_mode:=rgbd model:=$ROSWS/results/weights/yolov11_ltotal_synthetic.pt  # YOLOv11-L_total
ros2 launch drone_control sar_hitl.launch.py visibility:=20     # physically based fog, visibility 20 m
ros2 launch drone_control sar_hitl.launch.py sensor:=d435i      # RealSense D435i (see below)
ros2 launch drone_control sar_hitl.launch.py control:=false     # perception only, fly with QGC/keyboard
ros2 launch drone_control sar_hitl.launch.py points:=true       # also bridge /depth_camera/points (rviz2)
ros2 launch drone_control sar_hitl.launch.py robust:=true       # + robustness extensions, not in the paper (below)
```
Or node by node:
```bash
ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock /camera@sensor_msgs/msg/Image@gz.msgs.Image /camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo /depth_camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked /depth_camera@sensor_msgs/msg/Image@gz.msgs.Image
ros2 run drone_control gesture_node --ros-args -p refine:=false -p dehaze:=false -p kp_conf:=0.0 \
  -p one_arm:=false -p lateral_check:=false -p nearest_operator:=false   # Algorithm 1 of the paper
ros2 run drone_control drone_control
```

**Terminal 6 — rviz2 (optional)** `rviz2`  (add Image `/gesture/annotated`; PointCloud2 `/depth_camera/points` needs `points:=true`)

Topics: `/gesture/command` (LEFT/RIGHT/NEUTRAL/NONE), `/gesture/distance` (m), `/gesture/annotated`.
Control: the drone **turns toward where the operator points** (they face each other, so the
operator's RIGHT arm → drone yaws LEFT, LEFT arm → yaws RIGHT); NEUTRAL / nobody → hover and hold
heading. It holds its take-off position and altitude while turning. Ctrl+C on the launch lands.
**RealSense D435i** (the camera named in the paper): start PX4 with `PX4_SIM_MODEL=gz_x500_d435i`
(same `PX4_SYS_AUTOSTART=4002`) and launch with `sensor:=d435i`. The model uses the datasheet
FOV / resolution / range; `sensor_sim` adds Intel's stereo depth-noise model (σ ∝ z²).
**Physical fog** (`visibility:=V`, any world, any sensor): `sensor_sim` applies Koschmieder's law
(`I = J·t + A·(1−t)`, `t = exp(−3.912·r/V)`) using depth, and drops depth returns beyond V.

Useful parameters: `takeoff_alt:=1.5 yaw_rate:=15.0 max_turn:=90 threads:=0` (launch),
`device:=cpu|0` and `rate_hz:=10.0` (launch; see GPU/CPU note), `window`/`min_votes` (temporal filter 5 of 7).
The turn rate is timed with PX4's clock, so it is 15°/s of simulation time at any real-time factor.

**Robustness extensions (not part of the paper).** The launch file runs Algorithm 1 of the paper
(`robust:=false`, the default; Algorithm 1 includes `face_camera`, the left / right swap "as seen
by the UAV", and `both_arms_hover`, both arms extended → NEUTRAL). `robust:=true` adds the further
steps below, which help in some scenes; they are on by default when the gesture node is started by
hand with `ros2 run`:

| Parameter | What it does | Why |
|---|---|---|
| `refine` | person found but no pointing arm → run pose again on a magnified crop of that person (then a contrast-enhanced crop) | at 640 px input the arm is only a few pixels thick (Forest, City at 1.9 m) |
| `kp_conf` (0.5) | LEFT/RIGHT only if shoulder, elbow, wrist and hip on that side are located with confidence ≥ 0.5 | phantom "people" YOLO sees in fog (e.g. the City playground) were steering the drone |
| `dehaze` | when nobody is found in a low-contrast frame (`haze_std` 20), retry on a CLAHE contrast-enhanced frame | fog washes out the white figure |
| `imgsz` (0 = auto) | 640 for RGB, 1280 for `input_mode:=depth` | the person is too small in the 640×480 depth image |
| `one_arm` | a command needs exactly one extended arm | two-arm signals (e.g. "hover") read as RIGHT on real UAV-GESTURE joint data |
| `lateral_check` | LEFT/RIGHT from where the extended wrist is (outside the shoulders; image-left = operator's RIGHT) | YOLO sometimes attaches the extended arm to the wrong shoulder, flipping the label |
| `nearest_operator` | only the nearest detected person can command the UAV | "people" YOLO finds in the background (tree trunks) out-voted a NEUTRAL operator |

To change one of them on its own, run the node by hand, e.g.
`ros2 run drone_control gesture_node --ros-args -p refine:=false`.

## B. PX4 ROS 2 guide check — sensor_combined listener
```bash
# with PX4 SITL + MicroXRCEAgent running
ros2 launch px4_ros_com sensor_combined_listener.launch.py
```

---

## Notes

* **Worlds** (`src/sar_sim/scripts/make_worlds.py`, regenerate with `python3 make_worlds.py`) rebuild the
  paper's environment figures from the same Gazebo Fuel assets (Oak tree, Pine Tree, Playground,
  School, Collapsed industrial building). The human stands ~7.5–7.8 m in front of the drone spawn,
  facing it, pointing with the right arm (the paper's measured distances: City 7.13, Forest 9.79, Wall 7.74 m).
* **Human model**: the paper's human was a Blender mesh; `make_human.py` builds an equivalent
  white, pointing human from the rigged Fuel actor (`sar_human_right/left/neutral`).
* **Fog**: Gazebo Harmonic's renderer does not implement `<scene><fog>`; the `_fog` worlds use the
  particle-emitter fog (Fuel "Fog Generator"), which RGB cameras render and depth cameras scatter.
  The emitter `rate` is tuned per world (City/Forest 8, Wall 14) so that, as in the paper's fog test,
  the person is detected at 3.5 m and lost at 7.5–10 m. The fog takes ~20 s of sim time to build up.
  To fly the fog test at 3.5 m, spawn the drone closer, e.g. City:
  `PX4_GZ_MODEL_POSE="4.0,-1.2,0.2,0,0,0"` (Wall `4.3,-1.2,...`, Forest `4.3,0.4,...`) in Terminal 2.
* **Camera**: the paper setup uses the stock level `x500_depth` (OAK-D Lite); `x500_d435i` is the
  RealSense D435i variant (`sensor:=d435i`).
* **GPU / CPU**: with a single GPU (user in group `render`), the GPU is time-shared with Gazebo's
  renderer, so the launch file runs YOLOv11 on the **CPU** by default. With a second or a larger
  GPU, `device:=0` is fine. Use FP32 (default): FP16 is slower and can return NaN keypoints.
* **Simulation speed**: Gazebo + PX4 alone run at RTF 1.0. Streaming the 1920x1080 camera into ROS
  at 30 Hz and running YOLO bring the full launch down to ~0.55–0.75 (varies run to run; the
  point cloud costs more, so it is off by default). This does not change behaviour: PX4 runs in
  lockstep and the control node times its turns with Gazebo's `/clock` (bridged by the launch file).
  Before starting, make sure no old `gz sim` / `px4` / `MicroXRCEAgent` is still running
  (`pgrep -af "gz sim|bin/px4|MicroXRCE"`). Two Gazebo servers on the same world mix their
  clocks and sensor data, so PX4's estimate drifts and it refuses to arm.
* **Height estimate**: with PX4's defaults (GPS as height reference, GPS altitude fused) the
  simulated GPS altitude noise makes the estimated height wander by ~1 m, so the drone's true hover
  height drifted between ~0 and 2 m (and take-off was sometimes refused: "vertical velocity
  unstable"). The barometer is used for height instead, GPS for horizontal position only, and the
  barometer noise setting matches Gazebo's barometer (0.01 Pa ≈ 1 mm; PX4's default of 3.5 m made
  the estimate lag the true height by ~0.5 m). `./run_simulation.sh` and
  `eval/run_flight.sh` set it on every start; when starting PX4 by hand, set it once in the PX4
  shell and restart PX4 (it is then kept in PX4's `parameters.bson`):
  `param set EKF2_HGT_REF 0`, `param set EKF2_GPS_CTRL 5`, `param set EKF2_BARO_NOISE 0.2`. Measured true hover height afterwards: 1.33–1.69 m
  for the 1.5 m setpoint.
