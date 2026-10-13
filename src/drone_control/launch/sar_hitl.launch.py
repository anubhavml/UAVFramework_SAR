# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""ROS 2 side of the SAR HITL pipeline: Gazebo<->ROS bridge + gesture node + control node
(+ sensor_sim for physically based fog / RealSense D435i noise).

Start Gazebo (world), PX4 SITL and MicroXRCEAgent first (see RUN_SAR.md in the workspace), then:
    ros2 launch drone_control sar_hitl.launch.py                    # Algorithm 1 of the paper, RGB, fly on gestures
    ros2 launch drone_control sar_hitl.launch.py input_mode:=depth  # pose on depth pseudo-RGB
    ros2 launch drone_control sar_hitl.launch.py input_mode:=fused  # RGB + depth pose fused, Eq. (1)-(2)
    ros2 launch drone_control sar_hitl.launch.py control:=false     # perception only
    ros2 launch drone_control sar_hitl.launch.py points:=true       # + point cloud for rviz2
    ros2 launch drone_control sar_hitl.launch.py robust:=true       # + robustness extensions (not in the paper)
    ros2 launch drone_control sar_hitl.launch.py sensor:=d435i      # PX4_SIM_MODEL=gz_x500_d435i
    ros2 launch drone_control sar_hitl.launch.py visibility:=10     # physically based fog, V = 10 m
    ros2 launch drone_control sar_hitl.launch.py input_mode:=rgbd \
        model:=$ROSWS/results/weights/yolov11_ltotal_synthetic.pt   # YOLOv11-L_total, RGB-D
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

CLOCK = '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'       # sim time for drone_control
CAMERAS = ['/camera@sensor_msgs/msg/Image@gz.msgs.Image',
           '/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
           '/depth_camera@sensor_msgs/msg/Image@gz.msgs.Image']
POINTS = '/depth_camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked'
GT_DEPTH = '/camera_gt_depth@sensor_msgs/msg/Image@gz.msgs.Image'
# horizontal FOV (rad) of the RGB / depth cameras on each simulated vehicle
SENSORS = {'oakd': dict(rgb_hfov=1.204, depth_hfov=1.274),      # x500_depth (OakD-Lite)
           'd435i': dict(rgb_hfov=1.2112, depth_hfov=1.5184),   # x500_d435i (RealSense D435i)
           # real video (eval/demo_flight.py --source real): depth estimated from the same frame
           'video': dict(rgb_hfov=1.204, depth_hfov=1.204)}


def truthy(s):
    return s.lower() in ('1', 'true', 'yes')


def setup(context):
    a = context.launch_configurations                # argument name -> string value
    sensor, robust = a['sensor'], truthy(a['robust'])
    if sensor not in SENSORS:
        raise RuntimeError(f"sensor must be one of {list(SENSORS)}, got '{sensor}'")
    fov = SENSORS[sensor]
    vis = float(a['visibility'])
    video = sensor == 'video'                       # camera = real video frames, not Gazebo
    # sensor_sim: fog and/or D435i depth noise on the Gazebo images (real video: the demo driver
    # applies the fog itself, eval/demo_flight.py)
    sim = (vis > 0 or sensor == 'd435i') and not video
    bridge = [CLOCK] + CAMERAS + ([POINTS] if truthy(a['points']) else []) + \
             ([GT_DEPTH] if sensor == 'd435i' and vis > 0 else [])
    nodes = [Node(package='ros_gz_bridge', executable='parameter_bridge', name='gz_bridge',
                  arguments=bridge, output='screen')]
    if sim:
        nodes.append(Node(package='drone_control', executable='sensor_sim', name='sensor_sim',
                          parameters=[{'visibility': vis, 'd435i_noise': sensor == 'd435i',
                                       'gt_depth_hfov': fov['rgb_hfov'], **fov}],
                          output='screen'))
    nodes.append(Node(
        package='drone_control', executable='gesture_node', name='gesture_node',
        parameters=[{'input_mode': a['input_mode'], 'show': truthy(a['show']),
                     **({'model': a['model']} if a['model'] else {}),
                     'device': a['device'], 'rate_hz': float(a['rate_hz']),
                     'rgb_topic': '/video/camera' if video else '/sim/camera' if sim else '/camera',
                     'depth_topic': '/video/depth_camera' if video else '/sim/depth_camera' if sim else '/depth_camera',
                     # face_camera = Algorithm 1: left / right as seen by the UAV (uav_view), also with robust:=false
                     'refine': robust, 'face_camera': robust or truthy(a['uav_view']), 'dehaze': robust,
                     'one_arm': robust, 'lateral_check': robust, 'nearest_operator': robust,
                     'kp_conf': 0.5 if robust else 0.0,
                     'both_arms_hover': truthy(a['both_arms_hover']),
                     'window_width': int(a['window_width']),
                     'threads': int(a['threads']), **fov}],
        output='screen'))
    if truthy(a['control']):
        nodes.append(Node(package='drone_control', executable='drone_control',
                          name='drone_control',
                          parameters=[{'takeoff_alt': float(a['takeoff_alt']),
                                       'yaw_rate': float(a['yaw_rate']),
                                       'max_turn': float(a['max_turn']),
                                       'react': a['react'],
                                       'fly_turn': float(a['fly_turn']),
                                       'fly_distance': float(a['fly_distance'])}],
                          output='screen'))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('input_mode', default_value='rgb',
                              description='rgb | depth | rgbd (trained RGB-D model) | fused (Eq. (1)-(2) pose fusion)'),
        DeclareLaunchArgument('model', default_value='',
                              description='YOLO weights; empty = stock yolo11n-pose. A trained '
                                          'gesture model (train/runs/<stage>/weights/best.pt) '
                                          'gives the direction by its class; robust is then '
                                          'ignored'),
        DeclareLaunchArgument('show', default_value='true', description='open an OpenCV window'),
        DeclareLaunchArgument('window_width', default_value='0',
                              description='show: camera window width in px (0 = image size)'),
        DeclareLaunchArgument('control', default_value='true', description='start drone_control'),
        # On a single small GPU time-shared with Gazebo's renderer, CUDA inference can stall the
        # simulation. YOLOv11n-pose on CPU is fast enough for the 10 Hz gesture rate, so CPU is the
        # default; with a second GPU or a larger one, device:=0 is fine.
        DeclareLaunchArgument('device', default_value='cpu', description="cpu | 0 (GPU) | '' auto"),
        DeclareLaunchArgument('rate_hz', default_value='10.0', description='gesture inference rate'),
        DeclareLaunchArgument('threads', default_value='0',
                              description='CPU inference threads (0 = Ultralytics default, 8)'),
        DeclareLaunchArgument('takeoff_alt', default_value='1.5',
                              description='climb (m); camera ends ~1.9 m up, chest height at 3.5 m'),
        DeclareLaunchArgument('yaw_rate', default_value='15.0', description='turn rate (deg/s)'),
        DeclareLaunchArgument('max_turn', default_value='90.0',
                              description='max turn either side of the take-off heading (deg)'),
        DeclareLaunchArgument('react', default_value='turn',
                              description='turn = yaw towards the pointed side while the gesture '
                                          'is held (paper); fly = turn fly_turn deg, fly '
                                          'fly_distance m there, hover, return, face the operator'),
        DeclareLaunchArgument('fly_turn', default_value='90.0', description='react:=fly turn (deg)'),
        DeclareLaunchArgument('fly_distance', default_value='5.0', description='react:=fly distance (m)'),
        DeclareLaunchArgument('robust', default_value='false',
                              description='false = Algorithm 1 of the paper; true = adds robustness '
                                          'extensions that are not in the paper (refine, '
                                          'face_camera, kp_conf, dehaze, one_arm, lateral_check, '
                                          'nearest_operator)'),
        DeclareLaunchArgument('uav_view', default_value='true',
                              description='Algorithm 1 line 13b: left/right read as the UAV sees them '
                                          '(swap left/right keypoints when the left shoulder is on the '
                                          "image's left); false = without line 13b"),
        DeclareLaunchArgument('both_arms_hover', default_value='true',
                              description='gesture rules: one arm stretched out -> turn; both arms '
                                          'stretched out or anything else -> hover; '
                                          'false = without this rule'),
        DeclareLaunchArgument('sensor', default_value='oakd',
                              description='oakd (PX4_SIM_MODEL=gz_x500_depth) | d435i '
                                          '(PX4_SIM_MODEL=gz_x500_d435i) | video (camera = '
                                          '/video/camera + /video/depth_camera, real frames)'),
        DeclareLaunchArgument('visibility', default_value='0',
                              description='physically based fog: meteorological visibility (m), '
                                          '0 = off'),
        # The gesture node does not use the point cloud; bridging it costs simulation speed.
        DeclareLaunchArgument('points', default_value='false',
                              description='also bridge /depth_camera/points (for rviz2)'),
        OpaqueFunction(function=setup),
    ])
