# Gesture-switch flights

Person 5 m in front of the hovering drone; gesture switched at known simulation times. Latencies: simulation time (wall-clock time in brackets). Height: true mean height of the drone during the phase (Gazebo; take-off setpoint 1.5 m). See eval/flight_report.py. yolov11_ltotal: YOLOv11-L_total (synthetic-trained, RGB-D), Section 7.4. A command is sent when a gesture wins 5 of the last 7 frames (10 Hz).

| Run | Switch | Command latency | Motion latency | Turn rate | Turned | Wrong cmds | Height |
|---|---|---|---|---|---|---|---|
| city_yolov11_ltotal | neutral->right | 0.64 s (0.65 s) | 1.17 s (1.18 s) | 10.7 °/s | -19° | 0% | 1.58 m |
| city_yolov11_ltotal | right->left | 0.55 s (0.54 s) | 1.25 s (1.25 s) | 15.1 °/s | 38° | 2% | 1.56 m |
| city_yolov11_ltotal | left->neutral | 0.34 s (0.35 s) | – (–) | – | – | 4% | 1.59 m |
| city_fog_yolov11_ltotal | neutral->right | 0.60 s (0.59 s) | 1.12 s (1.12 s) | 14.0 °/s | -19° | 0% | 1.52 m |
| city_fog_yolov11_ltotal | right->left | 0.58 s (0.58 s) | 1.13 s (1.13 s) | 14.2 °/s | 39° | 2% | 1.49 m |
| city_fog_yolov11_ltotal | left->neutral | 0.39 s (0.42 s) | – (–) | – | – | 4% | 1.46 m |
| forest_yolov11_ltotal | neutral->right | 0.59 s (0.63 s) | 1.15 s (1.19 s) | 11.0 °/s | -20° | 0% | 1.51 m |
| forest_yolov11_ltotal | right->left | 0.59 s (0.59 s) | 1.17 s (1.17 s) | 14.7 °/s | 39° | 2% | 1.49 m |
| forest_yolov11_ltotal | left->neutral | 0.37 s (0.39 s) | – (–) | – | – | 4% | 1.57 m |
| forest_fog_yolov11_ltotal | neutral->right | 0.64 s (0.64 s) | 1.15 s (1.15 s) | 12.1 °/s | -20° | 0% | 1.43 m |
| forest_fog_yolov11_ltotal | right->left | 0.55 s (0.58 s) | 1.14 s (1.16 s) | 14.6 °/s | 38° | 2% | 1.38 m |
| forest_fog_yolov11_ltotal | left->neutral | 0.33 s (0.33 s) | – (–) | – | – | 4% | 1.40 m |
| wall_yolov11_ltotal | neutral->right | 0.64 s (0.66 s) | 1.18 s (1.19 s) | 10.2 °/s | -20° | 0% | 1.29 m |
| wall_yolov11_ltotal | right->left | 0.50 s (0.49 s) | 1.03 s (1.03 s) | 15.0 °/s | 39° | 1% | 1.37 m |
| wall_yolov11_ltotal | left->neutral | 0.54 s (0.56 s) | – (–) | – | – | 6% | 1.40 m |
| wall_fog_yolov11_ltotal | neutral->right | 0.66 s (0.66 s) | 1.30 s (1.30 s) | 13.1 °/s | -20° | 0% | 1.53 m |
| wall_fog_yolov11_ltotal | right->left | 0.57 s (0.60 s) | 1.20 s (1.22 s) | 14.5 °/s | 39° | 2% | 1.50 m |
| wall_fog_yolov11_ltotal | left->neutral | 0.43 s (0.42 s) | – (–) | – | – | 5% | 1.57 m |

| Run | RTF | Processing / frame (median, p95, wall) | Frame age at decision (sim) |
|---|---|---|---|
| city_yolov11_ltotal | 0.99 | 77 ms, 88 ms | 80 ms |
| city_fog_yolov11_ltotal | 0.99 | 79 ms, 90 ms | 108 ms |
| forest_yolov11_ltotal | 0.99 | 75 ms, 89 ms | 64 ms |
| forest_fog_yolov11_ltotal | 0.99 | 77 ms, 89 ms | 80 ms |
| wall_yolov11_ltotal | 0.99 | 76 ms, 88 ms | 76 ms |
| wall_fog_yolov11_ltotal | 0.99 | 76 ms, 87 ms | 70 ms |
