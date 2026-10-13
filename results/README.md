# Results of the paper

The logged results behind the extended evaluation of the paper (Section 7, Tables 13 and 14),
one row per frame, so that every number in the tables can be checked or recomputed.

```bash
python3 results/make_tables.py        # rebuilds Tables 13 and 14 -> results/TABLES.md (needs pandas only)
```

| Folder / file | Paper | Content |
|---|---|---|
| `TABLES.md` | Tables 13, 14 | The two tables, generated from the files below |
| `synthetic_test_set/frames.csv` | Section 7.4, Table 13 | Every frame of the simulated test set, for each pipeline and fog condition |
| `uav_gesture/frames.csv.gz` | Section 7.5, Table 14 | Every labelled frame of the UAV-GESTURE videos (every 3rd frame), for each pipeline and fog condition |
| `uav_gesture/frames_mirrored.csv.gz` | Table 14, "Mirrored" | The same frames mirrored, with LEFT and RIGHT swapped |
| `fusion_tuning/` | Section 7.1 | Choice of λ in Eq. (2) on the validation worlds (never the test worlds) |
| `training/yolov11_ltotal_synthetic/` | Section 7.2, Tables 11, 12 | Training curves, arguments and confusion matrix of YOLOv11-ℒ_total trained on synthetic data |
| `training/yolov11_ltotal_real/` | Section 7.2, Table 12 | The same, trained on the real training subject IDs (S4, S6, S9, S12, S15) |
| `weights/yolov11_ltotal_synthetic.pt` | Section 7 | Trained weights of YOLOv11-ℒ_total (synthetic-trained), usable in the ROS 2 pipeline |
| `closed_loop_flights/` | Section 7.4 | The six closed-loop flights (Gazebo + PX4 SITL + ROS 2): command and motion latency per gesture change, processing time, and the raw flight logs |

## Pipelines

| Name in the files | Paper |
|---|---|
| `alg1_rgb`, `alg1_depth` | Algorithm 1 with the stock YOLO11n-pose (COCO weights, no training) on the RGB or the depth image |
| `alg1_fused` | Algorithm 1 on the depth-weighted fused pose of Eq. (1)–(2), λ = 3000 |
| `yolov11_ltotal_synthetic` | YOLOv11-ℒ_total trained on synthetic data only |
| `yolov11_ltotal_real` | YOLOv11-ℒ_total trained on five real subject IDs (held-out subject IDs only) |

Fog conditions (`cond`): `clear`; `particle` = Gazebo particle fog (simulation only);
`phys50`, `phys20`, `phys10`, `phys5` = Koschmieder fog with visibility V = 50, 20, 10, 5 m.

## Columns

`synthetic_test_set/frames.csv`: `world` (city, forest, wall), `cond`, `pipeline`, `d` (distance
of the person, m), `k` (UAV position, 0–7), `gesture` (gesture shown by the human model), `frame` (0, or 0–1
for particle fog), `truth` (LEFT, RIGHT, NEUTRAL), `pred` (direction output), `detected` (person
found), `n_people`, `conf` (detection confidence), `dist` (estimated distance, m), `proc_ms`
(processing time of the frame).

`uav_gesture/*.csv.gz`: `video`, `cls` (UAV-GESTURE class), `subject` (subject ID), `frame`,
`label` (from the dataset's joint annotations, Section 7.2), `cond`, `pipeline`, `pred`,
`detected`, `conf`. The files can be read directly, e.g. `pandas.read_csv('frames.csv.gz')`.

A prediction is correct when `pred` equals `truth` / `label`; a false command is LEFT or RIGHT
for a NEUTRAL frame.

## Not included

- The UAV-GESTURE videos and annotations (excerpts are shown in `media/real_dataset_demo.mp4`):
  available for academic research from the authors, https://asankagp.github.io/uavgesture/, or
  from OpenDataLab, https://opendatalab.com/OpenDataLab/UAV-Gesture (UAV-GESTURE has eight
  participants, released as 11 subject IDs). For the same reason the weights trained on it (`yolov11_ltotal_real`) are not
  included; `train/build_real.py`, `train/build_real_t4.py` and `train/train.py T4R` rebuild them.
- The captured images of the simulated test set (0.6 GB) and of the training worlds (5.4 GB):
  `eval/run_capture.sh` and `train/run_capture_train.sh` regenerate them with fixed seeds.

The protocol of every test (written before each run) is in `train/PROTOCOL.md`.
