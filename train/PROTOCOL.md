# Training the YOLOv11 variant described in the paper: protocol

Written **before** any training data or model existed. The test set, metrics and
rules below are fixed now and are not changed after results are seen. Every run on the test set
is logged in `TEST_LOG.md`.

> **Notes for this repository.** This is the working protocol of the study, kept as written
> (sections and amendments were added before the runs they describe); only the names of the
> Algorithm 1 variants and of the dataset were aligned with the paper, notes about the development
> machine were removed, and section numbers may differ from it.
> Algorithm 1 of the paper includes the "gesture rules" (both arms extended → NEUTRAL; one arm
> extended with the other arm detected → that direction) and "line 13b" (left / right keypoints as
> seen by the UAV). "Algorithm 1 without the gesture rules" (`GesturePipeline.published` in the
> code) is Algorithm 1 without these two parts.
> UAV-GESTURE has eight participants released as 11 subject IDs (S1–S15); "people" here means
> subject IDs. The results of the paper are in `results/` (see `results/README.md`). The test
> log (`TEST_LOG.md`) and the other working files named below (e.g. `eval/report.py`,
> `eval/data/PAPER_COMPARISON.md`, `runs/T4_bnbug`) are not included.

## Goal

Build and train the gesture model of paper Sections 4.2–4.2.2 and measure it with the same
evaluation as the stock model (`eval/`). The aim is to measure what it does, not to reproduce
the paper's tables: whatever the test set gives is reported.

## Models (trained in stages, each measured separately = ablation)

| Stage | Model | Paper section |
|---|---|---|
| T1 | yolo11n-pose fine-tuned with 3 gesture classes (LEFT, RIGHT, NEUTRAL) on RGB | 4.2.1 (gesture classes), anchor-free head (stock YOLO11) |
| T2 | T1 + parallel depth stream fused into the feature pyramid (RGB-D input) | 4.2.1 a) depth-aware FPN |
| T3 | T2 + cross-scale attention blocks | 4.2.1 c) |
| T4 | T3 + depth (L1) and temporal-consistency loss terms | 4.2.2, Eq. 3 |

Starting weights: COCO yolo11n-pose (the stock model). Training hyperparameters: Ultralytics
defaults (as the paper states in Section 5.2) except where a change is needed for correctness,
listed here: `fliplr=0` (a mirrored image turns a LEFT gesture into RIGHT; Ultralytics swaps the
keypoints but not the class), `hsv` augmentation only on the RGB channels.

## Data

- **Training and validation:** synthetic, rendered in Gazebo in *procedurally generated
  training worlds* (random layouts of trees, buildings, walls; random ground and fog), never the
  paper's Forest / City / Wall layouts.
- **Humans:** posed from the same rigged Gazebo actor as the test humans (the only rigged body
  available), with sampled pose (pointing arm elevation, forward angle, elbow bend, other arm,
  torso turn), body proportions, height and colour. The three test meshes (`sar_human_*`) are
  never used for training.
- **Labels** come from the generator (the arm that was posed), not from Algorithm 1. Keypoints
  are projected from the posed skeleton; a keypoint is marked not visible when the ground-truth
  depth shows something in front of it.
- **Gesture definition used for labels:** LEFT / RIGHT = that arm raised sideways, shoulder
  abduction 60–110°, elbow bent at most 40°, other arm down (< 35°). NEUTRAL = both arms below
  35°, or both raised (a two-arm signal is not a turn command). Poses between 35° and 60° are not
  generated (ambiguous).
- **Validation** = separate training-world seeds; used for early stopping and every choice
  between settings.

## Test (frozen)

`eval/data/captures_test` (seed 1): the three paper worlds and their fog versions, distances
1–10 m, the same 2,520 frame-conditions per pipeline as `eval/data/results_test/REPORT.md`.
Metrics: those of `eval/report.py` (accuracy, false commands, left/right swaps, distance error,
confidence by distance) and the paper-table analogues in `eval/data/PAPER_COMPARISON.md`.
Stock pipelines (published, improved) are evaluated on the same frames, so the comparison is
paired (McNemar, cluster bootstrap as in `report.py`).

The dev captures (`eval/data/captures`, seed 0) are not used for training or model selection
either.

## Known limitations (to be stated with any result)

- Training and test humans share one base body mesh and the same renderer; results on these
  test frames will be optimistic compared with real people and cameras.
- Simulation only. Real data (UAV-GESTURE, if access is granted) would be the stronger test.
- End-to-end latency (gesture → command) is set by the 5-of-7 vote filter, not by the model.

## Implementation notes (added before any test run)

- `amp=False`: FP16 gave NaN outputs in a first check, so training runs in FP32. `seed=0, deterministic=True`. Epochs 100, batch 16, imgsz 640 (Ultralytics defaults).
- Ultralytics 8.4.13 rebuilds YOLO11's SPPF without the activation the COCO weights were trained
  with; `drone_control.rgbd_model.yolo11_sppf` restores it in every stage (checked: the untrained
  models reproduce the stock model's features exactly).
- RGB-D input: B, G, R + depth aligned to the RGB image, linear 0.1–20 m → 0–255. Depth sensor
  per training image: OakD-Lite or D435i (+ noise model), 50/50. HSV augmentation is applied to
  the colour channels of RGB-D images too (Ultralytics skips non-3-channel images).
- Fog augmentation on clear training worlds: Koschmieder fog with random visibility
  (log-uniform 4–60 m) on 35% of frames; fog worlds use the particle emitter (random density).
- Trained models are run through `GesturePipeline.trained`: the detection's class is the
  direction, nearest person commands, no other additions; same test frames and conditions as
  the stock pipelines (`eval/evaluate.py --trained`).
- Data split: 10 training worlds (+ fog variants) × 400 frames, 2 validation worlds (+ fog) ×
  300 frames; 120 training and 30 validation humans.
- Code checks of the evaluation use the dev captures (2 frames per world), never the test set.

## T4 loss terms (added before any T4 data, training or result)

The paper's Eq. 3 names the terms but not their targets, data or weights; chosen here:

- **ℒ_temporal** uses real consecutive frames: new short clips rendered in the training and
  validation worlds (never the test worlds or test humans), the drone hovering with small
  natural motion, the person's gesture constant within a clip. The term penalises the
  variation of the gesture logits of the same person across the frames of a clip.
- **ℒ_depth** (L1): one extra output per detection, the person's distance; target = the
  ground-truth distance from the rendered depth (median over the person's torso box).
- **Weights**: 1.0 for ℒ_depth and ℒ_temporal (Eq. 3 is an unweighted sum); the existing
  Ultralytics terms keep their default gains. Ultralytics' per-class BCE stands for ℒ_conf and
  ℒ_gesture; the keypoint terms of the pose model are kept (not in Eq. 3, needed for keypoints).
  If training is unstable, any change is chosen on the validation worlds only and listed here.
- T4 starts from the COCO weights like T1–T3. Clip counts and the batch layout for clips are
  fixed here before the T4 capture.

## Changes after the first test run

- **One machine for all results.** Every reported number comes from one workstation (2× RTX 4000
  Ada, torch 2.4.1). The paper-table analogues are in `train/results/REPORT.md`. The captured
  frames themselves (training and frozen test set) are unchanged.
- **Distance measurement fix (found by the first T1–T3 test run).** At 1 m the RGB camera sees
  only the person's head; the models put the shoulder and hip keypoints, which are out of
  view, on the image border with low confidence, the torso region collapsed and the distance
  fell back to the whole detection box (mostly background: ~6 m for the trained models).
  `pdd.torso_box` now uses only keypoints with confidence >= 0.5 (the project's existing
  `kp_conf` value), the head (nose, eyes, ears) when the torso is not confidently in view, and
  keeps the informative dimension when the torso region collapses on an image edge. Checked on
  the dev captures only (T1 1 m error 3.32 -> 0.01 m, stock 0.67 -> 0.37 m, 3–10 m and gesture
  accuracy unchanged). It changes the distance step for every pipeline alike, so all
  pipelines (stock and trained) are re-run on the test set with it; both runs stay in
  TEST_LOG.md. The direction decision is unchanged; the distance only matters for picking the
  nearest person when several are detected.

## T4 clip capture and training layout (fixed before the T4 capture)

- **Clips**: 4 consecutive frames at 10 Hz (the gesture node's rate), 0.1 s simulation time
  apart. Per clip one training human of the world's split at a random placement (same sampler
  as the single frames, always with a person); between frames the camera drifts like a
  hovering drone (position random walk σ 2 cm, yaw σ 0.5°, pitch σ 0.3° per frame) and the
  person sways (yaw σ 1°) while holding the gesture; particle fog evolves by itself.
  40 clips per training world (20 worlds incl. fog: 800 clips, 3,200 frames), 15 per
  validation world (4 worlds: 60 clips). Test worlds and test humans are never used.
- **ℒ_depth**: a distance branch (per scale: two 3×3 Conv + 1×1 conv, on the head's input
  features) predicts each anchor's person distance in metres; L1 against the ground truth
  (median rendered depth over the projected torso of that person) on the anchors the
  Ultralytics assigner matches to a person. Training images only (validation loss reports 0).
- **ℒ_temporal**: every training step also runs 2 clips × 4 frames (letterboxed to 640, no
  other augmentation) through the model; per frame the person's gesture logits are the mean
  class logits of the anchors whose centres lie in the central half of the person's labelled
  box; the term is the mean squared deviation of those logits from the clip mean.
  Clip frames feed only this term: the detection/keypoint data are exactly T3's 8,000 images.
- Both terms weighted 1.0 and scaled by the batch size like Ultralytics' own terms. Everything
  else as T3 (100 epochs, batch 16, seed 0, FP32, fliplr 0); best.pt chosen by the same
  validation fitness. The validation clips are used only to report frame-to-frame stability.
- Correction (after a 2-clip trial in train_3, before the capture): the frames of
  a clip are ~0.6 s simulation time apart, not 0.1 s: the simulation keeps running while the
  camera and person are repositioned between frames. The per-frame drift and sway are as
  stated; the actual time stamps are stored in each frame's meta. 0.6 s is within the
  gesture node's 5-of-7 vote window (0.7 s at 10 Hz).

## Real-data test: UAV-GESTURE (fixed before any model is run on it)

Data: the 119 UAV-GESTURE videos (Perera et al. 2018; 11 real people, outdoor, hovering UAV,
GoPro 1920×1080, 25 fps; `<workspace>/UAVGesture`) and the authors' joint annotations
(OpenPose COCO-18 order, checked visually: 2/3/4 = right arm, 5/6/7 = left arm; from
github.com/asankagp/UAV-GESTURE, MIT). RGB only, so only the RGB pipelines can be tested:
stock YOLOv11n-pose + Algorithm 1 without the gesture rules, and T1. Not used for training or any choice.

- **Frames**: every 3rd frame of all 119 videos (8.3 Hz, nearest to the node's 10 Hz).
- **Label per frame, from the annotated joints** (not from the video's class, none of the 13
  marshalling signals is the paper's gesture), with the gesture definition of this protocol in
  the image plane: per arm, abduction = angle between upper arm (shoulder→elbow) and the
  body's down direction (neck→mid-hip), bend = angle between upper arm and forearm.
  Raised = abduction 60–110° and bend ≤ 40°; down = abduction < 35°.
  LEFT = left arm raised, right arm down; RIGHT = the mirror; NEUTRAL = both arms down, or
  both arms at ≥ 60° (two-arm signal). Everything else (35–60°, a raised arm bent > 40°
  with the other down, overhead > 110° with the other down) and frames with missing joints
  are not labelled. Caveat: 2-D angles from an elevated camera approximate the 3-D ones.
- **Pipelines** exactly as on the test set (conf 0.25, 640 px, nearest person); no depth, so
  no distance.
- **Measures**: detection rate and mean confidence (all frames); per-frame accuracy,
  false commands (NEUTRAL → LEFT/RIGHT) and LEFT↔RIGHT swaps on labelled frames, also by
  UAV-GESTURE class; the 5-of-7 vote filter over each video's frame sequence (share of time a
  wrong LEFT/RIGHT command is active). Paired comparison Alg. 1 vs T1: exact McNemar, 95 %
  bootstrap interval with the 11 people as clusters.
- Run once, logged in TEST_LOG.md.

## Training with real images: T1R–T4R (fixed before any real training data is built)

Motivation: the UAV-GESTURE test (above) showed that T1, trained on synthetic people only,
does not transfer to real people. Each stage is retrained with real images added.

- **Split by person** (11 UAV-GESTURE people, `default_rng(2026).permutation`):
  test = S1, S3, S11, S14; validation = S5, S13; training = S4, S6, S9, S12, S15.
  No frame of a test person is used for training or any choice.
- **Real training / validation frames**: every 6th frame of those people's videos, only frames
  labelled from the joints (definition as in the UAV-GESTURE test), each also mirrored with
  LEFT↔RIGHT and the keypoints swapped (real LEFT frames are rare: 34, all S9).
  Labels from the authors' joints: class as defined; 17 COCO keypoints mapped from OpenPose
  (missing = not visible); box = joints' extent widened by 10 % of its width each side, top
  raised by 0.6 × the nose–neck distance (head), bottom lowered by 5 % of the height (feet).
  Images 1280×720 like the synthetic ones.
- **Depth for real frames**: estimated per frame with Depth-Anything-V2-Metric-Outdoor-Small
  (monocular, metric), resampled to the image and encoded like the sensor depth (0.1–20 m
  linear). Reported as estimated depth, not a depth sensor. The same estimate is used for
  every real frame at training and test time.
- **Training**: as T1–T4 (from COCO, 100 epochs, batch 16, seed 0, FP32, fliplr 0), with the
  real training frames added to the 8,000 synthetic images and the real validation frames to
  the synthetic validation set (best.pt by the same fitness). T4R: the same synthetic clips;
  real images have no ground-truth distance, so they add no ℒ_depth term.
- **Tests** (each model once, in TEST_LOG.md):
  (1) real: the 4 test people, every 3rd frame (UAV-GESTURE test measures); T1R and stock
  Alg. 1 RGB; T2R–T4R and the original T2–T4 with estimated depth;
  (2) real, mirrored: the same frames mirrored with LEFT↔RIGHT swapped (the only LEFT test
  on real people; the test people never show LEFT);
  (3) the frozen synthetic test set, as T1–T4.

### Amendment (before any T1R–T4R training): real-only training, separate studies

Synthetic and real data and results are kept fully separate. Replacing the
"Training" and "Tests" items above:
- **T1R–T4R are trained only on real images**: the real training frames (people S4, S6, S9,
  S12, S15, as above) from COCO yolo11n-pose with the T1–T4 recipes; best.pt chosen on the
  real validation people (S5, S13) only. No synthetic image or clip is used.
- **T4R**: ℒ_temporal on real clips: 4 consecutive labelled frames 3 video frames apart
  (0.12 s), one clip starting every 24 frames of each training person's videos, all 4 frames
  with the same label; person box from the joints as above. ℒ_depth target = median of the
  model's input depth channel (estimated depth) over the torso (shoulders + hips annotated),
  else over the central third of the box, i.e. "coherence between detection and the
  corresponding depth map" with the estimated map.
- **Tests**: held-out real people (S1, S3, S11, S14), original and mirrored, for T1R–T4R and
  stock Alg. 1 RGB. The synthetic models T1–T4 are also run on these frames (cross-domain,
  reported in its own section). T1R–T4R are not run on the synthetic test set.
- **Separate results**: real study in `train/results_real/` and
  `eval/data/results_uavgesture_heldout/`; the synthetic study stays in `train/results/`.

### Amendment (after the first T4 / T4R tests): BatchNorm side effect of the ℒ_temporal pass

Found after the held-out real test: T4R ranked people correctly (val mAP50 0.99) but output
confidences ≈ 0.05, below the pipeline's 0.25 threshold (detection 2 %); the synthetic T4
showed the same, weaker ("lower confidence"). Cause, an implementation error, not part of the
paper's method: the extra forward pass over the clips for ℒ_temporal ran the network in
training mode, so every clip batch (8 unaugmented letterboxed frames) also updated the
BatchNorm running statistics used at inference. Diagnosis (validation set only; 10-epoch T4R
runs, deterministic): with the clip pass not updating the running statistics, the training
losses are identical to 3 decimals (same weights) while val cls loss falls 0.71 → 0.27 like
a run without ℒ_temporal (0.26), instead of rising.
- **Fix**: during the ℒ_temporal clip pass the BatchNorm layers use batch statistics as before
  (gradients unchanged) but their running statistics are not updated (momentum 0 for that pass).
  Nothing else in T4 / T4R changes (data, clips, weights 1.0, recipe, seed).
- **Re-run**: T4 and T4R retrained from COCO; each tested once more (T4: frozen synthetic test
  set; T4R and T4: held-out real people, original + mirrored); the 6 T4 gesture-switch and
  6 T4 distance flights flown again; paper comparison rebuilt. The first T4 / T4R weights and
  all their results are kept (runs/T4_bnbug, runs/T4R_bnbug, *_bnbug result files and flight
  folders) and stay in TEST_LOG.md, marked as affected.
- **Outcome**: the fixed T4 / T4R were better (synthetic test: accuracy 88→91 %
  clear, 73→86 % particle fog; real held-out people: T4R accuracy 2→99 %; distance flights:
  confidence at 10 m 0.21–0.48 → 0.94–0.95). The first (affected) T4 / T4R
  weights and result files were then deleted; their TEST_LOG.md entries remain, marked.

### Amendment (before the run): synthetic-trained models on the full UAV-GESTURE dataset

Added test: the synthetic-trained models on all of UAV-GESTURE, no split.
- **Models**: stock yolo11n-pose with Algorithm 1 (RGB) and T1–T4 as trained on synthetic data
  only (runs/T1–T4, T4 after the BatchNorm fix). T1R–T4R are not run here (7 of the 11 people
  are their training / validation data); they stay on the held-out people.
- **Data**: all 119 videos, all 11 people (none was used to train or choose T1–T4); every 3rd
  frame; labels from the authors' joints as in the UAV-GESTURE test; frames resized to
  1280×720; T2–T4 get the estimated metric depth (Depth-Anything-V2-Metric-Outdoor-Small), as in
  the held-out real test. Original frames (includes the real LEFT frames of S9) and mirrored
  frames (LEFT↔RIGHT swapped).
- **Measures**: as the held-out real test (detection, confidence, accuracy, LEFT/RIGHT accuracy,
  false commands, swaps, filtered wrong commands; paired vs Alg. 1 with McNemar and a 95 %
  bootstrap CI over the 11 people as clusters).
- **Output**: eval/data/results_uavgesture_full/REPORT.md, copied to
  train/results/UAVGESTURE_FULL.md (synthetic study, reported as sim-to-real); each model
  logged once in TEST_LOG.md.

### Change: both-arms-raised line added to Algorithm 1

One line is added to Algorithm 1, step 15: if both arms are raised (both shoulder angles
≥ 65°, the lower bound of the paper's extended-arm range) the direction is NEUTRAL (the UAV
hovers). Implemented as `both_arms_hover` (pdd.classify_direction): **on by default in the ROS 2
gesture node** (launch `both_arms_hover:=true`, also with `robust:=false`). It does not apply to
the trained models (their class is the direction). All results so far, and the flight batch
scripts that reproduce them (`robust:=false both_arms_hover:=false`), use Algorithm 1 without this
rule; evaluation code (`GesturePipeline.published`) keeps it off unless asked.
Any test of Algorithm 1 with the line is a new, separately logged run.
- **Amendment (gesture rules replace the line above)**: "stretched out" = the paper's
  extended-arm thresholds (shoulder 65–106°, elbow 100–180°). Both arms stretched out → NEUTRAL
  (hover); exactly one arm stretched out with the other arm seen (shoulder and elbow keypoint
  confidence ≥ 0.5) down or up → that arm's direction (operator's left arm → UAV turns right,
  right arm → turns left, unchanged); anything else → NEUTRAL (hover). Same switch
  (`both_arms_hover`, default on in the node; off = without the gesture rules).
- **Second amendment**: one arm stretched out with the other arm not seen also follows that
  arm (no visibility check). Final rule: both arms stretched out → NEUTRAL (hover); exactly one
  arm stretched out → that arm's direction; no arm stretched out → NEUTRAL (hover). This is the
  same decision as the existing `one_arm` addition, now applied to Algorithm 1 itself.
- **Final**: back to the visibility check: one arm stretched out with the other
  arm not seen (its shoulder or elbow keypoint confidence < 0.5, e.g. hidden by fog) → NEUTRAL
  (hover). Final rule: both arms stretched out → hover; exactly one arm stretched out and the
  other arm seen (down or up) → that arm's direction; anything else → hover.

### Test plan (before the run): Algorithm 1 with the gesture rules

Stock yolo11n-pose, no training, Algorithm 1 with and without the final gesture rules, run
together in the same pass (paired, same frames):
1. **Frozen synthetic test set** (`eval/data/captures_test`, all conditions, OakD-Lite and D435i,
   RGB and depth modes): `evaluate.py --rules` -> `eval/data/results_test_rules/frames.csv`.
   Question: do the paper-reproduction results change (share of identical predictions,
   accuracy, detection, false commands per condition)?
2. **Full UAV-GESTURE** (all 11 people, 119 videos, every 3rd frame, original and mirrored, RGB):
   `uavgesture_heldout.py run --rules` -> `eval/data/results_uavgesture_rules/`, report
   `train/results/UAVGESTURE_RULES.md`. Same measures as the full UAV-GESTURE test.
Each logged once in TEST_LOG.md.

### Amendment (before re-scoring): UAV-GESTURE labels v2 for the gesture rules

The UAV-GESTURE frame labels (section "Real-data test: UAV-GESTURE") marked "one arm raised +
other arm also at ≥ 60°" as NEUTRAL; gesture rule 4 makes that pose a command. Labels v2
change only that case: one arm raised (abduction 60–110°, bend ≤ 40°) and the other arm up
(abduction ≥ 60° but not raised, i.e. overhead > 110° or bent > 40°) → the raised arm's
direction. Both arms raised stays NEUTRAL; every other label and the unlabelled cases are
unchanged. Applied by re-scoring the finished Algorithm 1 run (original and with the gesture
rules, `eval/data/results_uavgesture_rules/`), no model is run again ->
`eval/data/results_uavgesture_rules_labels2/`, `train/results/UAVGESTURE_RULES_LABELS2.md`.
The original labels remain the ones for every earlier result (incl. T1R–T4R training).

### Test plan (before the run): Algorithm 1 depth mode on UAV-GESTURE

Stock yolo11n-pose, no training, on all 11 people / 119 videos (every 3rd frame, 1280×720,
original + mirrored): Algorithm 1 RGB and depth mode, each without and with the gesture
rules (4 pipelines, same pass, same frames). Depth = Depth-Anything-V2-Metric-Outdoor-Small
estimate of each frame (as in the real-data tests), then Algorithm 1 step 4 unchanged (clip
0.2–12 m, normalise, JET colour map, pose on the colour map, 640 px). Estimated depth, not a
depth sensor: it shows how Algorithm 1's depth mode reads real people, not sensor behaviour.
Scored with the original labels and, by re-scoring, with labels v2 ->
`eval/data/results_uavgesture_depth{,_labels2}/`, `train/results/UAVGESTURE_DEPTH{,_LABELS2}.md`.
Logged once in TEST_LOG.md.
- **Amended before any result (the first start was stopped after ~25 min, nothing
  scored)**: two more pipelines in the same pass, Algorithm 1 depth mode (as
  published / with the gesture rules) on the estimated depth **with the D435i sensor model**
  (`sensor_model.d435i_depth_noise`, as in the synthetic study: stereo error σ = z²·0.08 /
  (f·0.05 m), f from the D435i depth FOV at the frame width, no return outside 0.28–10 m;
  noise seeded per video and frame, identical for the mirrored frame). Still not a physical
  sensor: no holes from texture or sunlight, no edge artefacts.

### Test plan (before the run): Algorithm 1 line 13b, sides as the UAV sees them

Decision (measure, then add): before the joint angles (step 14), if the left-shoulder keypoint
lies on the image's left of the right-shoulder keypoint, all left/right keypoint pairs are swapped
(pdd.face_camera), so "right" always means the arm on the UAV's left. This matches the gesture
rules (direction as seen by the UAV) for an operator facing the UAV or turned away; it does not
help a side-on operator. Stock yolo11n-pose, no training; Algorithm 1 with the gesture rules,
without and with line 13b, in the same pass on the same frames:
1. Frozen synthetic test set (all conditions, RGB and depth, OakD-Lite and D435i):
   `evaluate.py --rules --fix` -> `eval/data/results_test_uavview/`, report
   `train/results/UAVVIEW_SYNTHETIC.md` (identical predictions, accuracy, swaps, false commands).
2. Full UAV-GESTURE (11 people, 119 videos, every 3rd frame, original + mirrored): RGB, estimated
   depth, estimated depth + D435i noise -> `eval/data/results_uavgesture_uavview/`, reports
   `train/results/UAVGESTURE_UAVVIEW{,_LABELS2}.md` (original labels and labels v2).
Line 13b is added to Algorithm 1 (ROS 2 node default) only after these results are reported.
Each run logged once in TEST_LOG.md.
- **Outcome and adoption**: synthetic test set, Algorithm 1 + gesture rules without → with
  line 13b: depth swaps 19–37% → 0%, depth accuracy +12 to +25 points (OakD clear 36% → 61%), 968 of
  969 changed depth frames became correct and none became wrong; RGB 94–100% identical, accuracy +0 to
  +5 points, swaps 8–10% → 0–2% (128 frames better, 12 worse). UAV-GESTURE (labels v2): RGB identical
  (89% / 90%); estimated depth 73% → 86% (mirrored 73% → 85%), swaps 45% → 2%; with D435i noise 70% → 82%.
  Line 13b is therefore added to Algorithm 1: ROS 2 node default (launch uav_view:=true, also with
  robust:=false); the original behaviour stays available (uav_view:=false) and the flight batch scripts
  that reproduce the earlier numbers are pinned to it. The earlier results tables were produced
  without line 13b and without the gesture rules.

### Test plan (before the run): Algorithm 1 on trained keypoints

Question: does Algorithm 1 (gesture rules + line 13b) gain from keypoints of the synthetic-trained
model? The synthetic-trained T4 (YOLOv11-L_total, RGB + depth input) only supplies the keypoints;
Algorithm 1 decides the direction from them (`GesturePipeline.published(T4, mode='rgbd',
both_arms_hover=True, face_camera=True)`), instead of T4's own classes. Nothing is trained or tuned.
- Frozen synthetic test set (`eval/evaluate.py --only-trained --trained-alg1 T4=...:rgbd`), all
  conditions, OakD-Lite and D435i; compared with the logged stock Algorithm 1 (rules + 13b) and T4
  (classes) runs on the same frames.
- Full UAV-GESTURE (11 people, every 3rd frame, labels v1, original + mirrored), estimated depth
  (`eval/uavgesture_heldout.py run --alg1trained`); stock Algorithm 1 RGB / depth (rules + 13b) and
  T4 classes re-run in the same pass (must reproduce the logged predictions).

### Test plan (before the run): Algorithm 1 on real people in Koschmieder fog

Algorithm 1 (gesture rules + line 13b), stock yolo11n-pose, no training, RGB and depth mode, on the
full UAV-GESTURE dataset (11 people, every 3rd frame, labels v1, original + mirrored), in clear air
and Koschmieder fog V = 50, 20, 10, 5 m (`eval/uavgesture_fog.py`). The fog uses the same model
as the synthetic test set (`sensor_model.fog_rgb` / `fog_depth`) with the depth estimated from the
clear frame (Depth-Anything-V2); depth mode runs on the fogged estimate. Particle fog cannot be
applied to recorded videos. The clear condition must reproduce the logged Algorithm 1 predictions.

### Test plan (before the run): YOLOv11-L_total on real people in Koschmieder fog

The same frames, fog and conditions as the previous test, for YOLOv11-L_total (RGB-D: fogged RGB +
fogged estimated depth): T4 (synthetic-trained) on all 11 people, T4R (real-trained) on its 4
held-out people (S1, S3, S11, S14) only; each with its own classes and with Algorithm 1 (gesture
rules + line 13b) on its keypoints (`eval/uavgesture_fog.py --trained`). No training, no choices
on these frames. The clear condition must reproduce the logged T4 / T4R predictions
(results_uavgesture_full, results_uavgesture_heldout, results_uavgesture_alg1trained).

### Test plan (before the run): depth-weighted pose fusion, paper Eq. (1)-(2)

Implemented in `drone_control/fusion.py` (`FusedPipeline`, `input_mode = fused`), stock
yolo11n-pose, no training. Per frame: pose on the RGB frame and on the colour-mapped depth frame
(both 640 px, face_camera applied to each); depth persons are matched to RGB persons by box IoU
> 0.3 after mapping into RGB pixels; unmatched depth persons are kept. For every person and COCO
joint j, Eq. (1) P_j = a D_j + b RGB_j + g T_j + d C_j with
- RGB_j, D_j: the RGB / depth keypoint back-projected to 3D (pinhole, RGB camera) with z_j = median
  of the depth (aligned to RGB) within r = w/256 px of the keypoint, using only values within
  0.6 m of the person's torso distance (else z_j = torso distance; no depth at all: z = 5 m);
- T_j: mean of the person's fused joint over the last W = 5 processed frames of the same video
  (track: torso within 0.75 m of the previous one); none on single frames;
- C_j: body-plane prior: the mean of the available observations with z = torso distance (the
  operator stands upright, facing the UAV, arms near the torso plane).
Eq. (2): weights softmax(-lambda sigma_i^2) over the modalities available for that joint, with
sigma_RGB^2, sigma_D^2 = (s z_j / f)^2 / c_ij + var(depth samples), s = 4 px at 640 px input scaled
to the image, c_ij the keypoint confidence (< 0.05 or (0, 0) = not available); sigma_T^2 = variance
of the W positions + 0.05^2; sigma_C^2 = 0.25^2 (m^2). A keypoint is used only if RGB or D sees it;
its fused confidence is the weight-averaged RGB / D confidence. The fused joints are projected
back to RGB pixels and Algorithm 1 (gesture rules + line 13b) decides; distance as in RGB mode.
lambda is the only tuned value: chosen from {0, 10, 30, 100, 300, 1000, 3000} by direction
accuracy on the validation worlds val_0, val_1 (clear, particle fog and Koschmieder V = 50, 20,
10, 5 m; OakD), never on the test worlds or real people. Then one run on the synthetic test set
(captures_test, all conditions, both sensors) and on the full UAV-GESTURE dataset (11 people,
original + mirrored, clear + Koschmieder V = 50/20/10/5, estimated depth). Reported as measured,
whether better or worse than Algorithm 1 on RGB or depth alone.

### Amendment (after the first real-people run): temporal term T_j of Eq. (1)

The first run on real people (results_uavgesture_fog/fused_v1[_mirror].csv) showed a defect:
T_j was the mean of the person's previous *fused* joints, so once T_j won a joint (smallest
sigma^2), the fused output equalled T_j, went back into the history and kept its variance at the
floor: joints froze (e.g. a shoulder at the same pixel for the whole video) and pointing arms read
as NEUTRAL (LEFT/RIGHT accuracy 12%). The synthetic test set and the lambda choice were not
affected (independent single frames: T_j never active). Fix, without other changes: T_j is the
mean of the person's last W = 5 observation-based estimates (RGB_j, D_j, C_j fused without T_j),
and C_j is built from RGB_j and D_j only. Checked: the synthetic test predictions are unchanged
(1440 frame-conditions identical); unit test test_temporal_term_follows_the_observations fails on
the old code and passes on the new one. The real-people test is repeated with the fixed code;
both runs are reported.
