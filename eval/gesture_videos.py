#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Demonstration videos of Algorithm 1 (stock YOLOv11n-pose, no trained model, step 15 with the
gesture rules), one per gesture, with a legend of what happens (run_gesture_videos.sh).

    gesture_videos.py synthetic <flight dir> <gesture> <rgb|depth> <out.mp4>
        a recorded Gazebo + PX4 SITL + ROS 2 flight (RECORD=1 run_flight.sh): third-person view
        and the drone's annotated camera side by side, legend below
    gesture_videos.py real <rgb|depth> <out dir>
        UAV-GESTURE clips (the 5 s in which the annotated joints hold the pose best), Algorithm 1
        on every frame (25 fps), legend below; depth = estimated from the frame
    gesture_videos.py approach <out.mp4>
        one longer UAV-GESTURE video (RGB): three people, each with arms down, the right arm and the
        left arm (mirrored), while the UAV approaches 6.5 -> 5 -> 3.5 m, emulated by digital zoom as
        in the demo's "Real people" mode; expected direction per frame from the annotated joints
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

import media
import simtools as st

HERE = os.path.dirname(os.path.abspath(__file__))
from demo_common import (CLIP, CYAN, GESTURES, GREEN, GREY, REAL, RED, WHITE,  # noqa: F401
                         ok_colour, put)

ACTION = {'RIGHT': 'turn LEFT', 'LEFT': 'turn RIGHT', 'NEUTRAL': 'hover', 'NONE': 'hover (nobody seen)'}
RULE = ['Algorithm 1, step 15: one arm stretched out (shoulder 65-106 deg, elbow 100-180 deg) and the other arm seen',
        '-> turn towards the pointed side (operator\'s RIGHT arm = UAV turns LEFT); both arms out, no arm out,',
        'or the other arm not seen -> hover. A command is sent when it wins 5 of the last 7 frames.']
SKELETON = [((0, 1), 0), ((0, 2), 0), ((1, 3), 0), ((2, 4), 0), ((5, 7), 1), ((7, 9), 1),
            ((6, 8), 1), ((8, 10), 1), ((5, 6), 2), ((5, 11), 2), ((6, 12), 2), ((11, 12), 2),
            ((11, 13), 3), ((13, 15), 3), ((12, 14), 3), ((14, 16), 3)]
COLOURS = [(0, 255, 0), (255, 128, 0), (255, 0, 255), (0, 165, 255)]


def encoder(path, w, h, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return subprocess.Popen([media.ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                             '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-vf', 'scale=1920:-2',
                             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '21', path],
                            stdin=subprocess.PIPE)




# ---------------------------------------------------------------------------- synthetic flights
def synthetic(fdir, gesture, mode, out, fps=15.0):
    ct, cf = media.frames(f'{fdir}/frames', 'chase')
    ot, of = media.frames(f'{fdir}/frames', 'onboard')
    ev = [json.loads(line) for line in open(f'{fdir}/log.jsonl')]
    ser = lambda k: (np.array([e['sim'] for e in ev if e['kind'] == k]),       # noqa: E731
                     [e['data'] for e in ev if e['kind'] == k])
    sw_t, sw_v = ser('swap')
    cm_t, cm_v = ser('cmd')
    fr_t, fr_v = ser('frame')
    ps = [(e['sim'], e['data']) for e in ev if e['kind'] == 'pos' and e['data'][4] is not None]
    ps_t, ps_v = np.array([p[0] for p in ps]), [p[1] for p in ps]
    end = next(e['sim'] for e in ev if e['kind'] == 'end')
    lift = next(t for t, p in ps if p[4] > 0.5)
    h0 = media.latest(ps_t, ps_v, lift)[3]
    t0, t1 = max(ct[0], sw_t[0] - 2.0 if len(sw_t) else lift), min(ct[-1], end + 1.0)
    W, H, BAR = 1280, 720, 300
    enc = encoder(out, 2 * W, H + BAR, fps)
    name, desc, want, act = GESTURES[gesture]
    cam = 'RGB camera' if mode == 'rgb' else 'depth camera (OakD-Lite), colour-mapped as in step 4'
    blank = np.zeros((H, W, 3), np.uint8)
    for t in np.arange(t0, t1, 1.0 / fps):
        c, o = media.latest(ct, cf, t), media.latest(ot, of, t)
        left = cv2.imread(c) if c else blank.copy()
        right = cv2.imread(o) if o else blank.copy()
        put(left, 'Third-person view (Gazebo)', (20, 44), 1.0, WHITE)
        put(right, f'Drone camera + Algorithm 1 ({mode.upper()})', (W - 560, 44), 0.9, WHITE)
        bar = np.full((BAR, 2 * W, 3), 32, np.uint8)
        put(bar, f'SYNTHETIC  |  live flight: Gazebo + PX4 SITL + ROS 2  |  Algorithm 1 '
                 f'(stock YOLOv11n-pose, no training)  |  {cam}', (24, 46), 1.0, WHITE)
        shown = media.latest(sw_t, sw_v, t) or 'neutral'
        sdesc = GESTURES[shown][1] if shown in GESTURES else shown
        swant, sact = st.LABEL.get(shown, 'NEUTRAL'), ACTION[st.LABEL.get(shown, 'NEUTRAL')]
        put(bar, f'Person shows: {sdesc}   ->   expected: {swant}, UAV should {sact}', (24, 100), 1.0, CYAN)
        f = media.latest(fr_t, fr_v, t)
        raw = f.get('raw', '-') if isinstance(f, dict) else '-'
        cmd = media.latest(cm_t, cm_v, t) or '-'
        put(bar, f'Algorithm 1, this frame: {raw}', (24, 154), 1.0, ok_colour(raw, swant))
        put(bar, f'Command (5 of 7 frames): {cmd}', (760, 154), 1.0, ok_colour(cmd, swant))
        p = media.latest(ps_t, ps_v, t)
        if p:      # PX4 heading is clockwise from north: negative change = turned left
            turn = (p[3] - h0 + 180) % 360 - 180
            q = media.latest(ps_t, ps_v, t - 0.4)
            rate = ((p[3] - q[3] + 180) % 360 - 180) / 0.4 if q else 0.0
            want = {'RIGHT': -1, 'LEFT': 1}.get(cmd, 0)      # RIGHT -> the UAV turns left (heading down)
            if want and turn * want > 15 and rate * want <= 3:
                state = 'holding at the 20 deg turn limit'   # incl. settling back after a small overshoot
            elif not want and abs(turn) <= 3:
                state = 'hovering'                           # yaw jitter around the start heading
            elif not want and rate * turn < 0 and abs(rate) > 3:
                state = 'returning to face the person'
            elif rate < -3:
                state = 'turning LEFT'
            elif rate > 3:
                state = 'turning RIGHT'
            else:
                state = 'hovering'
            put(bar, f'UAV: {state}  (heading {turn:+.0f} deg, height {p[4]:.2f} m)', (1500, 154), 1.0, WHITE)
        put(bar, f'{t - lift:+5.1f} s after take-off', (2200, 46), 0.8, GREY)
        for i, line in enumerate(RULE):
            put(bar, line, (24, 204 + 32 * i), 0.75, GREY, 1)
        enc.stdin.write(np.vstack([np.hstack([left, right]), bar]).tobytes())
    enc.stdin.close()
    enc.wait()
    print(out)


# -------------------------------------------------------------------------------- real videos
# demo_common.REAL: gesture -> (video, first frame, mirrored), 5 s (125 frames) each; the dataset
# has no real left-arm-only pointing, so LEFT = a right-arm clip mirrored (labels swapped).
SWAPG = {'right': 'left', 'left': 'right', 'rightout_leftup': 'leftout_rightup',
         'leftout_rightup': 'rightout_leftup'}


def joint_categories(p):
    """Per frame: the pose from the annotated joints (same definition as the real-data tests)."""
    import uavgesture as ug
    j = p.transpose(2, 1, 0)
    lab, la, ra = ug.frame_labels(p)
    bend = {}
    for side, (s, e, w) in (('L', (ug.LSH, ug.LEL, ug.LWR)), ('R', (ug.RSH, ug.REL, ug.RWR))):
        bend[side] = ug.angle(j[:, e] - j[:, s], j[:, w] - j[:, e])
    ok = np.array([x is not None for x in lab]) | ((la > 110) & (ra > 110))
    rl = (la >= 60) & (la <= 110) & (bend['L'] <= 40)
    rr = (ra >= 60) & (ra <= 110) & (bend['R'] <= 40)
    c = np.full(len(lab), None, object)
    c[ok & rr & (la < 35)] = 'right'
    c[ok & rl & (ra < 35)] = 'left'
    c[ok & (la < 35) & (ra < 35)] = 'neutral'
    c[ok & rl & rr] = 'both_out'
    c[ok & (la > 110) & (ra > 110)] = 'both_up'
    c[ok & rr & ~rl & (la >= 60)] = 'rightout_leftup'
    c[ok & rl & ~rr & (ra >= 60)] = 'leftout_rightup'
    return c


def draw(img, people):
    lw = max(2, img.shape[1] // 640)
    for pr in people:
        x1, y1, x2, y2 = (int(v) for v in pr.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), lw)
        for (a, b), c in SKELETON:
            if np.any(pr.keypoints[a]) and np.any(pr.keypoints[b]):
                cv2.line(img, tuple(int(v) for v in pr.keypoints[a]), tuple(int(v) for v in pr.keypoints[b]),
                         COLOURS[c], lw)
        put(img, f'person {pr.conf:.2f}  {pr.direction}', (x1, max(20, y1 - 10)), 0.7, (0, 255, 0))


def real(mode, out_dir, device='1'):
    import scipy.io as sio
    from ultralytics import YOLO
    import uavgesture as ug
    sys.path.insert(0, os.path.join(HERE, '..', 'train'))
    from build_real import DepthEstimator, H, W
    from drone_control import pdd
    from drone_control.pipeline import DEFAULT_WEIGHTS, GesturePipeline
    est = DepthEstimator(f'cuda:{device}') if mode == 'depth' else None
    model = YOLO(DEFAULT_WEIGHTS)
    vids = {v['video']: v for v in ug.videos()}
    BAR = 250
    cam = ('RGB video frame' if mode == 'rgb' else
           'depth estimated from the frame (Depth-Anything-V2), colour-mapped as in step 4')
    for g, (vid, start, mirror) in REAL.items():
        v = vids[vid]
        cats = joint_categories(sio.loadmat(v['mat'])['pos_img'])
        pipe = GesturePipeline.published(model, mode=mode, device=device, both_arms_hover=True,
                                         face_camera=True)
        tf = pdd.TemporalFilter(7, 5)
        name, desc, want, act = GESTURES[g]
        out = f'{out_dir}/{name}.mp4'
        enc = encoder(out, W, H + BAR, 25)
        cap = cv2.VideoCapture(v['path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        n_raw = n_cmd = n = 0
        for f in range(start, start + CLIP):
            ok, img = cap.read()
            if not ok:
                break
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            depth = est(img).astype(np.float32) if est else None
            if mirror:
                img = np.ascontiguousarray(img[:, ::-1])
                depth = None if depth is None else np.ascontiguousarray(depth[:, ::-1])
            frame, people, raw, best = pipe.process(img if mode == 'rgb' else None, depth)
            cmd = tf.update(raw)
            n, n_raw, n_cmd = n + 1, n_raw + (raw == want), n_cmd + (cmd == want)
            view = frame.copy()
            draw(view, people)
            put(view, f'Direction (command): {cmd}', (16, 40), 1.0, (0, 0, 255))
            jc = cats[f] if f < len(cats) else None
            jc = SWAPG.get(jc, jc) if mirror else jc
            bar = np.full((BAR, W, 3), 32, np.uint8)
            src = f"REAL  |  UAV-GESTURE, person S{v['subject']}, signal '{v['cls']}'" + \
                  ('  |  MIRRORED left-right (no real left-arm-only pointing in the dataset)' if mirror else '') + \
                  '  |  recorded video, no flight'
            put(bar, src, (16, 34), 0.6, WHITE, 1)
            put(bar, f'Algorithm 1 (stock YOLOv11n-pose, no training) on the {cam}', (16, 62), 0.6, WHITE, 1)
            put(bar, f'Person shows: {desc}  ->  expected: {want} (UAV: {ACTION[want]})', (16, 98), 0.65, CYAN)
            put(bar, 'this frame, from the annotated joints: ' + (GESTURES[jc][1] if jc in GESTURES
                                                                   else 'between poses'), (16, 126), 0.55, GREY, 1)
            put(bar, f'Algorithm 1, this frame: {raw}', (16, 162), 0.7, ok_colour(raw, want))
            put(bar, f'Command (5 of 7): {cmd}', (450, 162), 0.7, ok_colour(cmd, want))
            put(bar, f'A UAV would: {ACTION.get(cmd, "hover")}', (850, 162), 0.7, WHITE)
            for i, line in enumerate(RULE):
                put(bar, line, (16, 196 + 20 * i), 0.45, GREY, 1)
            enc.stdin.write(np.vstack([view, bar]).tobytes())
        enc.stdin.close()
        enc.wait()
        print(f'{out}  frames {n}: Algorithm 1 = {want} in {100 * n_raw / n:.0f}%, command {100 * n_cmd / n:.0f}%',
              flush=True)


# ---------------------------------------------------------------- real videos, approach by zoom
# stop distance (m) -> segments (gesture, video, first frame, last frame, mirrored). Arm signals in
# which the arm is held inside the range of step 15 ('All Clear', 'Not Clear': the arm is
# stretched out and lowered about once a second); LEFT = a right-arm video mirrored (the dataset has
# no left-arm-only pointing). The first 3 s of each stop are the approach from the previous stop.
APPROACH = [
    (6.5, [('neutral', 'S6_land_HD', 0, 125, False), ('right', 'S6_allClear_HD', 5, 305, False),
           ('left', 'S6_notClear_HD', 5, 305, True)]),
    (5.0, [('neutral', 'S11_land_HD', 0, 145, False), ('right', 'S11_allClear_HD', 190, 429, False),
           ('left', 'S11_notClear_HD', 145, 382, True)]),
    (3.5, [('neutral', 'S12_land_HD', 0, 190, False), ('right', 'S12_allClear_HD', 100, 400, False),
           ('left', 'S12_notClear_HD', 0, 288, True)]),
]
RAMP = 75                    # frames (3 s) of zoom from the previous stop's distance to this one


def approach(out, device='0'):
    import scipy.io as sio
    from ultralytics import YOLO
    import uavgesture as ug
    sys.path.insert(0, os.path.join(HERE, '..', 'train'))
    from build_real import DepthEstimator, H, W
    from drone_control import pdd
    from drone_control.pipeline import DEFAULT_WEIGHTS, GesturePipeline
    est = DepthEstimator(f'cuda:{device}')
    model = YOLO(DEFAULT_WEIGHTS)
    vids = {v['video']: v for v in ug.videos()}

    def frames(vid, a, b):
        cap = cv2.VideoCapture(vids[vid]['path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        for _ in range(a, b):
            ok, img = cap.read()
            if not ok:
                return
            yield img

    def recorded(vid, a, b):
        """camera-person distance of the recording (m): median estimated depth over the torso,
        as for data/real_clip_distances.json"""
        pipe = GesturePipeline.published(model, mode='rgb', device=device, both_arms_hover=True,
                                         face_camera=True)
        ds = []
        for i, img in enumerate(frames(vid, a, b)):
            if i % 10:
                continue
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            _, _, _, best = pipe.process(img, None)
            if best is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in pdd.torso_box(best.keypoints, best.box))
            ds.append(float(np.median(est(img)[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)])))
        return float(np.median(ds))

    pipe = GesturePipeline.published(model, mode='rgb', device=device, both_arms_hover=True,
                                     face_camera=True)
    tf = pdd.TemporalFilter(7, 5)
    BAR = 290
    enc = encoder(out, W, H + BAR, 25)
    prev, total = APPROACH[0][0], {'n': 0, 'raw': 0, 'cmd': 0}
    for k, (target, segs) in enumerate(APPROACH):
        for s, (g, vid, a, b, mirror) in enumerate(segs):
            v = vids[vid]
            p = sio.loadmat(v['mat'])['pos_img']
            cats = joint_categories(p)
            j = p[:, :, a:b].transpose(2, 1, 0).reshape(-1, 2)
            cx0, cy0 = np.median(j[np.abs(j).sum(1) > 0], 0)        # person centre (original pixels)
            rec = recorded(vid, a, b)
            stat = {'n': 0, 'raw': 0, 'cmd': 0}
            for i, img in enumerate(frames(vid, a, b)):
                f = a + i
                ramp = s == 0 and i < RAMP and k > 0
                cur = prev + (target - prev) * (i + 1) / RAMP if ramp else target
                zoom = max(1.0, rec / cur)
                h0, w0 = img.shape[:2]
                cw, ch = w0 / zoom, h0 / zoom
                cx, cy = min(max(cx0, cw / 2), w0 - cw / 2), min(max(cy0, ch / 2), h0 - ch / 2)
                x0, y0 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
                img = cv2.resize(img[y0:y0 + int(ch), x0:x0 + int(cw)], (W, H), interpolation=cv2.INTER_AREA)
                if mirror:
                    img = np.ascontiguousarray(img[:, ::-1])
                frame, people, raw, best = pipe.process(img, None)
                cmd = tf.update(raw)
                jc = cats[f] if f < len(cats) else None
                jc = SWAPG.get(jc, jc) if mirror else jc
                want = GESTURES[jc][2] if jc in GESTURES else None
                if want:
                    stat['n'], stat['raw'], stat['cmd'] = stat['n'] + 1, stat['raw'] + (raw == want), \
                        stat['cmd'] + (cmd == want)
                view = frame.copy()
                draw(view, people)
                put(view, f'Direction (command): {cmd}', (16, 40), 1.0, (0, 0, 255))
                put(view, (f'UAV approaching: {prev:.1f} -> {target:.1f} m' if ramp else
                           f'Stop {k + 1} of {len(APPROACH)}: UAV at ~{rec / zoom:.1f} m'), (W - 470, 40), 0.9, WHITE)
                bar = np.full((BAR, W, 3), 32, np.uint8)
                put(bar, f"REAL  |  UAV-GESTURE, person S{v['subject']}, signal '{v['cls']}'"
                         + ('  |  MIRRORED left-right' if mirror else '') + '  |  recorded video, no flight',
                    (16, 30), 0.6, WHITE, 1)
                put(bar, f'Approach emulated by digital zoom: recorded at {rec:.1f} m, zoom x{zoom:.2f} '
                         f'-> person at ~{rec / zoom:.1f} m', (16, 56), 0.6, WHITE, 1)
                put(bar, 'Algorithm 1 (stock YOLOv11n-pose, no training) on every RGB video frame (25 fps)',
                    (16, 82), 0.6, WHITE, 1)
                if want:
                    put(bar, f'Person shows: {GESTURES[jc][1]}  ->  expected: {want} (UAV: {ACTION[want]})',
                        (16, 120), 0.65, CYAN)
                else:
                    put(bar, 'Person shows: moving the arm (between poses, not scored)', (16, 120), 0.65, GREY)
                put(bar, '(pose of this frame from the dataset\'s annotated joints)', (16, 146), 0.5, GREY, 1)
                put(bar, f'Algorithm 1, this frame: {raw}', (16, 184), 0.7, ok_colour(raw, want) if want else GREY)
                put(bar, f'Command (5 of 7): {cmd}', (450, 184), 0.7, ok_colour(cmd, want) if want else GREY)
                put(bar, f'A UAV would: {ACTION.get(cmd, "hover")}', (850, 184), 0.7, WHITE)
                for r, line in enumerate(RULE):
                    put(bar, line, (16, 222 + 20 * r), 0.45, GREY, 1)
                enc.stdin.write(np.vstack([view, bar]).tobytes())
            for key in total:
                total[key] += stat[key]
            print(f"stop {k + 1} {g:8s} S{v['subject']} {vid:18s} recorded {rec:.1f} m: {stat['n']} scored frames, "
                  f"Algorithm 1 {100 * stat['raw'] / max(stat['n'], 1):.0f}%, "
                  f"command {100 * stat['cmd'] / max(stat['n'], 1):.0f}%", flush=True)
        prev = target
    enc.stdin.close()
    enc.wait()
    print(f"{out}: {total['n']} scored frames, Algorithm 1 {100 * total['raw'] / total['n']:.0f}%, "
          f"command {100 * total['cmd'] / total['n']:.0f}%", flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'synthetic':
        synthetic(*sys.argv[2:6])
    elif sys.argv[1] == 'approach':
        approach(sys.argv[2])
    else:
        real(sys.argv[2], sys.argv[3])
