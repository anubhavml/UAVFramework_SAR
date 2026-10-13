#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Analyse gesture-switch flights (data/flights/*/log.jsonl) -> data/flights/FLIGHTS.md
(or another folder of flights: flight_report.py <folder>)

For each switch of the person's gesture (sim time t0):
  command latency   first filtered /gesture/command equal to the new gesture
  motion latency    RIGHT/LEFT: heading has moved >= 2 deg the expected way (RIGHT -> drone yaws
                    left, heading decreases; LEFT -> increases); NEUTRAL: heading stays within
                    +-1 deg for the next 1 s (not measured if it had already stopped at the
                    max_turn limit before the switch)
  wrong commands    fraction of commands during the phase that are LEFT/RIGHT but not expected
Latencies are given in simulation time (what the vehicle experiences) and wall-clock time.
"""
import glob
import json
import os
import sys

import numpy as np

EVAL = os.path.dirname(os.path.abspath(__file__))
EXPECT = {'right': 'RIGHT', 'left': 'LEFT', 'neutral': 'NEUTRAL'}
SIGN = {'right': -1, 'left': +1, 'neutral': 0}


def wrap(a):
    return (a + 180) % 360 - 180


def turning_at(pos, t0, window=1.0, min_deg=1.0):
    """True if the heading changed by more than min_deg during the window before t0."""
    hs = [r['data'][3] for r in pos if t0 - window <= r['sim'] <= t0]
    return len(hs) > 1 and max(abs(wrap(h - hs[-1])) for h in hs) > min_deg


def analyse(path):
    rows = [json.loads(l) for l in open(path)]
    if any(r['kind'] == 'error' for r in rows):
        return dict(error=[r['data'] for r in rows if r['kind'] == 'error'][0])
    swaps = [r for r in rows if r['kind'] == 'swap'][1:]          # skip the pre-takeoff one
    pos = [r for r in rows if r['kind'] == 'pos']
    cmds = [r for r in rows if r['kind'] == 'cmd']
    tim = np.array([r['data'] for r in rows if r['kind'] == 'timing'], float)
    ps, pw = np.array([r['sim'] for r in pos]), np.array([r['wall'] for r in pos])
    out, prev = [], 'neutral'
    for i, s in enumerate(swaps):
        # the swap is stamped with the sim time Gazebo first reported the new model in place;
        # its wall time is interpolated from the position samples
        g, t0 = s['data'], s['sim']
        w0 = float(np.interp(t0, ps, pw))
        t_end = swaps[i + 1]['sim'] if i + 1 < len(swaps) else rows[-1]['sim']
        if g == prev:
            prev = g
            continue
        res = dict(switch=f'{prev}->{g}')
        c = next((r for r in cmds if r['sim'] >= t0 and r['data'] == EXPECT[g]), None)
        res['cmd_sim'] = c['sim'] - t0 if c else None
        res['cmd_wall'] = c['wall'] - w0 if c else None
        h0 = next((r['data'][3] for r in reversed(pos) if r['sim'] <= t0), None)
        ph = [r for r in pos if t0 <= r['sim'] <= t_end]
        m = None
        if SIGN[g] and h0 is not None:
            m = next((r for r in ph if SIGN[g] * wrap(r['data'][3] - h0) >= 2.0), None)
        elif h0 is not None and turning_at(pos, t0):  # else already stopped (turn limit)
            for j, r in enumerate(ph):
                win = [q['data'][3] for q in ph[j:] if q['sim'] <= r['sim'] + 1.0]
                if ph[-1]['sim'] >= r['sim'] + 1.0 and max(abs(wrap(h - win[0])) for h in win) <= 1.0:
                    m = r
                    break
        res['move_sim'] = m['sim'] - t0 if m else None
        res['move_wall'] = m['wall'] - w0 if m else None
        if SIGN[g] and len(ph) > 2:                       # turn rate while turning (sim time)
            hs = np.array([[r['sim'], np.unwrap(np.radians([q['data'][3] for q in ph]))[k]]
                           for k, r in enumerate(ph)])
            moving = np.abs(np.gradient(hs[:, 1], hs[:, 0])) > np.radians(3)
            res['rate_deg_s'] = (float(np.degrees(np.median(np.abs(np.gradient(
                hs[moving, 1], hs[moving, 0]))))) if moving.sum() > 3 else None)
            res['turned_deg'] = float(wrap(ph[-1]['data'][3] - h0))
        hz = [r['data'][4] for r in ph if len(r['data']) > 4 and r['data'][4] is not None]
        res['height_m'] = float(np.mean(hz)) if hz else None     # true height (Gazebo)
        pc = [r['data'] for r in cmds if t0 <= r['sim'] <= t_end]
        res['wrong_cmd'] = (sum(1 for d in pc if d in ('LEFT', 'RIGHT') and d != EXPECT[g])
                            / len(pc) if pc else None)
        out.append(res)
        prev = g
    walls = [r['wall'] for r in rows if r['sim'] is not None]
    sims = [r['sim'] for r in rows if r['sim'] is not None]
    return dict(switches=out, rtf=(sims[-1] - sims[0]) / (walls[-1] - walls[0]),
                proc_ms=float(np.median(tim[:, 0]) * 1000) if len(tim) else None,
                proc_p95_ms=float(np.percentile(tim[:, 0], 95) * 1000) if len(tim) else None,
                frame_age_ms=float(np.nanmedian(tim[:, 1]) * 1000) if len(tim) else None)


def fmt(x, unit=' s', nd=2):
    return '–' if x is None else f'{x:.{nd}f}{unit}'


def main():
    out = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else f'{EVAL}/data/flights'
    runs = sorted(glob.glob(f'{out}/*/log.jsonl'))
    L = ['# Gesture-switch flights', '',
         'Person 5 m in front of the hovering drone; gesture switched at known simulation times. '
         'Latencies: simulation time (wall-clock time in brackets). Height: true mean height of the '
         'drone during the phase (Gazebo; take-off setpoint 1.5 m). See flight_report.py.', '',
         '| Run | Switch | Command latency | Motion latency | Turn rate | Turned | Wrong cmds | Height |',
         '|---|---|---|---|---|---|---|---|']
    allres = {}
    for p in runs:
        name = os.path.basename(os.path.dirname(p))
        r = analyse(p)
        allres[name] = r
        if 'error' in r:
            L.append(f'| {name} | {r["error"]} | | | | | | |')
            continue
        for s in r['switches']:
            L.append(f"| {name} | {s['switch']} | {fmt(s['cmd_sim'])} ({fmt(s['cmd_wall'])}) | "
                     f"{fmt(s['move_sim'])} ({fmt(s['move_wall'])}) | "
                     f"{fmt(s.get('rate_deg_s'), ' °/s', 1)} | {fmt(s.get('turned_deg'), '°', 0)} | "
                     f"{fmt(None if s['wrong_cmd'] is None else 100 * s['wrong_cmd'], '%', 0)} | "
                     f"{fmt(s.get('height_m'), ' m', 2)} |")
    L += ['', '| Run | RTF | Processing / frame (median, p95, wall) | Frame age at decision (sim) |',
          '|---|---|---|---|']
    for name, r in allres.items():
        if 'error' not in r:
            L.append(f"| {name} | {r['rtf']:.2f} | {fmt(r['proc_ms'], ' ms', 0)}, "
                     f"{fmt(r['proc_p95_ms'], ' ms', 0)} | {fmt(r['frame_age_ms'], ' ms', 0)} |")
    open(f'{out}/FLIGHTS.md', 'w').write('\n'.join(L) + '\n')
    json.dump(allres, open(f'{out}/flights.json', 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    sys.exit(main())
