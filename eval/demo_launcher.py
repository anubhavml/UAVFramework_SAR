#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Start window of the desktop demonstration (eval/run_gui_demo.sh without options, or the
"SAR UAV Demo" desktop icon).

Pick the person (synthetic Gazebo person / real UAV-GESTURE videos), the perception pipeline
(Algorithm 1 on RGB, depth or the fused RGB-D pose; YOLOv11-L_total on RGB-D), the world, the fog,
the distances and the UAV's reaction, then "Start demo": it runs run_gui_demo.sh with
these options (--panel) and hides itself once the "Demo control" panel is open.  When the demo is
stopped (the panel's "Stop demo" button, or eval/stop_gui_demo.sh) it comes back.
"""
import os
import signal
import subprocess
import tkinter as tk

import xwin

HERE = os.path.dirname(os.path.abspath(__file__))
RUN, STOP = f'{HERE}/run_gui_demo.sh', f'{HERE}/stop_gui_demo.sh'
LOG = f'{HERE}/data/gui_demo/launcher.log'
SIM = r'^(gz sim|MicroXRCEAgent)|px4_sitl_default/bin/px4'     # as run_gui_demo.sh checks
DRIVER = r'^python3 demo_flight\.py '
DEFAULT_DIST = {'synthetic': '10,7,5,3', 'real': '6.5,5,3.5'}


def running(pattern):
    return subprocess.run(['pgrep', '-f', pattern], capture_output=True).returncode == 0


class Launcher:
    def __init__(self):
        self.root = root = tk.Tk()
        root.title('SAR UAV demo: start')
        root.configure(padx=12, pady=10)
        F, B = ('DejaVu Sans', 11), ('DejaVu Sans', 11, 'bold')
        self.F = F
        self.v = dict(source=tk.StringVar(value='synthetic'), mode=tk.StringVar(value='rgb'),
                      world=tk.StringVar(value='city'), fog=tk.StringVar(value='none'),
                      fog_density=tk.DoubleVar(value=0.5), visibility=tk.DoubleVar(value=20),
                      dist=tk.StringVar(value=DEFAULT_DIST['synthetic']),
                      react=tk.StringVar(value='turn'), fly_turn=tk.DoubleVar(value=90),
                      fly_distance=tk.DoubleVar(value=5))

        def section(text):
            f = tk.LabelFrame(root, text=text, font=B, padx=8, pady=6)
            f.pack(fill='x', pady=4)
            return f

        def radios(parent, var, items, cmd=None):
            for i, (val, txt) in enumerate(items):
                tk.Radiobutton(parent, text=txt, font=F, variable=var, value=val,
                               command=cmd).grid(row=0, column=i, sticky='w', padx=(0, 12))

        f = section('Person (who shows the gestures)')
        radios(f, self.v['source'], [('synthetic', 'Synthetic person in Gazebo'),
                                     ('real', 'Real people (UAV-GESTURE videos)')], self.source_changed)
        f = section('Perception (gesture node)')
        for i, (val, txt) in enumerate([
                ('rgb', 'Algorithm 1, stock YOLOv11n-pose on the RGB image'),
                ('depth', 'Algorithm 1, stock YOLOv11n-pose on the depth image'),
                ('fused', 'Algorithm 1 on the depth-weighted fused RGB-D pose, Eq. (1)-(2)'),
                ('rgbd', 'YOLOv11-L_total on RGB + depth (synthetic-trained weights)')]):
            tk.Radiobutton(f, text=txt, font=F, variable=self.v['mode'], value=val).grid(row=i, column=0, sticky='w')
        f = section('World')
        radios(f, self.v['world'], [('city', 'City'), ('forest', 'Forest'), ('wall', 'Wall')])
        f = section('Fog')
        self.fog_radios = {}
        for i, (val, txt) in enumerate([
                ('none', 'No fog'),
                ('particle', 'Gazebo particle fog (seen in Gazebo and by the camera; the paper\'s fog worlds)'),
                ('camera', 'Camera fog by visibility (physically based, on the camera image only)')]):
            self.fog_radios[val] = tk.Radiobutton(f, text=txt, font=F, variable=self.v['fog'], value=val,
                                                  command=self.fog_changed)
            self.fog_radios[val].grid(row=i, column=0, columnspan=3, sticky='w')
        self.dens_lbl = tk.Label(f, text='density (x paper)', font=F)
        self.dens_lbl.grid(row=3, column=0, sticky='w')
        self.dens = tk.Scale(f, from_=0.1, to=1.5, resolution=0.05, orient='horizontal', length=280,
                             variable=self.v['fog_density'])
        self.dens.grid(row=3, column=1, sticky='w')
        self.vis_lbl = tk.Label(f, text='visibility (m)', font=F)
        self.vis_lbl.grid(row=4, column=0, sticky='w')
        self.vis = tk.Scale(f, from_=5, to=100, resolution=1, orient='horizontal', length=280,
                            variable=self.v['visibility'])
        self.vis.grid(row=4, column=1, sticky='w')
        self.fog_note = tk.Label(f, font=('DejaVu Sans', 9), justify='left', anchor='w', wraplength=540)
        self.fog_note.grid(row=5, column=0, columnspan=3, sticky='w')
        f = section('Distances to the person (m)')
        tk.Entry(f, textvariable=self.v['dist'], font=F, width=18).grid(row=0, column=0, sticky='w')
        self.dist_note = tk.Label(f, font=('DejaVu Sans', 9), justify='left', anchor='w')
        self.dist_note.grid(row=1, column=0, columnspan=3, sticky='w')
        f = section('UAV reaction to LEFT / RIGHT (also changeable in the panel)')
        tk.Radiobutton(f, text='Turn only, up to 20 deg while the arm is out (paper)', font=F,
                       variable=self.v['react'], value='turn').grid(row=0, column=0, columnspan=2, sticky='w')
        tk.Radiobutton(f, text='Turn, fly there and wait for "come back"', font=F, variable=self.v['react'],
                       value='fly').grid(row=1, column=0, columnspan=2, sticky='w')
        tk.Label(f, text='turn (deg)', font=F).grid(row=2, column=0, sticky='w')
        tk.Scale(f, from_=20, to=90, resolution=5, orient='horizontal', length=280,
                 variable=self.v['fly_turn']).grid(row=2, column=1, sticky='w')
        tk.Label(f, text='fly (m)', font=F).grid(row=3, column=0, sticky='w')
        tk.Scale(f, from_=1, to=8, resolution=0.5, orient='horizontal', length=280,
                 variable=self.v['fly_distance']).grid(row=3, column=1, sticky='w')

        bf = tk.Frame(root)
        bf.pack(fill='x', pady=(8, 2))
        self.start_btn = tk.Button(bf, text='Start demo', font=B, bg='#b9e3b0', width=14, command=self.start)
        self.start_btn.pack(side='left')
        self.stop_btn = tk.Button(bf, text='Stop running demo', font=F, command=self.stop)
        self.stop_btn.pack(side='left', padx=8)
        tk.Button(bf, text='Quit', font=F, command=root.destroy).pack(side='right')
        self.status = tk.Label(root, font=F, justify='left', anchor='w', wraplength=560)
        self.status.pack(fill='x', pady=4)
        self.source_changed()
        self.fog_changed()
        self.state, self.proc = 'idle', None
        if running(SIM) or running(DRIVER):
            self.state = 'running'
            self.say('A demo is already running. "Stop running demo" closes it.')
        root.after(1000, self.poll)

    def say(self, text):
        self.status.config(text=text)

    def source_changed(self):
        src = self.v['source'].get()
        if hasattr(self, 'fog_radios'):            # real videos are not rendered by Gazebo
            self.fog_radios['particle'].config(state='disabled' if src == 'real' else 'normal')
            if src == 'real' and self.v['fog'].get() == 'particle':
                self.v['fog'].set('camera')
                self.fog_changed()
        if self.v['dist'].get() in DEFAULT_DIST.values():
            self.v['dist'].set(DEFAULT_DIST[src])
        self.dist_note.config(text='the UAV starts at the first, the others are "go to" buttons'
                              + ('; real videos: the approach is emulated by zoom, recorded at 6.5-8.4 m'
                                 if src == 'real' else ''))

    def fog_changed(self):
        fog = self.v['fog'].get()
        for w in (self.dens_lbl, self.dens):
            w.config(state='normal' if fog == 'particle' else 'disabled')
        for w in (self.vis_lbl, self.vis):
            w.config(state='normal' if fog == 'camera' else 'disabled')
        self.fog_note.config(text={
            'none': '',
            'particle': "Measured (city, RGB): x1.0 = the paper's fog test, steers only at ~3 m; "
                        'x0.5 up to ~5 m; x0.25 almost like clear air (5 m, partly 7 m). '
                        'Builds up in the first ~30 s. Use Forward to get close.',
            'camera': 'V = distance at which contrast falls to 2 % (Koschmieder), as the paper\'s '
                      '"fog V=..." tests. Measured (city, RGB): V 30 or more like clear air (to 5-7 m); '
                      '20 up to 5 m; 10 up to 3 m; 5 nothing.'}[fog])

    def options(self):
        v = {k: x.get() for k, x in self.v.items()}
        dist = [float(x) for x in v['dist'].replace(' ', '').split(',') if x]
        if not dist or min(dist) < 2.0 or max(dist) > 20.0:
            raise ValueError('distances must be between 2 and 20 m, e.g. 10,7,5,3')
        world = v['world'] + ('_fog' if v['fog'] == 'particle' else '')
        fog = (['--fog-density', f"{v['fog_density']:g}"] if v['fog'] == 'particle' else []) + \
              (['--visibility', f"{v['visibility']:g}"] if v['fog'] == 'camera' else [])
        return ['--world', world, *fog, '--mode', v['mode'], '--source', v['source'],
                '--distances', ','.join(f'{d:g}' for d in dist), '--react', v['react'],
                '--fly-turn', f"{v['fly_turn']:g}", '--fly-distance', f"{v['fly_distance']:g}", '--panel']

    def start(self):
        if running(SIM) or running(DRIVER):
            self.say('A demo is still running: press "Stop running demo" first.')
            return
        try:
            opts = self.options()
        except ValueError as e:
            self.say(f'Check the options: {e}')
            return
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        self.proc = subprocess.Popen(['bash', RUN, *opts], stdout=open(LOG, 'w'),
                                     stderr=subprocess.STDOUT, start_new_session=True)
        self.state = 'starting'
        self.start_btn.config(state='disabled')
        self.say('Starting (about 1.5 min): Micro XRCE-DDS Agent, Gazebo, PX4 SITL, QGroundControl, '
                 'ROS 2 gesture + control nodes, then the "Demo control" panel opens.')

    def stop(self):
        self.say('Stopping the demo ...')
        self.root.update()
        if self.proc is not None and self.proc.poll() is None:     # still starting: stop that first
            os.killpg(self.proc.pid, signal.SIGTERM)
        subprocess.run(['bash', STOP], capture_output=True)
        self.state = 'idle'
        self.start_btn.config(state='normal')
        self.say('Demo stopped.')

    def poll(self):
        if self.state == 'starting':
            log = open(LOG).read() if os.path.exists(LOG) else ''
            if 'ABORT' in log or 'unknown option' in log or (self.proc.poll() not in (None, 0)):
                self.state = 'idle'
                self.start_btn.config(state='normal')
                self.say('Could not start: ' + log.strip().splitlines()[-1] if log.strip() else 'Could not start.')
            elif running(DRIVER) and (lambda w: not w or any(t.startswith('Demo control') for _, t in w))(xwin.windows()):
                self.state = 'running'
                self.root.withdraw()                       # the panel takes over
        elif self.state == 'running':
            if not running(DRIVER):
                if running(SIM):
                    self.say('The demo driver (panel) has ended but the simulation is still running: '
                             '"Stop running demo" closes it.')
                else:
                    self.say('Demo stopped. Choose the options and start again.')
                self.state = 'idle'
                self.start_btn.config(state='normal')
                self.root.deiconify()
                self.root.lift()
        self.root.after(1500, self.poll)


if __name__ == '__main__':
    Launcher().root.mainloop()
