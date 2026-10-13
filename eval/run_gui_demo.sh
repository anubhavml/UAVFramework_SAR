#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
# Live simulation on the desktop: every component in its own terminal window.
#   1 Micro XRCE-DDS Agent | 2 Gazebo server (headless) + Gazebo GUI | 3 PX4 SITL (pxh shell) |
#   QGroundControl (optional) | 4 ROS 2 bridge + gesture node + control node (drone camera window) |
#   5 demo driver (moves the UAV towards the person, makes the person show the gestures)
# The gesture node runs Algorithm 1 of the paper (both arms out -> hover; left / right as the UAV
# sees them) on the pipeline chosen with --mode.
#
#   ./run_simulation.sh                                    start window: choose the options there
#   ./run_simulation.sh [options]                          stop: ./stop_simulation.sh
#     --source synthetic|real       synthetic: Gazebo person; real: UAV-GESTURE videos as the
#                                   drone's camera stream (the simulated UAV reacts to them;
#                                   needs the dataset, see README)
#     --mode rgb|depth|fused|rgbd   rgb (default) / depth: Algorithm 1 with the stock YOLOv11n-pose
#                                   on the RGB / depth image; fused: on the depth-weighted fused
#                                   RGB-D pose of Eq. (1)-(2); rgbd: YOLOv11-L_total on RGB + depth
#                                   (--model). Real videos: depth estimated from each frame
#     --model PATH                  weights for --mode rgbd (default
#                                   results/weights/yolov11_ltotal_synthetic.pt, or
#                                   train/runs/T4/weights/best.pt after training)
#     --world city|forest|wall|city_fog|forest_fog|wall_fog   Gazebo world (default city); *_fog:
#                                   Gazebo particle fog, as in the paper's fog worlds
#     --fog-density 1.0             *_fog worlds: particle fog density, x the paper's (1.0:
#                                   steers only at ~3 m; 0.5 up to ~5 m; 0.25 almost clear)
#     --visibility V                camera fog (physically based, Koschmieder) with visibility V m on
#                                   the camera images, as in the paper's fog tests (V = 50, 20, 10,
#                                   5); 0 = off. Seen in the drone camera window, not in Gazebo;
#                                   real videos too
#     --distances 10,7,5,3          the UAV starts at the first distance and moves towards the
#                                   person, stopping at each (default 10,7,5,3; real: 6.5,5,3.5).
#                                   real: the approach is emulated by digital zoom of the video
#                                   (recorded at 6.5-8.4 m, so the person never appears farther)
#     --gestures right,left,both_out   shown at every stop; any of: right left neutral both_out
#                                   both_up rightout_leftup leftout_rightup
#     --once                        one pass instead of repeating
#     --react turn|fly              turn (default, paper): the UAV turns towards the pointed side
#                                   while the arm is out, up to 20 deg; fly: it turns --fly-turn deg,
#                                   flies --fly-distance m that way and waits for "come back"
#                                   (shortened / skipped if trees, buildings or walls are within
#                                   1.5 m of the path)
#     --fly-turn 90                 deg (react fly)
#     --fly-distance 5              m (react fly)
#     --panel                       no automatic sequence: a "Demo control" window with buttons to
#                                   move the UAV forward / back, choose the gesture the person shows
#                                   (the UAV reacts) and the reaction
#     --no-qgc                      do not start QGroundControl
#     --no-terminals                run the components in the background (logs in
#                                   eval/data/gui_demo/logs) instead of one terminal window each
#   Environment: SAR_TERMINAL (gnome-terminal|konsole|xfce4-terminal|xterm), QGC_APPIMAGE (path to
#   a QGroundControl AppImage), SAR_VGL=0 (do not use VirtualGL even if installed).
#   e.g. ./run_simulation.sh --world forest --mode depth
#        ./run_simulation.sh --world city_fog --fog-density 0.5 --mode fused
#        ./run_simulation.sh --visibility 20 --mode rgbd
#        ./run_simulation.sh --source real --mode fused --visibility 20
#        ./run_simulation.sh --panel --react fly

case "${1:-}" in -h|--help) awk 'NR > 4 && /^#/ {print; next} NR > 4 {exit}' "$0"; exit 0 ;; esac
EVAL=$(cd "$(dirname "$0")" && pwd)
WS=$(cd "$EVAL/.." && pwd)

# --- desktop: use $DISPLAY if it works, else look for a local X display (e.g. when started over ssh)
x_ok() { python3 -c "import ctypes, sys; x = ctypes.cdll.LoadLibrary('libX11.so.6'); x.XOpenDisplay.restype = ctypes.c_void_p; sys.exit(0 if x.XOpenDisplay(None) else 1)" 2>/dev/null; }
find_display() {
  [ -n "${DISPLAY:-}" ] && x_ok && return 0
  local d a
  for d in $(ls /tmp/.X11-unix 2>/dev/null | sed -n 's/^X\([0-9]*\)$/:\1/p'); do
    for a in "${XAUTHORITY:-}" "$HOME/.Xauthority" "/run/user/$(id -u)/gdm/Xauthority"; do
      [ -n "$a" ] && [ ! -f "$a" ] && continue
      DISPLAY=$d XAUTHORITY=$a x_ok && { export DISPLAY=$d XAUTHORITY=$a; return 0; }
    done
  done
  return 1
}
find_display || { echo "ABORT: no X display found (log in to a desktop session, or set DISPLAY)"; exit 1; }
[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ] && [ -S "/run/user/$(id -u)/bus" ] && \
  export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus

if [ $# -eq 0 ]; then                       # no options: the start window (eval/demo_launcher.py)
  cd "$EVAL" && exec python3 demo_launcher.py
fi
W=city; MODE=rgb; SRC=synthetic; DIST=""; GEST=right,left,both_out; ONCE=""
REACT=turn; FTURN=90; FDIST=5; PANEL=""; FOGD=1.0; VIS=0; TERMS=yes; QGC_ON=yes
MODEL=$WS/results/weights/yolov11_ltotal_synthetic.pt              # or the model trained with train/train.py T4
[ -f "$MODEL" ] || MODEL=$WS/train/runs/T4/weights/best.pt
[ -f "$MODEL" ] || MODEL=$WS/results/weights/yolov11_ltotal_synthetic.pt
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$(realpath -m "$2"); shift 2 ;;
    --no-qgc) QGC_ON=no; shift ;;
    --world) W=$2; shift 2 ;;
    --mode) MODE=$2; shift 2 ;;
    --source) SRC=$2; shift 2 ;;
    --distances) DIST=$2; shift 2 ;;
    --gestures) GEST=$2; shift 2 ;;
    --once) ONCE=--once; shift ;;
    --react) REACT=$2; shift 2 ;;
    --fly-turn) FTURN=$2; shift 2 ;;
    --fly-distance) FDIST=$2; shift 2 ;;
    --panel) PANEL=--panel; shift ;;
    --fog-density) FOGD=$2; shift 2 ;;
    --visibility) VIS=$2; shift 2 ;;
    --no-terminals) TERMS=no; shift ;;
    city|forest|wall|*_fog) W=$1; shift ;;                     # short form: run_gui_demo.sh <world>
    -h|--help) awk 'NR > 4 && /^#/ {print; next} NR > 4 {exit}' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 1 ;;
  esac
done
case $MODE in rgb|depth|fused|rgbd) ;; *) echo "--mode must be rgb, depth, fused or rgbd"; exit 1 ;; esac
[ $MODE = rgbd ] && [ ! -f "$MODEL" ] && { echo "ABORT: --mode rgbd needs the YOLOv11-L_total weights ($MODEL)"; exit 1; }
case $SRC in synthetic|real) ;; *) echo "--source must be synthetic or real"; exit 1 ;; esac
case $REACT in fly|turn) ;; *) echo "--react must be fly or turn"; exit 1 ;; esac
[ -z "$DIST" ] && DIST=$([ $SRC = synthetic ] && echo 10,7,5,3 || echo 6.5,5,3.5)
SPAWN=${DIST%%,*}                                                # UAV starts at the first distance
export SAR_GESTURES=right,left,neutral,both_out,both_up,rightout_leftup,leftout_rightup
source "$WS/px4_env.sh" >/dev/null 2>&1
export GZ_SIM_RESOURCE_PATH="$EVAL/models:$GZ_SIM_RESOURCE_PATH"
if pgrep -f "^(gz sim|MicroXRCEAgent)|px4_sitl_default/bin/px4" >/dev/null; then
  echo "ABORT: a simulation is already running (stop it with ./stop_simulation.sh)"; exit 1; fi
[ -x "$PX4_DIR/build/px4_sitl_default/bin/px4" ] || {
  echo "ABORT: PX4 SITL is not built ($PX4_DIR/build/px4_sitl_default); see README.md"; exit 1; }
DEMO=$EVAL/data/gui_demo; mkdir -p "$DEMO/logs"

# --- GPU: YOLO on GPU 0 if there is a CUDA GPU, else on the CPU
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then DEVICE=0; else DEVICE=cpu; fi
# --- VirtualGL (optional): draws the Gazebo GUI and QGroundControl on GPU 0 through EGL when the
# desktop itself has no hardware OpenGL (e.g. a remote desktop on a server's onboard graphics)
VGL=""
if [ "${SAR_VGL:-1}" != 0 ] && command -v vglrun >/dev/null && [ $DEVICE = 0 ]; then
  GPU0=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i 0 | tr 'A-F' 'a-f' | sed 's/^0000//')
  CARD=$(readlink -f /dev/dri/by-path/pci-0000${GPU0#0000}-card 2>/dev/null || readlink -f /dev/dri/by-path/pci-${GPU0}-card 2>/dev/null)
  [ -n "$CARD" ] && [ -e "$CARD" ] && VGL="vglrun -d $CARD"
fi
echo "YOLO on: $([ $DEVICE = 0 ] && echo "GPU 0" || echo CPU);  GUI windows: $([ -n "$VGL" ] && echo "VirtualGL ($VGL)" || echo "desktop OpenGL")"
# --- QGroundControl (optional)
QGC=${QGC_APPIMAGE:-}
if [ -z "$QGC" ] && [ $QGC_ON = yes ]; then
  for f in "$WS/external/QGroundControl-4.4.4.AppImage" "$WS/external/QGroundControl.AppImage" "$WS"/external/QGroundControl*.AppImage; do
    [ -x "$f" ] && { QGC=$f; break; }
  done
fi
# --- one terminal window per component (any of these terminals), or background processes
TERM_APP=${SAR_TERMINAL:-}
if [ $TERMS = yes ] && [ -z "$TERM_APP" ]; then
  for t in gnome-terminal konsole xfce4-terminal xterm; do command -v $t >/dev/null && { TERM_APP=$t; break; }; done
fi
[ $TERMS = yes ] && [ -z "$TERM_APP" ] && { echo "no terminal emulator found: components run in the background"; TERMS=no; }
cd "$EVAL"
sdf=$(python3 -c "import simtools as s; print(s.make_eval_world('$W', '$EVAL/data/worlds', rig=False, suffix='_gui'))")
# particle fog density: the emitter rate (steady-state particle count) x --fog-density
python3 - "$sdf" "$FOGD" <<'PY'
import re, sys
p, k = sys.argv[1], float(sys.argv[2])
s = open(p).read()
s = re.sub(r'(<particle_emitter.*?<rate>)([\d.]+)(</rate>)', lambda m: f'{m.group(1)}{float(m.group(2)) * k:g}{m.group(3)}', s, flags=re.S)
open(p, 'w').write(s)
PY
pose=$(python3 -c "import simtools as s; _, (x, y, _), _ = s.world_info('$W'); print(f'{x - $SPAWN:.2f},{y:.2f},0.2,0,0,0')")
ENV="source $WS/px4_env.sh >/dev/null 2>&1; export SAR_GESTURES=$SAR_GESTURES GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH;"
# SAR_DEMO_WINDOW / SAR_DEMO_BG mark these processes so stop_gui_demo.sh can close them
term() {
  local title=$1 cmd="$ENV $2"
  if [ $TERMS = no ]; then
    local log="$DEMO/logs/$(echo "$title" | tr -c 'A-Za-z0-9\n' '_').log"
    setsid bash -c "$cmd > '$log' 2>&1 # SAR_DEMO_BG" < /dev/null > /dev/null 2>&1 &
    return
  fi
  local wrap="$cmd; echo; echo '[$title ended - press Enter to close]'; read # SAR_DEMO_WINDOW"
  case $TERM_APP in
    gnome-terminal) gnome-terminal --title="$title" -- bash -c "$wrap" ;;
    konsole) setsid konsole -p tabtitle="$title" -e bash -c "$wrap" > /dev/null 2>&1 & ;;
    xfce4-terminal) xfce4-terminal --title="$title" -x bash -c "$wrap" ;;
    *) setsid $TERM_APP -T "$title" -e bash -c "$wrap" > /dev/null 2>&1 & ;;
  esac
}
term "1 Micro XRCE-DDS Agent" "MicroXRCEAgent udp4 -p 8888"
# Gazebo server headless (GPU through EGL), as in every recorded flight; the GUI attaches as a
# separate client, so the cameras are rendered on the GPU even when the desktop's OpenGL is not
term "2 Gazebo server (headless)" "gz sim -s -r --headless-rendering $sdf"
sleep 8
term "2 Gazebo (GUI)" "$VGL gz sim -g"
sleep 15
term "3 PX4 SITL" "cd $PX4_DIR && PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4002 PX4_SIM_MODEL=gz_x500_depth PX4_GZ_WORLD=$W PX4_GZ_MODEL_POSE=$pose ./build/px4_sitl_default/bin/px4"
sleep 15
# barometer height for PX4's height estimate (RUN_SAR.md, "Height estimate")
( cd "$PX4_DIR/build/px4_sitl_default/bin" && ./px4-param set EKF2_HGT_REF 0 && ./px4-param set EKF2_GPS_CTRL 5 &&
  ./px4-param set EKF2_BARO_NOISE 0.2 && ./px4-ekf2 stop && sleep 1 && ./px4-ekf2 start ) > "$DEMO/px4_params.log" 2>&1
if [ -n "$QGC" ] && [ $QGC_ON = yes ]; then
  setsid $VGL "$QGC" > "$DEMO/qgc.log" 2>&1 < /dev/null &
  sleep 10
elif [ $QGC_ON = yes ]; then
  echo "QGroundControl not found (optional): put QGroundControl-4.4.4.AppImage in external/ or set QGC_APPIMAGE"
fi
SENSOR=$([ $SRC = real ] && echo video || echo oakd)
term "4 ROS 2 gesture + control nodes" "ros2 launch drone_control sar_hitl.launch.py show:=true robust:=false both_arms_hover:=true uav_view:=true max_turn:=20 device:=$DEVICE input_mode:=$MODE $([ $MODE = rgbd ] && echo model:=$MODEL) sensor:=$SENSOR react:=$REACT fly_turn:=$FTURN fly_distance:=$FDIST visibility:=$VIS $([ -n "$PANEL" ] && echo window_width:=1040)"
sleep 20
term "5 Demo driver ($SRC, $MODE)" "cd $EVAL && python3 demo_flight.py $W $DEMO --source $SRC --mode $MODE --distances $DIST --gestures $GEST $ONCE --react $REACT --fly-turn $FTURN --fly-distance $FDIST --visibility $VIS --fog-density $FOGD $PANEL 2>&1 | grep -v -i warn"
echo "simulation started on $DISPLAY: world $W, source $SRC, mode $MODE, reaction $REACT," \
     "$([ -n "$PANEL" ] && echo "button panel" || echo "gestures $GEST"), distances $DIST m" \
     "$([ $SRC = real ] && echo "(approach emulated by digital zoom)")"
echo "stop with ./stop_simulation.sh"
