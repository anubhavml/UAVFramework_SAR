#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
"""Desktop window stacking for the GUI demo (X11, no extra packages): find top-level windows by
title and bring them to the front in a given order, as a taskbar/pager would
(_NET_ACTIVE_WINDOW, source = pager), so the component terminals end up behind the main windows.

    python3 xwin.py "Gazebo" "QGroundControl" "Unified-PDD" "Demo control"   (last = on top)
"""
import ctypes
import re
import subprocess
import sys
import time


class _ClientMessage(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int),
                ('display', ctypes.c_void_p), ('window', ctypes.c_ulong),
                ('message_type', ctypes.c_ulong), ('format', ctypes.c_int),
                ('data', ctypes.c_long * 5)]


class _XEvent(ctypes.Union):
    _fields_ = [('xclient', _ClientMessage), ('pad', ctypes.c_long * 24)]


def windows():
    """[(window id, title)] of the managed top-level windows, bottom to top"""
    try:
        out = subprocess.run(['xprop', '-root', '_NET_CLIENT_LIST_STACKING'], capture_output=True, text=True).stdout
    except FileNotFoundError:          # xprop (x11-utils) not installed: no stacking
        return []
    res = []
    for w in re.findall(r'0x[0-9a-f]+', out):
        t = subprocess.run(['xprop', '-id', w, '_NET_WM_NAME', 'WM_NAME'], capture_output=True, text=True).stdout
        m = re.search(r'= "(.*)"', t)
        res.append((int(w, 16), m.group(1) if m else ''))
    return res


def raise_in_order(titles, pause=0.3):
    """bring the windows whose title starts with each of `titles` to the front, in this order
    (so the last one ends up on top); returns the titles found"""
    x11 = ctypes.cdll.LoadLibrary('libX11.so.6')
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    d = x11.XOpenDisplay(None)
    if not d:
        return []
    d = ctypes.c_void_p(d)
    root = x11.XDefaultRootWindow(d)
    active = x11.XInternAtom(d, b'_NET_ACTIVE_WINDOW', 0)
    wins, found = windows(), []
    for t in titles:
        for wid, name in wins:
            if name.startswith(t):
                ev = _XEvent()
                ev.xclient.type, ev.xclient.send_event = 33, 1          # ClientMessage
                ev.xclient.window, ev.xclient.message_type, ev.xclient.format = wid, active, 32
                ev.xclient.data[0] = 2                                   # source: pager
                x11.XSendEvent(d, ctypes.c_ulong(root), 0, ctypes.c_long((1 << 19) | (1 << 20)),
                               ctypes.byref(ev))
                x11.XFlush(d)
                found.append(name)
                time.sleep(pause)
    x11.XCloseDisplay(d)
    return found


if __name__ == '__main__':
    print(raise_in_order(sys.argv[1:]))
