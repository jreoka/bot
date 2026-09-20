"""
Calibration: work out which keys the bot is allowed to press.

The bot's whitelist is not discovered automatically on purpose. A generic agent
that may press anything will eventually press Escape, Alt+Tab, or a debug
overlay key, and at that point it is no longer playing the game - it is
operating the user's computer. So the keys have to be demonstrated once.

This module watches the real keyboard and mouse with Win32 low-level hooks
while the user plays for a minute, and writes down:

* which keys were actually used, and whether each was *held* (a movement key)
  or *tapped* (a menu or slot key) - inferred from how long it was down;
* how far the cursor travels per mouse movement, which becomes the mouse-look
  step size;
* which mouse buttons were used.

Nothing here runs during training, and the injection switch is held off for the
whole session so calibration can never press a key itself.
"""

from __future__ import annotations

import ctypes
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .capture import require_windows
from .keys import KNOWN_KEYS, Keymap, key_name

if os.name == "nt":  # pragma: no cover
    import win32gui

WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
WM_KEYDOWN, WM_KEYUP = 0x0100, 0x0101
WM_SYSKEYDOWN, WM_SYSKEYUP = 0x0104, 0x0105
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN, WM_LBUTTONUP = 0x0201, 0x0202
WM_RBUTTONDOWN, WM_RBUTTONUP = 0x0204, 0x0205
WM_MBUTTONDOWN, WM_MBUTTONUP = 0x0207, 0x0208

_LLKHF_INJECTED = 0x10   # KBDLLHOOKSTRUCT.flags
_LLMHF_INJECTED = 0x01   # MSLLHOOKSTRUCT.flags

_MOUSE_BUTTONS = {"left": "MOUSE_LEFT", "right": "MOUSE_RIGHT",
                  "middle": "MOUSE_MIDDLE"}

# Keys that are never learned and never pressed. F1/F3 are debug overlays in a
# great many games; the Windows key and Alt would take focus away from the game
# and break every subsequent injection.
BLOCKED_VKS = {0x70, 0x72, 0x5B, 0x5C, 0x5D, 0x12, 0x09, 0x1B}


class KeyRecorder:
    """
    Records key presses and mouse movement while the user plays.

    Hooks must be installed, serviced and removed from one thread that pumps a
    Win32 message loop, which is why start()/stop() bracket the whole session
    and pump() must be called regularly.
    """

    def __init__(self, toggle_vk: int = 0x77, target_hwnd: Optional[int] = None,
                 only_when_focused: bool = True):
        require_windows("Key calibration")
        self.toggle_vk = int(toggle_vk)
        self.target_hwnd = target_hwnd
        self.only_when_focused = bool(only_when_focused)

        self.running = False
        self.recording = False
        self.toggles = 0
        self.last_toggle = 0.0
        self.errors: List[str] = []

        # name -> [count, total_held_seconds, longest_hold]
        self.stats: Dict[str, List[float]] = defaultdict(lambda: [0, 0.0, 0.0])
        self.pressed: Dict[str, float] = {}
        self.mouse_pixels: List[float] = []
        self.events = 0
        self.focus_ignored = 0

        self._user32 = ctypes.windll.user32
        self._kb_proc = None
        self._ms_proc = None
        self._kb_handle = None
        self._ms_handle = None
        self._mouse_pos: Optional[Tuple[int, int]] = None

    # ---- lifecycle ----
    def start(self) -> bool:
        if self.running:
            return True
        self.running = True

        def keyboard_proc(code, wparam, lparam):
            try:
                return self._on_key(code, wparam, lparam)
            except Exception:
                return 0

        def mouse_proc(code, wparam, lparam):
            try:
                return self._on_mouse(code, wparam, lparam)
            except Exception:
                return 0

        self._kb_proc = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t,
            ctypes.c_ssize_t)(keyboard_proc)
        self._ms_proc = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t,
            ctypes.c_ssize_t)(mouse_proc)

        for hook_id, proc, attr in ((WH_KEYBOARD_LL, self._kb_proc, "_kb_handle"),
                                    (WH_MOUSE_LL, self._ms_proc, "_ms_handle")):
            try:
                handle = self._user32.SetWindowsHookExW(hook_id, proc, None, 0)
            except Exception as exc:
                handle = None
                self.errors.append(str(exc))
            if not handle:
                self.errors.append(f"hook {hook_id} could not be installed")
            else:
                setattr(self, attr, handle)
        return self._kb_handle is not None

    def stop(self) -> None:
        for name in list(self.pressed):
            self._close_key(name)
        self.running = False
        self.recording = False
        for attr in ("_kb_handle", "_ms_handle"):
            handle = getattr(self, attr)
            if handle:
                try:
                    self._user32.UnhookWindowsHookEx(handle)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._kb_proc = None
        self._ms_proc = None

    def pump(self) -> None:
        """Drain messages so Windows keeps delivering hook callbacks."""
        class _MSG(ctypes.Structure):
            _fields_ = [("hwnd", ctypes.c_void_p), ("message", ctypes.c_uint),
                        ("wParam", ctypes.c_void_p),
                        ("lParam", ctypes.c_void_p), ("time", ctypes.c_uint),
                        ("pt_x", ctypes.c_long), ("pt_y", ctypes.c_long)]

        msg = _MSG()
        while self._user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
            self._user32.TranslateMessage(ctypes.byref(msg))
            self._user32.DispatchMessageW(ctypes.byref(msg))

    # ---- recording ----
    def toggle(self) -> bool:
        self.toggles += 1
        self.last_toggle = time.perf_counter()
        self.recording = not self.recording
        if self.recording:
            self._mouse_pos = None
        else:
            for name in list(self.pressed):
                self._close_key(name)
        return self.recording

    def _wants_input(self) -> bool:
        if not self.only_when_focused or not self.target_hwnd:
            return True
        try:
            return self._user32.GetForegroundWindow() == self.target_hwnd
        except Exception:
            return True

    def _note_press(self, name: str) -> None:
        if name in self.pressed:
            return                       # OS auto-repeat, not a new press
        self.pressed[name] = time.perf_counter()
        self.events += 1

    def _close_key(self, name: str) -> None:
        started = self.pressed.pop(name, None)
        if started is None:
            return
        held = max(0.0, time.perf_counter() - started)
        entry = self.stats[name]
        entry[0] += 1
        entry[1] += held
        entry[2] = max(entry[2], held)

    # ---- hook callbacks ----
    def _on_key(self, code, wparam, lparam):
        if code == 0:
            vk = int(ctypes.cast(lparam, ctypes.POINTER(ctypes.c_uint32))[0])
            flags = int(ctypes.cast(lparam, ctypes.POINTER(ctypes.c_uint32))[1])
            down = wparam in (WM_KEYDOWN, WM_SYSKEYDOWN)
            if vk == self.toggle_vk and down and not (flags & _LLKHF_INJECTED):
                if time.perf_counter() - self.last_toggle > 0.35:
                    self.toggle()
                return self._user32.CallNextHookEx(
                    None, code, wparam, ctypes.c_void_p(lparam))
            if self.recording and not (flags & _LLKHF_INJECTED):
                if not self._wants_input():
                    self.focus_ignored += 1
                elif vk not in BLOCKED_VKS:
                    name = key_name(vk)
                    if down:
                        self._note_press(name)
                    else:
                        self._close_key(name)
        return self._user32.CallNextHookEx(
            None, code, wparam, ctypes.c_void_p(lparam))

    def _on_mouse(self, code, wparam, lparam):
        if code == 0 and self.recording:
            msg = int(wparam)
            if not self._wants_input():
                self.focus_ignored += 1
                return self._user32.CallNextHookEx(
                    None, code, wparam, ctypes.c_void_p(lparam))
            if msg == WM_MOUSEMOVE:
                # MSLLHOOKSTRUCT starts with the cursor position in screen
                # coordinates; the relative delta is what a game reads.
                x = ctypes.cast(lparam, ctypes.POINTER(ctypes.c_long))[0]
                y = ctypes.cast(lparam, ctypes.POINTER(ctypes.c_long))[1]
                if self._mouse_pos is not None:
                    dx = x - self._mouse_pos[0]
                    dy = y - self._mouse_pos[1]
                    if dx or dy:
                        self.mouse_pixels.append(
                            (dx * dx + dy * dy) ** 0.5)
                        self.events += 1
                self._mouse_pos = (x, y)
            else:
                button = {WM_LBUTTONDOWN: ("left", True),
                          WM_LBUTTONUP: ("left", False),
                          WM_RBUTTONDOWN: ("right", True),
                          WM_RBUTTONUP: ("right", False),
                          WM_MBUTTONDOWN: ("middle", True),
                          WM_MBUTTONUP: ("middle", False)}.get(msg)
                if button:
                    name = _MOUSE_BUTTONS[button[0]]
                    if button[1]:
                        self._note_press(name)
                    else:
                        self._close_key(name)
        return self._user32.CallNextHookEx(
            None, code, wparam, ctypes.c_void_p(lparam))

    # ---- results ----
    def is_hold(self, name: str, min_hold: float) -> bool:
        """
        Decide whether a key is a movement key or a momentary one.

        The test is the *typical* hold, not the longest: a key tapped once and
        held once is ambiguous, but a movement key is held for a large share of
        its presses, and comparing the mean against the threshold separates the
        two reliably.
        """
        count, total, _longest = self.stats[name]
        if count <= 0:
            return False
        return (total / count) >= float(min_hold)

    def build_keymap(self, existing: Optional[Keymap] = None,
                     min_hold: float = 0.12) -> Keymap:
        keymap = existing or Keymap()
        keymap.origin = "calibrated"
        for name, (count, total, longest) in sorted(self.stats.items()):
            if name.startswith("MOUSE_"):
                if name not in keymap.mouse_buttons:
                    keymap.mouse_buttons.append(name)
                continue
            if name not in KNOWN_KEYS:
                continue
            hold = self.is_hold(name, min_hold)
            if hold:
                keymap.add(name, hold=True, source="calibrated")
            else:
                keymap.add(name, hold=False, source="calibrated")
        keymap.mouse_buttons.sort()
        keymap.sort_keys()
        if self.mouse_pixels:
            values = sorted(self.mouse_pixels)
            typical = values[len(values) // 2]
            keymap.default_mouse_turn = max(1, int(round(typical)))
            keymap.mouse_sensitivity = {
                "samples": len(values),
                "median_pixels_per_step": float(typical),
                "max_pixels_per_step": float(values[-1]),
            }
        keymap.samples = self.events
        if self.target_hwnd:
            keymap.game_hwnd = int(self.target_hwnd)
            try:
                keymap.game = win32gui.GetWindowText(self.target_hwnd)
                _l, _t, right, bottom = win32gui.GetClientRect(self.target_hwnd)
                keymap.game_resolution = [int(right), int(bottom)]
            except Exception:
                pass
        return keymap

    def report(self) -> str:
        lines = []
        for name, (count, total, longest) in sorted(
                self.stats.items(), key=lambda kv: -kv[1][0]):
            mean = total / max(1, count)
            kind = "hold" if self.is_hold(name, 0.12) else "tap "
            lines.append(f"  {name:<12} {kind}  pressed {int(count):4}x  "
                         f"mean {mean * 1000:6.0f} ms  longest "
                         f"{longest * 1000:6.0f} ms")
        if self.mouse_pixels:
            values = sorted(self.mouse_pixels)
            lines.append(f"  {'MOUSE MOVE':<12}       {len(values):4} events  "
                         f"median {values[len(values) // 2]:5.1f} px  "
                         f"max {values[-1]:5.1f} px")
        return "\n".join(lines) if lines else "  (nothing recorded)"


def find_toggle_vk(spec: str) -> int:
    """Turn the --calibrate hotkey name into a virtual key code."""
    from .runtime import parse_hotkey
    _modifiers, vk = parse_hotkey(spec)
    return vk
