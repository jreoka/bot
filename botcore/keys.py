"""
What the bot is allowed to press, and how it presses it.

Two ideas carry this file:

**The whitelist.**  The bot may only ever press keys that have been explicitly
allowed.  That list comes from a keymap file, which ``--calibrate`` writes by
watching the user play.  A key that is not in the list cannot be pressed and
cannot be learned, so there is no path by which the bot reaches into a game's
menus or its own dev overlays.

**The factored action space.**  The obvious encoding - one action per
combination of held keys, look direction and click - explodes into hundreds of
discrete actions, most of which are nonsense ("hold W+A+S+D and turn left and
right-click").  Instead the action is three independent decisions:

    held keys   |S| + 1 choices   (which key to toggle, or nothing)
    turn        |T| + 1 choices   (how far to swing the view, or not at all)
    tap/click   |A| + 1 choices   (one momentary button, or nothing)

Each decision has its own policy head, each is trained by the same PPO update,
and the bot can still learn combinations (holding W while turning) because the
heads are sampled together and credited together.  The decision count stays in
the tens instead of the hundreds, which is the difference between an action
space the bot can explore and one it cannot.
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .capture import require_windows

if os.name == "nt":  # pragma: no cover - platform split
    import win32api
    import win32gui

# =============================================================================
# Virtual key names
# =============================================================================

# Canonical name -> virtual key code.  Sided modifiers matter: games bind
# VK_LSHIFT / VK_LCONTROL (0xA0/0xA2), and the generic VK_SHIFT (0x10) is a
# different key that many games ignore entirely.
KNOWN_KEYS: Dict[str, int] = {
    "SPACE": 0x20, "ESC": 0x1B, "TAB": 0x09, "ENTER": 0x0D,
    "BACKSPACE": 0x08,
    "LSHIFT": 0xA0, "RSHIFT": 0xA1, "SHIFT": 0xA0,
    "LCTRL": 0xA2, "RCTRL": 0xA3, "CTRL": 0xA2,
    "LALT": 0xA4, "RALT": 0xA5, "ALT": 0xA4,
    "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27,
    "INSERT": 0x2D, "DELETE": 0x2E, "HOME": 0x24, "END": 0x23,
    "PAGEUP": 0x21, "PAGEDOWN": 0x22,
    "MOUSE_LEFT": 0x01, "MOUSE_RIGHT": 0x02, "MOUSE_MIDDLE": 0x04,
}
for _c in range(0x41, 0x5B):            # A-Z
    KNOWN_KEYS.setdefault(chr(_c), _c)
for _c in range(0x30, 0x3A):            # 0-9
    KNOWN_KEYS.setdefault(chr(_c), _c)
for _i in range(10):                    # numpad
    KNOWN_KEYS.setdefault(f"NUMPAD{_i}", 0x60 + _i)
for _i in range(1, 25):                 # F1-F24
    KNOWN_KEYS.setdefault(f"F{_i}", 0x6F + _i)

_EXTRA_NAMES = {
    0xBA: ";", 0xBB: "=", 0xBC: ",", 0xBD: "-", 0xBE: ".", 0xBF: "/",
    0xC0: "`", 0xDB: "[", 0xDC: "\\", 0xDD: "]", 0xDE: "'",
}
KNOWN_KEYS.update({v: k for k, v in _EXTRA_NAMES.items()})

MOUSE_BUTTON_VK = {"left": 0x01, "right": 0x02, "middle": 0x04}
MOUSE_VK_NAME = {0x01: "MOUSE_LEFT", 0x02: "MOUSE_RIGHT", 0x04: "MOUSE_MIDDLE"}

# Keys that need the extended-key flag on SendInput, or they register as the
# numeric-keypad equivalent.
_EXTENDED_VKS = {0xA1, 0xA3, 0xA5, 0x2D, 0x2E, 0x24, 0x23, 0x21, 0x22,
                 0x25, 0x26, 0x27, 0x28, 0x5B, 0x5C, 0x5D}

# The keys a bot needs most, used to order the action space so the useful
# entries come first.
_KEY_PRIORITY = ["W", "A", "S", "D", "SPACE", "LSHIFT", "LCTRL",
                 "MOUSE_LEFT", "MOUSE_RIGHT", "E", "Q"]

# Falls back to this when no keymap file exists: a conventional FPS layout.
DEFAULT_HOLD_KEYS = ["W", "A", "S", "D", "SPACE", "LSHIFT", "LCTRL"]
DEFAULT_TAP_KEYS = ["E", "Q", "1", "2", "3", "4"]
DEFAULT_MOUSE_BUTTONS = ["MOUSE_LEFT", "MOUSE_RIGHT"]


def key_name(vk: int, extended: bool = False) -> str:
    """Canonical name for a virtual-key code."""
    if 0x41 <= vk <= 0x5A or 0x30 <= vk <= 0x39:
        return chr(vk)
    if 0x60 <= vk <= 0x69:
        return f"NUMPAD{vk - 0x60}"
    if 0x70 <= vk <= 0x87:
        return f"F{vk - 0x6F}"
    if vk == 0x10:
        return "RSHIFT" if extended else "LSHIFT"
    if vk == 0x11:
        return "RCTRL" if extended else "LCTRL"
    if vk == 0x12:
        return "RALT" if extended else "LALT"
    return MOUSE_VK_NAME.get(vk) or _EXTRA_NAMES.get(vk) or f"VK_{vk:02X}"


def is_mouse_vk(vk: int) -> bool:
    return int(vk) in (0x01, 0x02, 0x04)


def _sort_key(name: str) -> Tuple[int, str]:
    try:
        return (_KEY_PRIORITY.index(name), name)
    except ValueError:
        return (len(_KEY_PRIORITY), name)


# =============================================================================
# Keymap
# =============================================================================

class Keymap:
    """
    The bot's whitelist: which keys exist, which are held vs tapped, and how
    far a mouse turn step is.
    """

    VERSION = 2
    DEFAULT_PATH = "keymap.json"

    def __init__(self, data: Optional[dict] = None):
        data = dict(data or {})
        self.path: Optional[str] = None
        self.origin: str = str(data.get("origin") or "default")
        self.created: str = str(data.get("created")
                                or datetime.now().isoformat(timespec="seconds"))
        self.updated: Optional[str] = data.get("updated")
        self.game: str = str(data.get("game") or "")
        self.game_hwnd: Optional[int] = data.get("game_hwnd")
        self.game_resolution = list(data.get("game_resolution") or [0, 0])
        self.samples: int = int(data.get("samples") or 0)
        self.default_mouse_turn: int = int(data.get("default_mouse_turn") or 20)
        self.mouse_sensitivity = dict(data.get("mouse_sensitivity") or {})

        self.holds: List[str] = []
        self.taps: List[str] = []
        self.vks: Dict[str, int] = {}

        for name, entry in (data.get("keys") or {}).items():
            if isinstance(entry, int):
                entry = {"vk": entry}
            upper = str(name).upper()
            code = int(entry.get("vk") or KNOWN_KEYS.get(upper) or 0)
            if not code:
                continue
            self.vks[upper] = code
            if str(entry.get("hold", "hold")) == "hold":
                self.holds.append(upper)
            else:
                self.taps.append(upper)
        self.mouse_buttons: List[str] = [
            str(b).upper() for b in (data.get("mouse_buttons") or [])
        ]

    # ---- construction ----
    @classmethod
    def default(cls) -> "Keymap":
        """The built-in whitelist, used before any --calibrate run."""
        keymap = cls({"origin": "default",
                      "game": "generic defaults (FPS-style layout)",
                      "default_mouse_turn": 20})
        for name in DEFAULT_HOLD_KEYS:
            keymap.add(name, hold=True, source="default")
        for name in DEFAULT_TAP_KEYS:
            keymap.add(name, hold=False, source="default")
        for name in DEFAULT_MOUSE_BUTTONS:
            keymap.add(name, hold=True, source="default")
        return keymap

    @classmethod
    def load(cls, path: str = DEFAULT_PATH, required: bool = False
             ) -> Optional["Keymap"]:
        if not path or not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"No keymap at '{path}'")
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception as exc:
            print(f"[Keymap] Could not read '{path}': {exc}")
            if required:
                raise
            return None
        if not isinstance(raw, dict):
            print(f"[Keymap] '{path}' is not a keymap object; ignoring it.")
            return None
        keymap = cls(raw)
        keymap.path = os.path.abspath(path)
        return keymap

    # ---- mutation ----
    def add(self, name: str, hold: bool, source: str = "calibrated",
            vk: Optional[int] = None) -> bool:
        """Add a key. Returns True when it was not already present."""
        upper = str(name).upper()
        code = int(vk) if vk else KNOWN_KEYS.get(upper)
        if not code:
            return False
        self.vks[upper] = int(code)
        bucket, other = (self.holds, self.taps) if hold else (self.taps, self.holds)
        if upper in other:
            other.remove(upper)
        if upper not in bucket:
            bucket.append(upper)
            return True
        return False

    def sort_keys(self) -> None:
        self.holds.sort(key=_sort_key)
        self.taps.sort(key=_sort_key)
        self.mouse_buttons.sort(key=_sort_key)

    def set_mouse_turn(self, pixels: Optional[int]) -> bool:
        """
        Override the measured mouse step. 0 or None keeps the measurement.

        Worth having as a flag because the measurement is unreliable in exactly
        the games that need mouse-look most: a title that locks the cursor
        (Minecraft, most first-person games) reports a nearly still cursor to a
        low-level mouse hook, so the recorded median comes out at a couple of
        pixels and the bot's turn step is a nudge nothing can see. The hook
        cannot measure what the game is actually reading from the device.
        """
        try:
            value = int(pixels or 0)
        except (TypeError, ValueError):
            return False
        if value <= 0 or value == int(self.default_mouse_turn):
            return False
        print(f"[Keymap] Mouse turn step: {int(self.default_mouse_turn)} px "
              f"(measured) -> {value} px (--mouse-turn).")
        self.default_mouse_turn = value
        return True

    def describe(self) -> str:
        return (f"hold=[{', '.join(self.holds) or '-'}]  "
                f"tap=[{', '.join(self.taps) or '-'}]  "
                f"mouse=[{', '.join(self.mouse_buttons) or '-'}]")

    # ---- persistence ----
    def to_dict(self) -> dict:
        keys: Dict[str, dict] = {}
        for name in self.holds:
            keys[name] = {"vk": self.vks[name], "hold": "hold"}
        for name in self.taps:
            keys[name] = {"vk": self.vks[name], "hold": "tap"}
        return {
            "version": self.VERSION,
            "origin": self.origin,
            "created": self.created,
            "updated": datetime.now().isoformat(timespec="seconds"),
            "game": self.game,
            "game_hwnd": self.game_hwnd,
            "game_resolution": list(self.game_resolution),
            "samples": self.samples,
            "default_mouse_turn": self.default_mouse_turn,
            "mouse_sensitivity": self.mouse_sensitivity,
            "keys": keys,
            "mouse_buttons": list(self.mouse_buttons),
        }

    def save(self, path: Optional[str] = None) -> str:
        target = os.path.abspath(path or self.path or self.DEFAULT_PATH)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        os.replace(tmp, target)
        self.path = target
        return target


# =============================================================================
# The factored action space
# =============================================================================

@dataclass
class ActionSpace:
    """
    Three independent decisions, plus the code that turns them into input.

        held_keys : indices into `hold_names`, at most max_held at once
        turns     : (dx, dy) pixel deltas; index 0 is always "do not turn"
        taps      : momentary buttons; index 0 is always "press nothing"
    """

    hold_names: List[str] = field(default_factory=list)
    hold_vks: List[int] = field(default_factory=list)
    turns: List[Tuple[int, int]] = field(default_factory=list)
    taps: List[dict] = field(default_factory=list)
    max_held: int = 4
    mouse_enabled: bool = True

    # ---- shape ----
    @property
    def n_hold(self) -> int:
        """One toggle choice per key, plus "toggle nothing"."""
        return len(self.hold_vks) + 1

    @property
    def n_turn(self) -> int:
        return len(self.turns) if self.turns else 1

    @property
    def n_tap(self) -> int:
        return len(self.taps) if self.taps else 1

    @property
    def head_sizes(self) -> Tuple[int, int, int]:
        return (self.n_hold, self.n_turn, self.n_tap)

    @property
    def action_vector_size(self) -> int:
        """Width of the one-hot action description fed to the reward nets."""
        return self.n_hold + self.n_turn + self.n_tap

    def action_vector(self, held_idx: int, turn_idx: int, tap_idx: int
                      ) -> np.ndarray:
        """Flat one-hot description of a decision, for the novelty nets."""
        vec = np.zeros(self.action_vector_size, dtype=np.float32)
        vec[max(0, min(held_idx, self.n_hold - 1))] = 1.0
        offset = self.n_hold
        vec[offset + max(0, min(turn_idx, self.n_turn - 1))] = 1.0
        offset += self.n_turn
        vec[offset + max(0, min(tap_idx, self.n_tap - 1))] = 1.0
        return vec

    # ---- labels ----
    def label(self, held_idx: int, turn_idx: int, tap_idx: int) -> str:
        parts: List[str] = []
        if 0 <= held_idx < len(self.hold_vks):
            parts.append(self.hold_names[held_idx])
        if 0 < turn_idx < len(self.turns):
            dx, dy = self.turns[turn_idx]
            parts.append(f"turn({dx:+d},{dy:+d})")
        if 0 < tap_idx < len(self.taps):
            parts.append(str(self.taps[tap_idx].get("label", "?")))
        return "+".join(parts) if parts else "noop"

    def is_noop(self, held_idx: int, turn_idx: int, tap_idx: int) -> bool:
        """True when this decision presses nothing at all."""
        return (held_idx == 0 and turn_idx == 0 and tap_idx == 0)

    def describe_action(self, held: np.ndarray, turn: int, tap: int
                        ) -> np.ndarray:
        """
        Flat action description for the novelty nets.

        The held mask is summarised by its lowest set key plus a count bit,
        rather than by the whole mask. The novelty nets only need to tell
        decisions apart, and a compact description keeps their input small:
        the difference between "hold W" and "hold W and shift" is one bit.
        """
        held = np.asarray(held).reshape(-1)
        first = 0
        if held.size:
            set_bits = np.flatnonzero(held > 0.5)
            first = int(set_bits[0]) + 1 if set_bits.size else 0
        return self.action_vector(first, int(turn), int(tap))

    # ---- serialisation ----
    def to_dict(self) -> dict:
        return {
            "hold_names": list(self.hold_names),
            "hold_vks": list(self.hold_vks),
            "turns": [list(t) for t in self.turns],
            "taps": [dict(t) for t in self.taps],
            "max_held": int(self.max_held),
            "mouse_enabled": bool(self.mouse_enabled),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionSpace":
        return cls(
            hold_names=list(data.get("hold_names") or []),
            hold_vks=[int(v) for v in (data.get("hold_vks") or [])],
            turns=[(int(t[0]), int(t[1])) for t in (data.get("turns") or [])],
            taps=[dict(t) for t in (data.get("taps") or [])],
            max_held=int(data.get("max_held") or 4),
            mouse_enabled=bool(data.get("mouse_enabled", True)),
        )

    # ---- construction ----
    @classmethod
    def build(cls, keymap: Keymap, max_held: int = 4,
              turn_levels: Sequence[Tuple[str, float]] = (
                  ("fine", 0.4), ("normal", 1.0), ("fast", 2.5)),
              mouse_enabled: bool = True,
              include_mouse_taps: bool = True) -> "ActionSpace":
        """
        Derive the action space from a keymap.

        Turns are generated from the keymap's calibrated pixels-per-step scaled
        by each configured level, in four directions, which is what makes a
        fine aiming correction and a spin on the spot the same kind of decision.
        """
        base = max(1, int(keymap.default_mouse_turn))

        hold_names = sorted(keymap.holds, key=_sort_key)
        hold_vks = [int(keymap.vks[n]) for n in hold_names if keymap.vks.get(n)]

        turns: List[Tuple[int, int]] = [(0, 0)]
        if mouse_enabled:
            for _level_name, scale in turn_levels:
                step = max(1, int(round(base * float(scale))))
                turns.extend([(-step, 0), (step, 0), (0, -step), (0, step)])

        taps: List[dict] = [{"label": "nothing", "vk": 0}]
        for name in sorted(keymap.taps, key=_sort_key):
            vk = keymap.vks.get(name)
            if vk:
                taps.append({"label": f"tap:{name}", "vk": int(vk)})
        if include_mouse_taps:
            for button in ("left", "right", "middle"):
                vk = MOUSE_BUTTON_VK[button]
                if vk not in hold_vks and vk not in [t["vk"] for t in taps]:
                    taps.append({"label": f"click:{button}", "vk": int(vk)})

        return cls(hold_names=hold_names, hold_vks=hold_vks, turns=turns,
                   taps=taps, max_held=max(1, int(max_held)),
                   mouse_enabled=bool(mouse_enabled))

    def describe(self) -> str:
        return (f"{len(self.hold_vks)} hold key(s) "
                f"(<= {self.max_held} at once), {self.n_turn} turn choice(s), "
                f"{self.n_tap} tap/click choice(s)  ->  "
                f"{self.n_hold * self.n_turn * self.n_tap} combinations")


# =============================================================================
# Input injection
# =============================================================================

class InputInjector:
    """
    Sends real keyboard and mouse input to the focused window.

    Mouse-look is deliberately a real cursor delta through SendInput: games
    read look from raw relative mouse motion, so a message-based approach
    cannot turn the view at all.  That also means the game window must be
    focused, which is why this object watches the foreground window rather than
    assuming it.

    **Focus is checked, not fought for.**  Input is only sent while the game is
    the foreground window; ``focus_policy`` decides whether the bot may take
    focus at all ("once" - when a run starts or resumes; "always" - on every
    action, which is what makes the terminal and the game wrestle for the
    keyboard; "never").  Sending keys at a window that is not in front does not
    reach the game - it goes to whatever *is* in front, which for a bot started
    from a terminal means the terminal itself.

    ``suspended`` is the soft switch used while the user has the controls:
    while it is set, this object injects nothing and says nothing, because it
    is expected rather than a bug.  ``enabled`` is the hard switch used by
    calibration, where any injection at all would be a bug worth shouting
    about.
    """

    POLICIES = ("once", "always", "never")

    def __init__(self, hwnd: int, focus_policy: str = "once"):
        require_windows("Input injection")
        self.hwnd = int(hwnd)
        self.focus_policy = (focus_policy if focus_policy in self.POLICIES
                             else "once")
        self.suspended = False
        self.enabled = True
        self.blocked = 0
        self.failures = 0
        self.skipped_unfocused = 0
        # What actually left this process. Counting the events Windows
        # accepted is the only way to tell "the bot is deciding" apart from
        # "the bot is playing"; a run that decides 20 times a second while
        # delivering nothing looks identical in every other number.
        self.events_sent = 0
        self.keys_sent = 0
        self.mouse_sent = 0
        self.last_error: Optional[str] = None
        self._focused = False
        self._focus_attempted = False
        self._focus_warned = False
        self._focus_notice_at = 0.0
        self._structs = None
        self.transient_vks: set = set()

    # ---- focus ----
    def focused(self) -> bool:
        """Would Windows deliver injected input to the game window?"""
        try:
            return win32gui.GetForegroundWindow() == self.hwnd
        except Exception:
            # Fail open: an API hiccup must not silently stop the bot.
            return True

    def acquire_focus(self, force: bool = False) -> bool:
        """
        Bring the game window forward. Called at start and on resume, not per
        step, so the desktop is not repeatedly yanked out from under the user.

        ``force`` asks for one attempt even under the "once" policy.
        """
        if self.focused():
            self._focused = True
            self._focus_warned = False
            return True
        if self.focus_policy == "never":
            return False
        if not force and self.focus_policy == "once" and self._focus_attempted:
            return False
        self._focus_attempted = True
        try:
            win32gui.SetForegroundWindow(self.hwnd)
        except Exception:
            # Windows refuses SetForegroundWindow from a process that does not
            # own the foreground; attaching to the foreground thread's input
            # queue is the standard way around it.
            try:
                import win32process
                fg = win32gui.GetForegroundWindow()
                fg_thread = win32process.GetWindowThreadProcessId(fg)[0]
                my_thread = ctypes.windll.kernel32.GetCurrentThreadId()
                attached = bool(ctypes.windll.user32.AttachThreadInput(
                    my_thread, fg_thread, True))
                try:
                    win32gui.SetForegroundWindow(self.hwnd)
                finally:
                    # Always detach: a leaked attachment couples this thread's
                    # input queue to the other window's, and focus then behaves
                    # strangely for every window afterwards.
                    if attached:
                        ctypes.windll.user32.AttachThreadInput(
                            my_thread, fg_thread, False)
            except Exception:
                return False
        self._focused = self.focused()
        if self._focused:
            self._focus_warned = False
        return self._focused

    def begin_action(self) -> None:
        """Called before a batch of input; decides whether it may be sent."""
        self._focused = self.focused()
        if self._focused:
            if self._focus_warned:
                self._focus_warned = False
                print("[Input] The game window is focused again; input resumed.",
                      flush=True)
            return
        if self.focus_policy == "always" and self.acquire_focus(force=True):
            return
        if not self._focus_warned:
            self._focus_warned = True
            self._focus_notice_at = time.monotonic()
            print("[Input] The game window is not focused, so nothing is being "
                  "sent. Click the game - input resumes on its own. (The "
                  "terminal can be used normally meanwhile.)", flush=True)
        elif time.monotonic() - self._focus_notice_at >= 60.0:
            # Repeating it once a minute matters: "the bot is running and
            # nothing is happening" is this, and the first message is easy to
            # miss in a wall of startup output.
            self._focus_notice_at = time.monotonic()
            print(f"[Input] Still sending nothing: the game window has not been "
                  f"the front window for {self.skipped_unfocused} step(s). "
                  f"Click the game to let the bot play it.", flush=True)

    def status_line(self) -> str:
        """One line: is input reaching a window, and how much of it."""
        state = "focused" if self.focused() else "NOT focused"
        line = (f"[Input] game window {state} - {self.keys_sent} key event(s) "
                f"and {self.mouse_sent} mouse event(s) delivered, "
                f"{self.skipped_unfocused} step(s) skipped while unfocused")
        if self.failures:
            line += f", {self.failures} SendInput failure(s)"
        if self.last_error:
            line += f" ({self.last_error})"
        return line

    def end_action(self) -> None:
        self._focused = False

    # ---- low-level plumbing ----
    @staticmethod
    def _input_structs():
        ULONG_PTR = (ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8
                     else ctypes.c_ulong)

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                        ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                        ("dwExtraInfo", ULONG_PTR)]

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                        ("mouseData", ctypes.c_ulong),
                        ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                        ("dwExtraInfo", ULONG_PTR)]

        class INPUT(ctypes.Structure):
            # The union must be as large as its largest member (MOUSEINPUT),
            # which is what makes sizeof(INPUT) == 40 on x64. Sizing it by
            # KEYBDINPUT gives 32 and SendInput then fails with
            # ERROR_INVALID_PARAMETER, silently.
            class _Union(ctypes.Union):
                _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]
            _anonymous_ = ("i",)
            _fields_ = [("type", ctypes.c_ulong), ("i", _Union)]

        return KEYBDINPUT, MOUSEINPUT, INPUT

    def _may_inject(self, what: str, forcing: bool = False) -> bool:
        if not self.enabled:
            self.blocked += 1
            if self.blocked <= 5:
                print(f"[Input] BLOCKED injection ({what}): input is disabled "
                      f"in this mode. This is a bug.")
            return False
        if self.suspended:
            return False
        if not forcing and not self._focused:
            # No focus, no input: a key sent now would land in some other
            # window, and the game would never see it.
            self.skipped_unfocused += 1
            return False
        return True

    def _send_key(self, vk: int, up: bool, forcing: bool = False) -> None:
        if not self._may_inject(f"vk 0x{int(vk):02X} {'up' if up else 'down'}",
                                forcing=forcing):
            return
        if self._structs is None:
            self._structs = self._input_structs()
        KEYBDINPUT, _MOUSEINPUT, INPUT = self._structs
        scan = win32api.MapVirtualKey(int(vk), 0)
        flags = 0x0002 if up else 0                    # KEYEVENTF_KEYUP
        if int(vk) in _EXTENDED_VKS:
            flags |= 0x0001                            # KEYEVENTF_EXTENDEDKEY
        inp = INPUT()
        inp.type = 1                                   # INPUT_KEYBOARD
        inp.ki = KEYBDINPUT(int(vk), scan, flags, 0, 0)
        sent = ctypes.windll.user32.SendInput(1, ctypes.byref(inp),
                                              ctypes.sizeof(INPUT))
        if not sent:
            self.failures += 1
            self.last_error = f"SendInput(key) failed, err={ctypes.get_last_error()}"
        else:
            self.keys_sent += 1
            self.events_sent += 1

    def _send_mouse(self, dx: int, dy: int, flags: int,
                    forcing: bool = False) -> None:
        if not self._may_inject(f"mouse dx={dx} dy={dy} flags=0x{flags:04X}",
                                forcing=forcing):
            return
        if self._structs is None:
            self._structs = self._input_structs()
        _KEYBDINPUT, MOUSEINPUT, INPUT = self._structs
        inp = INPUT()
        inp.type = 0                                   # INPUT_MOUSE
        inp.mi = MOUSEINPUT(int(dx), int(dy), 0, int(flags), 0, 0)
        sent = ctypes.windll.user32.SendInput(1, ctypes.byref(inp),
                                              ctypes.sizeof(INPUT))
        if not sent:
            self.failures += 1
            self.last_error = f"SendInput(mouse) failed, err={ctypes.get_last_error()}"
        else:
            self.mouse_sent += 1
            self.events_sent += 1

    # ---- public API ----
    def press_vk(self, vk: int) -> None:
        if vk:
            self._send_key(int(vk), False)

    def release_vk(self, vk: int) -> None:
        # Releases are always sent, focused or not: a key left down is worse
        # than a key-up delivered to the wrong window (a key-up for a key that
        # is not down does nothing).
        if vk:
            self._send_key(int(vk), True, forcing=True)

    def tap_vk(self, vk: int, seconds: float = 0.05) -> None:
        """Down, hold briefly, up. The key is noted while it is physically down."""
        vk = int(vk)
        if not vk:
            return
        self.transient_vks.add(vk)
        try:
            self._send_key(vk, False)
            time.sleep(max(0.0, float(seconds)))
            self._send_key(vk, True)
        finally:
            self.transient_vks.discard(vk)

    def mouse_move(self, dx: int, dy: int) -> None:
        self._send_mouse(int(dx), int(dy), 0x0001)     # MOUSEEVENTF_MOVE

    def mouse_down(self, button: str) -> None:
        flag = {"left": 0x0002, "right": 0x0008, "middle": 0x0020}.get(button)
        if flag:
            self._send_mouse(0, 0, flag)

    def mouse_up(self, button: str) -> None:
        flag = {"left": 0x0004, "right": 0x0010, "middle": 0x0040}.get(button)
        if flag:
            # Forced for the same reason as a key release: never leave a
            # button down because the window lost focus mid-click.
            self._send_mouse(0, 0, flag, forcing=True)

    def click(self, button: str, seconds: float = 0.05) -> None:
        vk = MOUSE_BUTTON_VK.get(button)
        if vk:
            self.transient_vks.add(vk)
        try:
            self.mouse_down(button)
            time.sleep(max(0.0, float(seconds)))
            self.mouse_up(button)
        finally:
            if vk:
                self.transient_vks.discard(vk)

    def release_all(self, held_vks: Sequence[int]) -> None:
        for vk in list(held_vks):
            self.release_vk(vk)

