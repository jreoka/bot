"""
Finding the game window, grabbing frames from it cheaply, and stacking them
into the observation the model sees.

The capture path is the bot's eyes and it runs on every control step, so it is
built to do as little as possible: one PrintWindow into a device context and
bitmap that are allocated once and reused, one grayscale conversion, one resize
into a preallocated buffer, and one difference channel produced as part of the
same pass.

Windows-only for anything that touches a window. Everything else (the stacker,
the resizer, the pacer) is plain numpy so it can be tested anywhere.
"""

from __future__ import annotations

import ctypes
import os
import time
from typing import List, Optional, Tuple

import numpy as np

if os.name == "nt":  # pragma: no cover - platform split
    import win32api
    import win32con
    import win32gui
    import win32process
    import win32ui

    _WINDOWS = True
else:  # The module still imports elsewhere so the model and trainer are testable.
    win32api = win32con = win32gui = win32process = win32ui = None  # type: ignore
    _WINDOWS = False

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


def require_windows(what: str) -> None:
    """Raise a readable error instead of an AttributeError on None."""
    if not _WINDOWS:
        raise RuntimeError(
            f"{what} needs Windows: this build of the bot drives a real game "
            f"window through Win32. The model, the trainer and the self-test "
            f"all run anywhere.")


# =============================================================================
# Window discovery
# =============================================================================

def get_window_pid(hwnd: int) -> int:
    try:
        return int(win32process.GetWindowThreadProcessId(hwnd)[1])
    except Exception:
        return 0


def get_process_name(pid: int) -> str:
    """Executable name owning pid, or '' when it cannot be read."""
    if not _WINDOWS or pid <= 0:
        return ""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return ""
    try:
        size = ctypes.c_ulong(1024)
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.QueryFullProcessImageNameW(
                handle, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
        return ""
    except Exception:
        return ""
    finally:
        try:
            ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass


def client_size(hwnd: int) -> Tuple[int, int]:
    try:
        _l, _t, right, bottom = win32gui.GetClientRect(hwnd)
        return int(right), int(bottom)
    except Exception:
        return 0, 0


def is_minimized(hwnd: int) -> bool:
    try:
        return bool(win32gui.IsIconic(hwnd))
    except Exception:
        return False


def window_title(hwnd: int) -> str:
    try:
        return win32gui.GetWindowText(hwnd) or ""
    except Exception:
        return ""


def enumerate_windows(min_area: int = 64 * 64) -> List[dict]:
    """Every visible, titled, large-enough top-level window."""
    require_windows("Listing windows")
    found: List[dict] = []

    def visit(hwnd, _param):
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title.strip():
                return True
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width, height = right - left, bottom - top
            if width * height < min_area:
                return True
            pid = get_window_pid(hwnd)
            found.append({
                "hwnd": int(hwnd),
                "title": title,
                "pid": pid,
                "process": get_process_name(pid),
                "rect": (int(left), int(top), int(width), int(height)),
                "client": client_size(hwnd),
                "minimized": is_minimized(hwnd),
            })
        except Exception:
            pass
        return True

    win32gui.EnumWindows(visit, None)
    found.sort(key=lambda w: (w["rect"][2] * w["rect"][3]), reverse=True)
    return found


# Windows that are almost always the desktop shell rather than a game. Kept
# short deliberately: a false negative (refusing a real game) is worse than a
# false positive (offering a picker).
_SHELL_PROCESSES = {
    "explorer.exe", "searchhost.exe", "startmenuexperiencehost.exe",
    "shellexperiencehost.exe", "textinputhost.exe", "dwm.exe",
    "applicationframehost.exe", "taskmgr.exe", "systemsettings.exe",
}
_SHELL_TITLES = ("program manager", "settings", "task manager")


def looks_like_game(win: dict) -> bool:
    """A cheap heuristic used only to decide whether to prompt."""
    process = (win.get("process") or "").lower()
    title = (win.get("title") or "").lower()
    if process in _SHELL_PROCESSES:
        return False
    if any(s in title for s in _SHELL_TITLES):
        return False
    width, height = win["rect"][2], win["rect"][3]
    if width < 480 or height < 320:
        return False
    return True


def find_game_window(prefer_largest: bool = True) -> Optional[int]:
    """Return a likely game hwnd, or None when nothing convincing is found."""
    for win in enumerate_windows():
        if looks_like_game(win):
            return int(win["hwnd"]) if prefer_largest else None
    return None


def format_window_table(windows: List[dict]) -> str:
    lines = []
    for i, win in enumerate(windows, 1):
        w, h = win["rect"][2], win["rect"][3]
        flag = " (minimized)" if win["minimized"] else ""
        lines.append(f"  [{i:2}] {w:5}x{h:<5} hwnd={win['hwnd']:<10} "
                     f"{win['process'] or '?':<24} {win['title'][:44]}{flag}")
    return "\n".join(lines)


def pick_window_interactive(windows: List[dict],
                            input_fn=input) -> Optional[dict]:
    """Ask which window to drive. Returns the chosen entry or None."""
    if not windows:
        print("[Window] No usable windows found. Start the game first.")
        return None
    print()
    print("-" * 78)
    print("  WHICH WINDOW SHOULD THE BOT PLAY?")
    print("-" * 78)
    print(format_window_table(windows))
    print("-" * 78)
    while True:
        try:
            raw = input_fn("  Number (blank = cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            return None
        try:
            index = int(raw)
        except ValueError:
            print("  Please type the number shown in brackets.")
            continue
        if 1 <= index <= len(windows):
            return windows[index - 1]
        print("  That number is not in the list.")


def resolve_target_window(configured: Optional[int] = None,
                          prefer_game_window: bool = True,
                          allow_prompt: bool = True) -> Optional[dict]:
    """
    Decide which window to play.

    Order: an explicit hwnd, then a confident auto-detect, then the picker.
    """
    require_windows("Choosing a game window")

    if configured:
        if not win32gui.IsWindow(int(configured)):
            print(f"[Window] hwnd {configured} is not a valid window.")
            return None
        pid = get_window_pid(int(configured))
        return {"hwnd": int(configured), "pid": pid,
                "title": window_title(int(configured)),
                "process": get_process_name(pid),
                "client": client_size(int(configured))}

    windows = enumerate_windows()
    if prefer_game_window:
        for win in windows:
            if looks_like_game(win):
                print(f"[Window] Auto-selected: '{win['title']}' "
                      f"({win['process']}) {win['rect'][2]}x{win['rect'][3]}")
                print("[Window] Use --window to choose a different one, or run "
                      "with --pick.")
                return win

    if not allow_prompt:
        return None

    chosen = pick_window_interactive(windows)
    if chosen is None:
        return None
    print(f"[Window] Using '{chosen['title']}' ({chosen['process']})")
    return chosen


# =============================================================================
# Frame capture
# =============================================================================

class FrameGrabber:
    """
    Reusable PrintWindow grabber.

    ``capture_window`` (the naive version) allocates a device context and a
    bitmap per call, which caps the loop at a few frames per second before any
    model runs at all. This keeps one DC and one bitmap alive for the window's
    current client size and only rebuilds them on resize, so the per-frame cost
    is PrintWindow itself.
    """

    PW_RENDERFULLCONTENT = 0x00000002

    def __init__(self, hwnd: int):
        require_windows("Frame capture")
        self.hwnd = int(hwnd)
        self._size: Tuple[int, int] = (0, 0)
        self._hwnd_dc = None
        self._mfc_dc = None
        self._save_dc = None
        self._bitmap = None
        self.last_error: Optional[str] = None
        self.frames = 0
        self.total_seconds = 0.0

    # ---- lifecycle ----
    def _release(self) -> None:
        if self._bitmap is not None:
            try:
                win32gui.DeleteObject(self._bitmap.GetHandle())
            except Exception:
                pass
            self._bitmap = None
        for attr in ("_save_dc", "_mfc_dc"):
            dc = getattr(self, attr)
            if dc is not None:
                try:
                    dc.DeleteDC()
                except Exception:
                    pass
                setattr(self, attr, None)
        if self._hwnd_dc is not None:
            try:
                win32gui.ReleaseDC(self.hwnd, self._hwnd_dc)
            except Exception:
                pass
            self._hwnd_dc = None
        self._size = (0, 0)

    def close(self) -> None:
        self._release()

    def __enter__(self) -> "FrameGrabber":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- capture ----
    def _ensure_buffers(self, width: int, height: int) -> None:
        if (width, height) == self._size:
            return
        self._release()
        self._hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        self._mfc_dc = win32ui.CreateDCFromHandle(self._hwnd_dc)
        self._save_dc = self._mfc_dc.CreateCompatibleDC()
        self._bitmap = win32ui.CreateBitmap()
        self._bitmap.CreateCompatibleBitmap(self._mfc_dc, width, height)
        self._save_dc.SelectObject(self._bitmap)
        self._size = (width, height)

    def grab(self) -> Optional[np.ndarray]:
        """
        Capture the client area as a BGR array, or None.

        A window that is closed, minimized or refusing PrintWindow returns None
        rather than raising: the caller decides what that means.
        """
        started = time.perf_counter()
        try:
            _l, _t, right, bottom = win32gui.GetClientRect(self.hwnd)
            width, height = int(right), int(bottom)
            if width <= 0 or height <= 0:
                self.last_error = "window has no client area"
                return None
            self._ensure_buffers(width, height)

            result = ctypes.windll.user32.PrintWindow(
                self.hwnd, self._save_dc.GetSafeHdc(),
                self.PW_RENDERFULLCONTENT)
            if result != 1:
                # Some games only render into a plain PrintWindow.
                result = ctypes.windll.user32.PrintWindow(
                    self.hwnd, self._save_dc.GetSafeHdc(), 0)
            if result != 1:
                self.last_error = f"PrintWindow returned {result}"
                return None

            info = self._bitmap.GetInfo()
            bits = self._bitmap.GetBitmapBits(True)
            buf = np.frombuffer(bits, dtype=np.uint8)
            buf = buf.reshape((info["bmHeight"], info["bmWidth"], 4))
            self.last_error = None
            return _to_bgr(buf)
        except Exception as exc:
            self.last_error = str(exc)
            return None
        finally:
            self.frames += 1
            self.total_seconds += time.perf_counter() - started

    @property
    def mean_ms(self) -> float:
        return 1000.0 * self.total_seconds / max(1, self.frames)


def _to_bgr(bgra: np.ndarray) -> np.ndarray:
    """BGRA -> BGR, using numpy when cv2 is unavailable."""
    if cv2 is not None:
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(bgra[:, :, :3])


# =============================================================================
# The observation the model actually sees
# =============================================================================

class FrameStack:
    """
    Turns a stream of screenshots into a fixed-shape tensor.

    Output is ``(frame_stack + 1, size, size)`` float32 in [0, 1]:

        channels [0 .. n-1]  the last n frames, oldest first, as grayscale
        channel  n           |newest - oldest|, a motion/change channel

    The difference channel costs a subtraction, and it is what lets a small
    network see motion without needing a recurrent pass over the pixels or a
    stack long enough to infer it from per-frame appearance alone.
    """

    def __init__(self, size: int, stack: int):
        self.size = int(size)
        self.stack = int(stack)
        self.channels = self.stack + 1
        self._buf = np.zeros((self.stack, self.size, self.size), dtype=np.uint8)
        self._filled = 0
        self._out = np.zeros((self.channels, self.size, self.size),
                             dtype=np.float32)
        # Work buffers, allocated once.
        self._gray = np.zeros((self.size, self.size), dtype=np.uint8)
        self._diff = np.zeros((self.size, self.size), dtype=np.uint8)

    def clear(self) -> None:
        self._buf[:] = 0
        self._filled = 0

    def _resize_into(self, bgr: np.ndarray) -> np.ndarray:
        """BGR -> size x size grayscale, reusing the internal buffer."""
        if cv2 is not None:
            small = cv2.resize(bgr, (self.size, self.size),
                               interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Fallback without cv2: integer box downsample, then luma weights.
        h, w = bgr.shape[:2]
        ys = (np.arange(self.size) * h // self.size)
        xs = (np.arange(self.size) * w // self.size)
        small = bgr[np.ix_(ys, xs)].astype(np.float32)
        gray = (0.114 * small[:, :, 0] + 0.587 * small[:, :, 1]
                + 0.299 * small[:, :, 2])
        return gray.astype(np.uint8)

    def push(self, bgr: np.ndarray) -> np.ndarray:
        """Add one frame and return the observation tensor (a live view)."""
        gray = self._resize_into(bgr)
        if self.stack > 1:
            self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1] = gray
        self._filled = min(self.stack, self._filled + 1)
        return self.observe()

    def observe(self) -> np.ndarray:
        """Current observation without advancing the stack."""
        n = self.stack
        self._out[:n] = self._buf.astype(np.float32) * (1.0 / 255.0)
        if n > 1:
            np.subtract(self._buf[-1], self._buf[0], out=self._diff)
        else:
            self._diff[:] = 0
        self._out[n] = self._diff.astype(np.float32) * (1.0 / 255.0)
        return self._out


def gray_signature(frame: np.ndarray, size: int = 32) -> np.ndarray:
    """
    Tiny fingerprint of a frame, used by novelty and reset detection.

    Deliberately blunt: a sharp fingerprint makes every frame look like a new
    situation, which turns count-based novelty into a flat per-step bonus and
    makes reset detection fire constantly.

    Accepts either a grey 2-D frame, an HxWx3 BGR image, or float values that
    are already in [0, 1]. Getting that wrong is quiet and expensive - a
    fingerprint of all zeros makes every frame look identical, which silently
    removes the entire episodic reward - so the range is detected rather than
    assumed.
    """
    if frame.ndim == 3:
        if frame.shape[2] == 3 and cv2 is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.mean(axis=2)
    else:
        gray = frame

    if gray.dtype == np.uint8:
        small = (cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
                 if cv2 is not None else _box_downsample(gray, size))
        return (small.astype(np.float32) / 255.0).reshape(-1)

    data = gray.astype(np.float32)
    scale = 255.0 if float(data.max(initial=0.0)) > 1.5 else 1.0
    small = (cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)
             if cv2 is not None else _box_downsample(data, size))
    return (small / scale).reshape(-1)


def _box_downsample(image: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour downsample, for when cv2 is unavailable."""
    h, w = image.shape[:2]
    return image[np.ix_(np.arange(size) * h // size, np.arange(size) * w // size)]


# =============================================================================
# Pacing
# =============================================================================

class Pacer:
    """
    Holds the control loop at a target rate.

    When the machine cannot keep up, the schedule slips instead of sprinting to
    catch up: a bot that owes twenty frames must not deliver twenty steps of
    input back to back, because the game would see a burst of input with no
    time to simulate in between.
    """

    def __init__(self, target_fps: float, extra_sleep: float = 0.0):
        self.target_fps = max(0.0, float(target_fps))
        self.extra_sleep = max(0.0, float(extra_sleep))
        self.step_seconds = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self._next = time.perf_counter()
        self.achieved_hz = 0.0
        self._last_report = self._next
        self._count = 0
        self.overruns = 0

    def reset(self) -> None:
        self._next = time.perf_counter()
        self._last_report = self._next
        self._count = 0

    def tick(self) -> None:
        if self.step_seconds > 0:
            self._next += self.step_seconds
            wait = self._next - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            elif wait < -self.step_seconds:
                # Behind schedule: resynchronise rather than fire a burst.
                self.overruns += 1
                self._next = time.perf_counter()
        if self.extra_sleep > 0:
            time.sleep(self.extra_sleep)
        self._count += 1

    def report(self, every: float = 15.0) -> Optional[str]:
        now = time.perf_counter()
        if now - self._last_report < every:
            return None
        window = now - self._last_report
        self.achieved_hz = self._count / max(1e-6, window)
        self._count = 0
        self._last_report = now
        line = f"[Perf] {self.achieved_hz:5.1f} steps/s"
        if self.target_fps > 0:
            line += f" (target {self.target_fps:.0f})"
            if self.achieved_hz < self.target_fps * 0.6:
                line += "  <- behind target; lower frame_size or action_repeat"
        return line
