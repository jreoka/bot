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
from typing import Dict, List, Optional, Set, Tuple

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
    own = own_process_ids()
    console = get_console_window()

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
                # Marked here, once, so every consumer (auto-detect, the
                # picker, the table, --window validation) agrees about which
                # windows are the bot's own and must never be driven.
                "own": pid > 0 and pid in own,
                "console": bool(console) and int(hwnd) == console,
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


# Programs that are not games, in two groups.
#
# The first group is the one that matters, and it is the reason this table
# exists at all: the terminal, IDE or shell the bot was *launched from*. A bot
# that drives its own terminal types its actions into a shell prompt, and
# because the injector only checks that the target is the *foreground* window,
# that run looks perfectly healthy from the inside - 20 decisions a second, no
# error, no warning - while the game sits untouched. If the bot's own window can
# ever win "largest likely game", this failure is one keypress away.
_TERMINAL_PROCESSES = {
    "windowsterminal.exe", "openconsole.exe", "conhost.exe", "wt.exe",
    "cmd.exe", "powershell.exe", "pwsh.exe", "bash.exe", "wsl.exe",
    "mintty.exe", "putty.exe", "alacritty.exe", "wezterm-gui.exe",
    "conemu.exe", "conemu64.exe", "tabby.exe", "hyper.exe",
}
# The second group is a guess rather than a hazard: driving a browser or an
# editor is merely the wrong window, not a way to run a command. They are
# excluded from auto-detection (a maximised browser is otherwise the largest
# window on most desktops and beats a windowed game every time) and warned
# about, but never refused.
_APP_PROCESSES = {
    "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe",
    "vivaldi.exe", "iexplore.exe",
    "code.exe", "code - insiders.exe", "devenv.exe", "notepad++.exe",
    "sublime_text.exe", "pycharm64.exe", "idea64.exe", "rider64.exe",
    "discord.exe", "slack.exe", "teams.exe", "ms-teams.exe", "zoom.exe",
    "spotify.exe",
}
_NON_GAME_PROCESSES = _TERMINAL_PROCESSES | _APP_PROCESSES


def looks_like_non_game_process(process: str) -> bool:
    """Is this executable one of the known non-games (terminal, browser, IDE)?"""
    return str(process or "").lower() in _NON_GAME_PROCESSES


def get_console_window() -> int:
    """
    The console window this process writes to, or 0.

    A classic console answers here; Windows Terminal and an IDE terminal do
    not, because ConPTY hands the process a hidden pseudo-window instead of the
    window the user can see. ``own_process_ids`` covers those.
    """
    if not _WINDOWS:
        return 0
    try:
        return int(ctypes.windll.kernel32.GetConsoleWindow() or 0)
    except Exception:
        return 0


def own_process_ids() -> Set[int]:
    """
    This process plus every ancestor: the shell, the terminal, the IDE.

    Walking the parent chain is what makes this work where ``GetConsoleWindow``
    does not: running the bot from Windows Terminal gives a hidden ConPTY
    window, not the visible terminal, so the only reliable link to the window
    the user is looking at is that Windows Terminal started the shell that
    started the bot.
    """
    if not _WINDOWS:
        return set()
    TH32CS_SNAPPROCESS = 0x00000002
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [("dwSize", ctypes.c_ulong),
                    ("cntUsage", ctypes.c_ulong),
                    ("th32ProcessID", ctypes.c_ulong),
                    ("th32DefaultHeapID", ctypes.c_void_p),
                    ("th32ModuleID", ctypes.c_ulong),
                    ("cntThreads", ctypes.c_ulong),
                    ("th32ParentProcessID", ctypes.c_ulong),
                    ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", ctypes.c_ulong),
                    ("szExeFile", ctypes.c_char * 260)]

    parents: Dict[int, int] = {}
    try:
        k32 = ctypes.windll.kernel32
        k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        k32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
        snapshot = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if not snapshot or snapshot == INVALID_HANDLE_VALUE:
            return {os.getpid()}
        try:
            entry = PROCESSENTRY32()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
            more = k32.Process32First(snapshot, ctypes.byref(entry))
            while more:
                parents[int(entry.th32ProcessID)] = int(
                    entry.th32ParentProcessID)
                more = k32.Process32Next(snapshot, ctypes.byref(entry))
        finally:
            try:
                k32.CloseHandle(snapshot)
            except Exception:
                pass
    except Exception:
        return {os.getpid()}

    ids = {os.getpid()}
    pid = os.getpid()
    for _ in range(16):                    # depth cap: never loop forever
        parent = parents.get(pid, 0)
        if not parent or parent in ids:
            break
        ids.add(parent)
        pid = parent
    return ids


def window_is_own_process(hwnd: int) -> bool:
    """Does this window belong to the bot, or to whatever launched it?"""
    if not _WINDOWS or not hwnd:
        return False
    if int(hwnd) == get_console_window():
        return True
    pid = get_window_pid(int(hwnd))
    return pid > 0 and pid in own_process_ids()


def window_risk(win) -> str:
    """
    Why this window must not be driven, or '' when it looks safe to drive.

    Accepts a window dict from ``enumerate_windows`` or a bare hwnd. Only
    self-inflicted cases are refused: the bot's own terminal is a guaranteed
    way to send every keystroke into a shell instead of a game. A browser is a
    wrong guess, not a hazard, so it is warned about rather than blocked.
    """
    if win is None:
        return ""
    hwnd = int(win.get("hwnd") or 0) if isinstance(win, dict) else int(win or 0)
    if not hwnd:
        return "it has no window handle"
    if hwnd == get_console_window():
        return ("it is this process's own console window, so the bot would "
                "type its actions into a shell")
    own = bool(win.get("own")) if isinstance(win, dict) else False
    console = bool(win.get("console")) if isinstance(win, dict) else False
    if own or console or window_is_own_process(hwnd):
        return ("it belongs to this process or to the shell that launched it "
                "(the terminal/IDE), so the bot would type its actions into a "
                "shell prompt instead of the game")
    return ""


def describe_window(win) -> str:
    """'Title' (process.exe) hwnd=N, for logs."""
    if not win:
        return "(no window)"
    hwnd = int(win.get("hwnd") or 0) if isinstance(win, dict) else int(win)
    if isinstance(win, dict):
        title, process = win.get("title") or "", win.get("process") or ""
    else:
        title, process = window_title(hwnd), get_process_name(
            get_window_pid(hwnd))
    return f"'{title or '?'}' ({process or '?'}) hwnd={hwnd}"


def looks_like_game(win: dict) -> bool:
    """
    A cheap heuristic used only to decide whether to prompt.

    It refuses this process's own windows outright, however large they are, and
    it refuses the programs a developer already has open (browsers, editors,
    chat). Both refusals are about the same failure: auto-detection that picks
    the biggest window on the desktop picks the terminal the bot was started
    from.
    """
    if win.get("own") or win.get("console"):
        return False
    process = (win.get("process") or "").lower()
    title = (win.get("title") or "").lower()
    if process in _SHELL_PROCESSES or process in _NON_GAME_PROCESSES:
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
        notes = []
        if win.get("minimized"):
            notes.append("minimized")
        if win.get("own") or win.get("console"):
            notes.append("the terminal the bot runs in - never driven")
        elif looks_like_non_game_process(win.get("process") or ""):
            notes.append("not a game")
        flag = f"  ({'; '.join(notes)})" if notes else ""
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


def _window_entry(hwnd: int) -> Optional[dict]:
    """One window in the same shape as ``enumerate_windows``, by handle."""
    if not hwnd or not win32gui.IsWindow(int(hwnd)):
        return None
    pid = get_window_pid(int(hwnd))
    console = get_console_window()
    try:
        left, top, right, bottom = win32gui.GetWindowRect(int(hwnd))
    except Exception:
        left = top = right = bottom = 0
    return {"hwnd": int(hwnd), "pid": pid, "title": window_title(int(hwnd)),
            "process": get_process_name(pid),
            "rect": (int(left), int(top), int(right - left), int(bottom - top)),
            "client": client_size(int(hwnd)),
            "minimized": is_minimized(int(hwnd)),
            "own": pid > 0 and pid in own_process_ids(),
            "console": bool(console) and int(hwnd) == console}


def titles_match(remembered: str, current: str) -> bool:
    """
    Loose title comparison, because a game's title changes with its level.

    Either title containing the other counts, and so does a shared first word
    of at least three characters: "Minecraft 26.3 - Singleplayer" matches
    "Minecraft 26.3 - Multiplayer". A handle that Windows has recycled onto an
    unrelated window fails all three checks, which is the whole point of
    checking - a stale handle is a wrong window.
    """
    a = " ".join(str(remembered or "").lower().split())
    b = " ".join(str(current or "").lower().split())
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return len(a.split()[0]) >= 3 and a.split()[0] == b.split()[0]


def resolve_target_window(configured: Optional[int] = None,
                          prefer_game_window: bool = True,
                          allow_prompt: bool = True,
                          remembered_hwnd: Optional[int] = None,
                          remembered_title: str = "") -> Optional[dict]:
    """
    Decide which window to play.

    Order: an explicit hwnd, then the window a previous ``--calibrate``
    recorded - by handle, then by title - then the window the user is looking
    at, then the largest plausible game, then the picker.

    The calibrated window comes second on purpose: it is the only evidence the
    bot has that the user has already pointed at the game. Ignoring it is how a
    run ends up driving the terminal it was launched from - which is the
    largest window on most desktops, and the one window that must never be
    driven, because every action would be typed into a shell prompt.
    """
    require_windows("Choosing a game window")

    if configured:
        win = _window_entry(int(configured))
        if win is None:
            print(f"[Window] hwnd {configured} is not a valid window.")
            return None
        risk = window_risk(win)
        if risk:
            print(f"[Window] WARNING: {describe_window(win)} - {risk}.")
            print("[Window] Pass the game's handle instead: --list-windows, "
                  "then --window HWND.")
        return win

    if remembered_hwnd:
        win = _window_entry(int(remembered_hwnd))
        if win is None:
            print(f"[Window] The window calibrated earlier (hwnd "
                  f"{remembered_hwnd}) is not open any more; looking for the "
                  f"game.")
        elif win.get("own") or win.get("console"):
            print(f"[Window] The handle calibrated earlier now points at "
                  f"{describe_window(win)} - that is this terminal, not the "
                  f"game. Ignoring it.")
        elif remembered_title and not titles_match(remembered_title,
                                                   win.get("title") or ""):
            print(f"[Window] hwnd {remembered_hwnd} is now "
                  f"'{win.get('title')}', not '{remembered_title}'; ignoring "
                  f"the remembered handle, because Windows reuses them.")
        elif not remembered_title and looks_like_non_game_process(
                win.get("process") or ""):
            print(f"[Window] The calibrated handle {remembered_hwnd} now "
                  f"belongs to {win.get('process')}, which is not a game; "
                  f"ignoring it.")
        else:
            print(f"[Window] Using the window you calibrated against: "
                  f"{describe_window(win)}")
            return win

    windows = enumerate_windows()
    # The handle is often gone (a game gets a new one every launch, and Windows
    # hands old numbers out again), but the title usually survives, so the
    # calibration still identifies the window it was recorded against.
    if remembered_title:
        for win in windows:
            if looks_like_game(win) and titles_match(remembered_title,
                                                     win.get("title") or ""):
                print(f"[Window] Found the game you calibrated against, by "
                      f"title: {describe_window(win)}")
                return win
    if prefer_game_window:
        ordered: List[dict] = []
        foreground = _window_entry(int(win32gui.GetForegroundWindow() or 0))
        if foreground is not None and looks_like_game(foreground):
            ordered.append(foreground)
        for win in windows:
            if looks_like_game(win) and all(w["hwnd"] != win["hwnd"]
                                            for w in ordered):
                ordered.append(win)
        if ordered:
            chosen = ordered[0]
            if foreground is not None and chosen["hwnd"] == foreground["hwnd"]:
                print(f"[Window] Using the window in front: "
                      f"{describe_window(chosen)}")
            else:
                print(f"[Window] Auto-selected the largest likely game: "
                      f"{describe_window(chosen)} "
                      f"({chosen['rect'][2]}x{chosen['rect'][3]})")
            print("[Window] This terminal, any open browser/editor, and this "
                  "process's own windows are never auto-selected. Use --pick "
                  "or --window HWND for a different window.")
            return chosen
        print("[Window] Nothing on screen looks like a game (terminals, "
              "browsers and editors are excluded on purpose).")

    if not allow_prompt:
        return None

    chosen = pick_window_interactive(windows)
    if chosen is None:
        return None
    risk = window_risk(chosen)
    if risk:
        print(f"[Window] WARNING: {describe_window(chosen)} - {risk}.")
        print("[Window] Driving it would send every action into that program, "
              "not into a game.")
    print(f"[Window] Using {describe_window(chosen)}")
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

    The difference channel costs a subtraction, and it is what lets the model
    see motion without having to infer it from per-frame appearance alone, or
    from a stack long enough to make the motion obvious.
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

    def __init__(self, target_fps: float, extra_sleep: float = 0.0,
                 decisions_per_step: int = 1):
        self.target_fps = max(0.0, float(target_fps))
        self.extra_sleep = max(0.0, float(extra_sleep))
        # How many of these steps one decision is held for. Only used to label
        # the rate honestly: with action_repeat=2 the bot takes 20 game steps a
        # second and 10 decisions a second, and reporting one number for both
        # is how "it says 20 steps a second" ends up meaning nothing.
        self.decisions_per_step = max(1, int(decisions_per_step))
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
        line = f"[Perf] {self.achieved_hz:5.1f} game step(s)/s"
        if self.decisions_per_step > 1:
            line += (f"  = {self.achieved_hz / self.decisions_per_step:.1f} "
                     f"decision(s)/s at action_repeat {self.decisions_per_step}")
        if self.target_fps > 0:
            line += f" (target {self.target_fps:.0f})"
            if self.achieved_hz < self.target_fps * 0.6:
                line += "  <- behind target; lower frame_size or action_repeat"
        return line
