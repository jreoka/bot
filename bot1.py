import argparse
import ctypes
import hashlib
import itertools
import json
import os
import queue
import re
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def _require(module: str, pip_name: str = None):
    """
    Import a hard dependency, or exit with one readable line.

    On a machine that has only just been handed this script, a missing wheel
    should say which package to install rather than dying with a bare
    ModuleNotFoundError from somewhere deep in the import block.
    """
    try:
        return __import__(module)
    except ImportError as exc:
        pkg = pip_name or module
        print(f"[Setup] Missing Python package '{pkg}' ({exc}).")
        print(f"[Setup] Install the training dependencies with:")
        print(f"[Setup]   python -m pip install torch numpy opencv-python "
              f"gymnasium pywin32 prodigyopt soundcard")
        sys.exit(2)


_require("numpy")
_require("torch")
if os.name == "nt":
    _require("win32process", "pywin32")
    _require("win32gui", "pywin32")
    _require("win32api", "pywin32")
    _require("win32ui", "pywin32")
    _require("win32con", "pywin32")

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
if os.name == "nt":
    import win32process
    import win32gui
    import win32api
    import win32ui
    import win32con
else:
    # The script is a Windows tool (PrintWindow capture plus SendInput). These
    # placeholders exist only so the model, the config and the checkpoint logic
    # can still be imported and unit-tested on another platform; anything that
    # actually touches a window fails loudly with a readable message instead of
    # an AttributeError on None.
    def _windows_only(*_args, **_kwargs):
        raise RuntimeError(
            "window capture and input injection are Windows-only; "
            "this module is only importable elsewhere for testing")

    win32process = win32gui = win32api = win32ui = win32con = None  # type: ignore
try:
    import soundcard as sc
    _HAS_SOUNDCARD = True
except ImportError:
    _HAS_SOUNDCARD = False
    print("[Audio] 'soundcard' not installed; audio will be silence.")
    print("        Install with: python -m pip install soundcard")
_require("gymnasium", "gymnasium")
import gymnasium as gym
from gymnasium import spaces
_require("prodigyopt", "prodigyopt")
from prodigyopt import Prodigy



# =============================================================================
# SECTION 0: TUNABLE CONFIGURATION
#
# Everything you might want to change lives here. With these defaults you can
# just run:  python bot1.py
# =============================================================================

# ---- audio -----------------------------------------------------------------
# "process"     capture ONLY the target window's process (recommended)
# "exclude-self"everything except this script's own audio
# "system"      whole default output device (your music/videos included)
# "auto"        try process, fall back to system with a warning
# "off"         no audio at all, observations are silence
AUDIO_MODE = "process"

# ---- global hotkeys (work even when the console is not focused) -------------
# Format: "modifier+modifier+KEY". Valid modifiers: ctrl, alt, shift, win.
# Valid keys: F1-F24, A-Z, 0-9, SPACE, ESC, TAB, INSERT, DELETE, HOME, END.
# Set any of these to "" to disable that individual hotkey.
HOTKEY_PAUSE = "f8"    # pause / resume the bot
HOTKEY_SAVE = "f9"     # write a checkpoint immediately
HOTKEY_QUIT = "f10"    # quit cleanly (saves a checkpoint first)

# ---- rolling checkpoints ---------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL_SEC = 5 * 60   # save every 5 minutes
CHECKPOINT_KEEP = 3                # keep only the 3 newest timestamped checkpoints
RESUME_ON_START = True             # continue from the newest checkpoint if present

# ---- window selection ------------------------------------------------------
# None = auto-detect a likely game window, or show the interactive picker when
# nothing looks like a game.
# An int = use that hwnd directly (set DIAGNOSTIC = "windows" to list them).
TARGET_WINDOW_HWND = None
# True  = auto-target the largest likely game window without prompting.
# False = always show the interactive picker instead.
PREFER_GAME_WINDOW = True

# ---- training / machine load ----------------------------------------------
# The bot trains forever by default. TOTAL_STEPS is the only step target, and
# "forever" is expressed as an unreachable step count so the loop keeps
# resuming, checkpointing and printing stats exactly as it always did. Stop it
# with F10 (clean quit + checkpoint) or Ctrl+C; both save first, and the next
# launch resumes automatically. Set TOTAL_STEPS to a number for a finite run.
FOREVER = "forever"
FOREVER_STEPS = 10 ** 12            # effectively unlimited: ~1000 years at 30 Hz
TOTAL_STEPS = FOREVER              # forever | an int, e.g. 100_000
BATCH_SIZE = 16                     # transitions per PPO update
# Control rate. TARGET_FPS is the wall-clock ceiling on how often the bot sees
# the screen and can act; CAPTURE_DELAY is an extra sleep on top of that. At
# 96px/20Hz the loop reacts in about 50 ms, which is fast enough to answer most
# things a game throws at you. Raise CAPTURE_DELAY or lower TARGET_FPS if the
# game stutters (watch the [Perf] line).
TARGET_FPS = 20.0
CAPTURE_DELAY = 0.0                # extra seconds slept between steps
# ---- observation size: the single biggest lever on CPU cost -----------------
# These defaults are sized for a 6-core laptop CPU with no CUDA, and were
# measured on one (Ryzen 5 7530U): a whole PPO update here costs about 0.6 s of
# CPU, roughly 60% of wall clock left for actually playing the game.
#
# What actually drives the cost, in order:
#   1. SEQ_LEN - every frame is a full CNN pass, and attention is SEQ_LEN^2.
#      Halving it roughly halves the update.
#   2. the raw-audio CNN (the model pools the waveform by AUDIO_DOWNSAMPLE
#      first, which is worth about 3.5x on its own; see SECTION 3).
#   3. IMG_SIZE - much weaker than it looks, because the stride-4 first conv
#      dominates and its cost is nearly flat from 64px to 96px. 96px is
#      therefore kept for visual detail; it is not what is slowing you down.
# Raise these only if the [Perf] line in the log shows room to spare.
IMG_SIZE = 96
SEQ_LEN = 16
DEVICE = "auto"                    # auto | cuda | cpu
PRIORITY = "normal"          # below_normal | normal | high
SHOW_PREVIEW = True               # small live view of what the model sees
ENABLE_HOTKEYS = True              # global pause / save / quit hotkeys
RUN_STARTUP_TESTS = True           # capture/audio/input checks at startup

# ---- transformer / PPO update budget ----------------------------------------
# The model is a SwiGLU + RoPE transformer (SECTION 3). These knobs are what
# keep a CPU run moving instead of appearing to hang:#   * MINIBATCH_SIZE splits each PPO epoch into small backward passes. Smaller
#     passes are also what makes the update fit in cache on a weak CPU, so this
#     is usually faster AND better for the optimizer than one big batch. On the
#     reference machine 8 measured slightly faster than 4 or 16.
#   * UPDATE_SECONDS_BUDGET caps how long a single PPO update may keep looping
#     over minibatches; once exceeded, that update stops where it is and
#     control returns to the game. This is the hard guarantee against a stall:
#     no single update can run away with the whole run.
#   * The optimizer gets a short warmup then a cosine decay to LR_MIN_FRACTION,
#     which is what stops the policy from oscillating forever without ever
#     converging once the novelty bonus inevitably shrinks.
MINIBATCH_SIZE = 8
UPDATE_SECONDS_BUDGET = 20.0        # 0 disables the cap
EPOCHS_PER_UPDATE = 2               # passes over each rollout
# Optimizer. "adam" is the predictable choice: pair it with the warmup/cosine
# schedule below and convergence is monotone. "prodigy" needs no learning rate
# (leave OPTIMIZER_LR at 1.0) but it estimates its own step size, so the decay
# schedule below fights it; use Prodigy only when you set LR_DECAY_UPDATES high
# and LR_MIN_FRACTION to 1.0.
OPTIMIZER_KIND = "adam"             # adam | prodigy
OPTIMIZER_LR = 3e-4
LR_WARMUP_UPDATES = 20              # updates spent ramping the LR up
LR_DECAY_UPDATES = 5000             # updates over which it cosine-decays
LR_MIN_FRACTION = 0.1               # floor of the decay, as a fraction of base
GRAD_CLIP_NORM = 1.0                # was 0.5; 1.0 is the PPO default
# Transformer shape. num_heads must divide VISUAL_DIM.
VISUAL_DIM = 128
AUDIO_DIM = 64
HIDDEN_DIM = 128
NUM_BLOCKS = 2
NUM_HEADS = 4
FFN_HIDDEN = None                   # None = the standard 2/3 * 4 * d_model
DROPOUT = 0.0
SEQ_POOL = "last"                   # last | mean | max
AUDIO_DOWNSAMPLE = 8                # average-pool the waveform before the CNN

# ---- frozen-window watchdog -------------------------------------------------
# Warn when screen-grab frames stop changing for this long. Many games stop
# simulating or stop rendering while unfocused, which leaves the bot training
# on a still image; the only fix is a game setting or a different window mode.
# Set to 0 to disable.
FROZEN_WARN_SECONDS = 150.0

# ---- input ------------------------------------------------------------------
# How long a momentary press lasts (seconds). Buttons the bot wants to keep
# down are held for as many control steps as it likes, so this only applies to
# taps the action set defines (a menu key, a slot key, jump) and to clicks.
ACTION_TAP_SECONDS = 0.06

# Global injection switch, checked by every SendInput call. --calibrate turns
# this off for its whole duration, so no code path can press a key behind your
# back while you are demonstrating. Human takeovers do NOT use it: they flip
# BackgroundInput's own per-instance switch (see Input.suspended) instead, so a
# yield can be released and re-armed many times without being reported as a
# bug. Leave this alone unless you are debugging: set INJECTION_AUDIT to True
# to dump a stack trace whenever an injection is refused.
INPUT_INJECTION_ENABLED = True
INJECTION_AUDIT = False

# This design assumes a machine dedicated to the bot: the game window owns
# the foreground, so real keystrokes and real mouse movement both land in it.
# That is what makes mouse-look possible - injected input can only reach the
# focused window, and games read look from raw mouse deltas. Before each
# action the bot re-asserts focus on the target window, which is a no-op when
# the game already owns it.

# Mouse-look: pixels of cursor travel per turn action, as a cursor delta. Games
# turn by the raw mouse delta, so this is a deliberate, real cursor movement.
# --calibrate measures your own value and stores it in the keymap.
MOUSE_TURN_PIXELS = 60

# ---- keymap / calibration ---------------------------------------------------
# The JSON whitelist of keys the bot may press. Written by --calibrate, read by
# every other run. Delete it to fall back to the built-in defaults.
KEYMAP_PATH = "keymap.json"
# The start/stop key --calibrate listens for.
CALIBRATE_KEY = "f8"
# True = --calibrate starts from the built-in defaults and ignores the keymap
# that is already on disk, instead of merging into it.
CALIBRATE_FRESH_KEYMAP = False
# Ignore presses shorter than this during --calibrate (a stray tap on a key you
# do not actually play with). 0.0 records every press.
CALIBRATE_MIN_HOLD = 0.12
# How long to wait for the first toggle key once --calibrate starts.
CALIBRATE_WAIT_SECONDS = 300.0

# Virtual keys the recorder refuses to learn and the bot refuses to press.
# F1/F3 are hardware/debug overlays in many games; unbinding them keeps the bot
# from reaching into the game's own settings or its dev overlays. 0x70 = F1,
# 0x72 = F3. Extend this if your game has other "do not touch" keys.
RECORDER_BLOCKED_VKS = {0x70, 0x72}

# ---- human takeover (the bot hands you the controls and learns from you) ----
# The moment you touch the keyboard or the mouse, the bot lets go of every
# button it was holding and sends nothing at all. This is how long it waits
# after your last input before it takes the controls back.
HUMAN_RELEASE_GRACE_SECONDS = 3.0
# Count real mouse *movement* as you taking over. Set this to False if your game
# re-centres the cursor itself and that arrives as an un-injected move, which
# would look like you touching the mouse.
HUMAN_MOUSE_TAKEOVER = True
# Frames of your play kept in memory to learn from, stored at IMG_SIZE in RGB
# (~77 KB each at 160px, so 1500 frames is about 115 MB).
HUMAN_REPLAY_MAX_FRAMES = 1500
# A takeover with fewer recorded steps than this is not worth imitating.
HUMAN_SEGMENT_MIN_STEPS = 8
# Imitation of your play: one extra supervised gradient step after each PPO
# update, on a minibatch of (frames -> what you did), scaled by this weight. It
# is a nudge alongside PPO, not a second objective.
HUMAN_IMITATION_WEIGHT = 0.5
HUMAN_IMITATION_BATCH_SIZE = 32

# ---- one-shot diagnostics ---------------------------------------------------
# "" does nothing extra. Otherwise one of:
#   "capture"  are the grabbed pixels actually live? (BLACK / FROZEN / LIVE)
#   "input"    does the game react to injected input at all?
#   "safety"   prove that calibration cannot inject input
#   "release"  lift every key the bot could have left pressed
#   "windows"  list every window, for TARGET_WINDOW_HWND
#   "actions"  print the action list built from the keymap
# These were command-line flags before --calibrate was left as the only one.
DIAGNOSTIC = ""
DIAGNOSTIC_SECONDS = 5.0           # how long "capture" watches for
DIAG_INPUT_KEY = "E"               # the key "input" taps (inventory in most games)

# ---- calibrated action set --------------------------------------------------
# Upper bound on distinct button combinations the policy chooses between.
MAX_ACTIONS = 256

# ---- curiosity: what the game-agnostic reward actually pays for -------------
# This is the part that decides whether the bot explores the game or quietly
# learns to sit still. The defaults are deliberately biased toward *touching*
# the game:
#
#   * Nothing is paid for merely staying alive. A flat "you did not die" bonus
#     is the single easiest way to buy a policy that never presses a button,
#     because standing still is the cheapest way to keep it.
#   * Pressing nothing costs a little, and the longer the bot does it the more
#     it costs, so inaction has to be justified by something else rather than
#     being the free safe option.
#   * Trying an action in a situation where it has not been tried yet pays.
#     That is what makes "what does this button do?" a profitable question
#     instead of a risk.
#   * A large sudden change the bot caused is never charged - only one it did
#     not cause. Pressing things must not be the risky choice.
#
# If the bot still parks on noop, raise W_IDLE and W_NOVEL_PAIR. If it thrashes
# without ever settling on anything, lower W_RND/W_NOVEL_PAIR, lower
# ENTROPY_COEF_MAX, or raise BATCH_SIZE.
W_RND = 1.0              # RND novelty, conditioned on the action taken
W_NOVEL_STATE = 0.25     # first time this coarse view has ever been seen
W_NOVEL_PAIR = 0.75      # this action has not been tried in this view yet
W_CONTROL = 0.30         # the world moved more than it does with no input
CONTROL_SCALE = 0.05     # mean frame delta (0-1) that counts as full movement
CONTROL_CAP = 3.0        # ceiling on the scaled controllability term
RND_MIN_SCALE = 1e-3     # floor on the RND novelty divisor
RND_CAP = 5.0            # ceiling on the RND novelty term
W_IDLE = 0.02            # per-step cost of an action that presses nothing
IDLE_RAMP_STEPS = 20.0   # consecutive idle steps before that cost doubles
IDLE_RAMP_MAX = 5.0      # ceiling on the idle-cost multiplier
W_DISRUPTION = 1.0       # large change the bot did not cause (cutscene/respawn)
DISRUPTION_THRESHOLD = 0.5   # mean |frame delta|/255 that counts as "large"
RND_EMBED_DIM = 256      # width of the RND MLPs (the frame embedding is padded)
# Novelty memory is deliberately persistent rather than per-episode: an episode
# here is just a segment of a game that never restarts, so re-paying for the
# same room every 1000 steps would only teach the bot to go in circles. The cap
# bounds memory; when it is exceeded the oldest half is forgotten.
NOVELTY_MEMORY = 100_000

# ---- exploration pressure in PPO -------------------------------------------
# The entropy bonus is what stops the policy from collapsing onto one action
# and staying there. With ADAPTIVE_ENTROPY the coefficient is raised on its own
# whenever the policy's entropy drops under ENTROPY_TARGET_FRAC of the maximum
# entropy for this action space, and decays back toward ENTROPY_COEF once the
# policy is exploring again.
ENTROPY_COEF = 0.02
ADAPTIVE_ENTROPY = True
ENTROPY_TARGET_FRAC = 0.35
ENTROPY_COEF_MAX = 0.15
ENTROPY_COEF_STEP = 1.03     # growth per update while entropy is too low
ENTROPY_COEF_DECAY = 0.995   # decay per update once entropy is healthy

# Bumped whenever the reward or the action-conditioned RND input changes. A
# checkpoint written by an older version is refused (the run starts fresh)
# rather than silently resuming a policy that was trained against a different
# objective - which is exactly how you end up babysitting a noop policy.
REWARD_VERSION = 3


# =============================================================================
# SECTION 1: WINDOW FINDING & VISUAL CAPTURE (PrintWindow)
# =============================================================================

def get_window_pid(hwnd: int) -> int:
    """Return the process id that owns hwnd, or 0 if it cannot be read."""
    try:
        return win32process.GetWindowThreadProcessId(hwnd)[1]
    except Exception:
        return 0


def get_process_name(pid: int) -> str:
    """Return the executable file name for a pid, or '' if it cannot be read."""
    if not pid:
        return ""
    handle = None
    try:
        handle = win32api.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFO
        exe_path = win32process.GetModuleFileNameEx(handle, 0)
        return exe_path.rsplit("\\", 1)[-1].lower()
    except Exception:
        return ""
    finally:
        if handle:
            try:
                win32api.CloseHandle(handle)
            except Exception:
                pass


def get_client_area(hwnd: int) -> int:
    """Area of the window's client rect, in pixels (0 on failure)."""
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        return max(0, right - left) * max(0, bottom - top)
    except Exception:
        return 0


def is_minimized(hwnd: int) -> bool:
    try:
        return bool(win32gui.IsIconic(hwnd))
    except Exception:
        return False


def enumerate_windows(min_area: int = 64 * 64) -> List[dict]:
    """
    Return a list of visible, titled, non-tiny top-level windows with the
    metadata needed to pick one. Sorted by client area, largest first.
    """
    # Shell/desktop plumbing that is technically a window but never a target.
    skip_exes = {"textinputhost.exe", "shellexperiencehost.exe",
                 "searchhost.exe", "startmenuexperiencehost.exe",
                 "applicationframehost.exe"}
    skip_titles = {"program manager", "windows input experience",
                   "microsoft text input application", "default ime",
                   "windows shell experience host", "search"}

    windows: List[dict] = []

    def callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if not title.strip():
            return True

        area = get_client_area(hwnd)
        if area < min_area:
            return True

        pid = get_window_pid(hwnd)
        exe = get_process_name(pid) or "?"
        if exe.lower() in skip_exes or title.strip().lower() in skip_titles:
            return True

        windows.append({
            "hwnd": hwnd,
            "title": title,
            "pid": pid,
            "exe": exe,
            "area": area,
            "minimized": is_minimized(hwnd),
        })
        return True

    win32gui.EnumWindows(callback, None)
    windows.sort(key=lambda w: w["area"], reverse=True)
    return windows


def describe_window(win: dict) -> str:
    state = "  [minimized]" if win["minimized"] else ""
    return (f"{win['exe']:<22} {win['title'][:58]:<58} "
            f"{win['area']:>9,}px{state}")


def format_window_table(windows: List[dict], numbered: bool = True) -> str:
    lines = []
    for i, win in enumerate(windows):
        prefix = f"[{i:>2}] " if numbered else "     "
        lines.append(prefix + describe_window(win))
    return "\n".join(lines)


# Executables and title keywords that usually mean "a game". Used only to order
# the picker and to seed auto-detection - never to decide what the bot may do -
# so a game that is not listed here still works, it just has to be picked from
# the list. Add your own to the end of either tuple.
GAME_EXE_HINTS = (
    "unityplayer", "unity", "unreal", "ue4", "ue5", "godot", "gamemaker",
    "love.exe", "pygame", "javaw.exe", "java.exe", "mono.exe", "wine",
    "dota2", "cs2.exe", "csgo", "r5apex", "fortnite", "genshin", "roblox",
    "terraria", "factorio", "rimworld", "valheim", "amongus", "eldenring",
    "hd-2d", "game.exe",
)
GAME_TITLE_HINTS = (
    "unreal", "unity", "godot", "direct3d", "vulkan", "opengl",
    "fullscreen", "borderless",
)
# Applications that are definitely not the target, so the auto-pick never
# guesses a browser or an editor. Only the auto-detection uses this; the
# interactive picker still lists them.
NON_GAME_EXE_HINTS = (
    "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe",
    "explorer.exe", "code.exe", "devenv.exe", "pycharm", "idea64",
    "windowsterminal.exe", "powershell.exe", "pwsh.exe", "cmd.exe",
    "notepad.exe", "notepad++.exe", "winword.exe", "excel.exe", "powerpnt",
    "outlook.exe", "teams.exe", "discord.exe", "slack.exe", "spotify.exe",
    "vlc.exe", "mpc-hc", "obs64.exe", "python.exe", "pythonw.exe",
    # Desktop shell / settings surfaces that are large but never the target.
    "systemsettings.exe", "applicationframehost.exe", "searchhost.exe",
    "shellexperiencehost.exe", "startmenuexperiencehost.exe",
)


def looks_like_game(win: dict, strict: bool = False) -> bool:
    """
    Heuristic: is this window probably a game? Used only for list ordering and
    for seeding auto-detection, never to gate behaviour.

    strict=True (used by auto-detection) additionally rejects known non-games,
    so the bot never silently picks a browser because it happens to be large.
    """
    exe = str(win.get("exe", "")).lower()
    title = str(win.get("title", "")).lower()
    if strict and any(hint in exe or hint in title
                      for hint in NON_GAME_EXE_HINTS):
        return False
    if any(hint in exe for hint in GAME_EXE_HINTS):
        return True
    if any(hint in title for hint in GAME_TITLE_HINTS):
        return True
    # A big client area is weak evidence, only trusted when nothing marks the
    # window as an ordinary desktop application.
    if win.get("area", 0) >= 1280 * 720:
        return not any(hint in exe for hint in NON_GAME_EXE_HINTS)
    return False


def pick_window_interactive(
    games_first: bool = True,
    min_area: int = 64 * 64,
    show_all: bool = False,
) -> Optional[dict]:
    """
    Let the user choose which window to drive at runtime.

    By default the picker lists the windows most likely to be games first; press
    'a' to see every visible window instead. Returns a window dict or None if
    cancelled.
    """
    all_windows = enumerate_windows(min_area=min_area)
    if not all_windows:
        print("[picker] No visible, titled windows found.")
        return None

    likely = [w for w in all_windows if looks_like_game(w)]
    listed = all_windows if (show_all or not likely) else likely
    if games_first and not show_all:
        # Likely games first, then everything else by size.
        listed.sort(key=lambda w: (not looks_like_game(w), -w["area"]))

    print()
    print("=" * 100)
    print("SELECT A TARGET WINDOW"
          + ("" if show_all or not likely else "  (likely game windows)"))
    print("=" * 100)
    print(format_window_table(listed, numbered=True))
    print("-" * 100)
    if not show_all and likely:
        print(f"Showing {len(listed)} of {len(all_windows)} windows. "
              f"Type 'a' to show all windows.")
    print("Type a number to select, 'r' to refresh the list, or 'q' to quit.")

    while True:
        try:
            raw = input("Target window > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if raw in ("q", "quit", "exit"):
            return None
        if raw in ("r", "refresh"):
            return pick_window_interactive(games_first, min_area, show_all)
        if raw == "a":
            return pick_window_interactive(games_first, min_area, show_all=True)

        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(listed):
                chosen = listed[idx]
                print(f"[picker] Selected hwnd={chosen['hwnd']} "
                      f"({chosen['exe']}) title='{chosen['title']}'")
                if chosen["minimized"]:
                    print("[picker] WARNING: that window is minimized. PrintWindow")
                    print("         often returns a black frame when minimized -")
                    print("         restore the window (unfocused is fine) for capture.")
                return chosen
            print(f"[picker] '{raw}' is out of range (0-{len(listed) - 1}).")
        else:
            print("[picker] Enter a number, 'a', 'r' or 'q'.")


def find_game_window() -> Optional[int]:
    """
    Auto-detect a likely game window, largest first. Returns an hwnd or None.

    This is a convenience, not a requirement: anything not recognised still
    works, it just has to be chosen from the picker (or pinned with the
    TARGET_WINDOW_HWND constant). Known non-games (browsers, editors,
    terminals) are excluded so the bot never silently targets the wrong window.
    """
    candidates = [w for w in enumerate_windows(min_area=1)
                  if looks_like_game(w, strict=True)]
    if not candidates:
        print("[find] No window looked like a game; showing the window list.")
        return None

    best = candidates[0]  # enumerate_windows already sorts by area
    print(f"[find] Selected hwnd={best['hwnd']} ({best['exe']}) "
          f"title='{best['title']}'")
    if len(candidates) > 1:
        print(f"[find] ({len(candidates)} candidates; used the largest - "
              f"set TARGET_WINDOW_HWND, or PREFER_GAME_WINDOW = False, to "
              f"choose yourself)")
    return best["hwnd"]


def capture_window_printwindow(hwnd: int) -> Optional[np.ndarray]:
    """
    Capture the client area of a window using PrintWindow.
    Works even when the window is behind other windows or minimized.
    Returns BGR numpy array (H, W, 3) or None on failure.
    """
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()

    save_bitmap = win32ui.CreateBitmap()
    save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(save_bitmap)

    # PW_RENDERFULLCONTENT is required for DirectX / hardware-accelerated windows
    PW_RENDERFULLCONTENT = 0x00000002
    result = ctypes.windll.user32.PrintWindow(
        hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT
    )
    if result != 1:
        result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 0)

    img = None
    if result == 1:
        bmp_info = save_bitmap.GetInfo()
        bmp_str = save_bitmap.GetBitmapBits(True)
        img = np.frombuffer(bmp_str, dtype=np.uint8)
        img = img.reshape((bmp_info['bmHeight'], bmp_info['bmWidth'], 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    win32gui.DeleteObject(save_bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)
    return img


# =============================================================================
# SECTION 1.5: REAL-TIME INPUT CAPTURE, KEYMAP & CALIBRATED ACTION SETS
#
# This is what powers --calibrate (which keys exist) and the human-takeover
# watcher in SECTION 1.7 (what you did while you had the controls): it listens
# to the *real* keyboard and mouse through low-level Win32 hooks.
#
# It deliberately uses WH_KEYBOARD_LL / WH_MOUSE_LL rather than polling
# GetAsyncKeyState, because the low-level hooks also deliver raw relative mouse
# movement - the same signal the game uses for looking around - which polling
# cannot see. Every event also carries an "injected" flag, which is how the bot
# tells its own SendInput traffic apart from your hands.
# =============================================================================

# ---- low-level hook plumbing ----
WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_MBUTTONDOWN = 0x0207
WM_MBUTTONUP = 0x0208
WM_MOUSEWHEEL = 0x020A
WM_MOUSEMOVE = 0x0200

LLKHF_EXTENDED = 0x01
LLMHF_INJECTED = 0x00000001

# Virtual-key codes that are not in BackgroundInput.VK but that the recorder
# needs a name for. The sided modifier codes matter: the low-level hooks report
# VK_LSHIFT / VK_RSHIFT / VK_LCONTROL / ... (0xA0-0xA5) rather than the generic
# VK_SHIFT / VK_CONTROL, so without these names a shift or ctrl press would be
# named "VK_A0" and could never be matched to the LSHIFT / LCTRL entries that
# the keymap and the default hold keys use.
_EXTRA_VK_NAMES = {
    0x08: "BACKSPACE", 0x0D: "ENTER", 0x14: "CAPSLOCK", 0x2C: "PRINTSCREEN",
    0x5B: "LWIN", 0x5C: "RWIN", 0x5D: "APPS", 0x90: "NUMLOCK",
    0x91: "SCROLLLOCK", 0xA0: "LSHIFT", 0xA1: "RSHIFT", 0xA2: "LCTRL",
    0xA3: "RCTRL", 0xA4: "LALT", 0xA5: "RALT",
    0xBA: ";", 0xBB: "=", 0xBC: ",", 0xBD: "-",
    0xBE: ".", 0xBF: "/", 0xC0: "`", 0xDB: "[", 0xDC: "\\", 0xDD: "]",
    0xDE: "'",
}

_MOUSE_BUTTON_NAMES = {"left": "MOUSE_LEFT", "right": "MOUSE_RIGHT",
                       "middle": "MOUSE_MIDDLE"}

# Mouse button flags used by BackgroundInput._send_input_mouse.
_MOUSE_DOWN_FLAG = {"left": 0x0002, "right": 0x0008, "middle": 0x0020}
_MOUSE_UP_FLAG = {"left": 0x0004, "right": 0x0010, "middle": 0x0040}


def _vk_to_name(vk: int, extended: bool = False) -> str:
    """Turn a virtual-key code into the canonical name used everywhere else."""
    if 0x41 <= vk <= 0x5A:                     # A-Z
        return chr(vk)
    if 0x30 <= vk <= 0x39:                     # 0-9
        return chr(vk)
    if 0x60 <= vk <= 0x69:                     # numpad 0-9
        return f"NUMPAD{vk - 0x60}"
    if 0x70 <= vk <= 0x87:                     # F1-F24
        return f"F{vk - 0x6F}"
    if vk == 0x10:
        return "RSHIFT" if extended else "LSHIFT"
    if vk == 0x11:
        return "RCTRL" if extended else "LCTRL"
    if vk == 0x12:
        return "RALT" if extended else "LALT"
    return _EXTRA_VK_NAMES.get(vk, f"VK_{vk:02X}")


def calibratable_keys() -> Dict[str, int]:
    """
    Every key the bot is allowed to press, mapped to its virtual-key code.

    BackgroundInput.VK is the canonical source. The rest of the alphabet, the
    numpad, F1-F24 and the three mouse buttons are merged in so that a key you
    happen to press during calibration always gets a real name and a real code
    - otherwise a key the defaults never mentioned would be silently dropped
    from the whitelist.
    """
    keys = dict(BackgroundInput.VK)
    for vk in range(0x41, 0x5B):                       # A-Z
        keys.setdefault(chr(vk), vk)
    for vk in range(0x30, 0x3A):                       # 0-9
        keys.setdefault(chr(vk), vk)
    for i in range(10):                                # numpad 0-9
        keys.setdefault(f"NUMPAD{i}", 0x60 + i)
    for i in range(1, 25):                             # F1-F24
        keys.setdefault(f"F{i}", 0x6F + i)
    for vk, name in _EXTRA_VK_NAMES.items():
        keys.setdefault(name, vk)
    for name in _MOUSE_BUTTON_NAMES.values():          # mouse buttons are VKs
        keys.setdefault(name, {"MOUSE_LEFT": 0x01, "MOUSE_RIGHT": 0x02,
                               "MOUSE_MIDDLE": 0x04}[name])
    return keys


# Keys treated as "hold these while moving / looking" in the default keymap.
# Everything else the recorder sees becomes a momentary tap.
DEFAULT_HOLD_KEYS = ["W", "A", "S", "D", "SPACE", "LSHIFT", "LCTRL"]
DEFAULT_TAP_KEYS = ["E", "Q", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# `use` in an action spec means "hold the keys that the human was holding at
# this moment" - which is exact for recorded demonstrations and falls back to
# the spec / keymap default when there is no recording.
TRACKED_HOLD = "use"


class Keymap:
    """
    The bot's whitelist of input, plus a little self-tuning.

    Built once by --calibrate, then reloaded by every run. While a training run
    is watching you play, any *new* key you press is folded in and written back
    to disk, so the keymap grows with you instead of needing a fresh
    calibration run every time you rebind something. The action set is rebuilt
    from the keymap at each launch, so such a key becomes pressable then - and
    because the action space changes size when that happens, that launch starts
    a fresh policy.
    """

    VERSION = 1
    DEFAULT_PATH = KEYMAP_PATH

    def __init__(self, data: Optional[dict] = None):
        data = data or {}
        self.path: Optional[str] = data.get("_path")
        self.origin: str = data.get("origin", "default")
        self.created: str = data.get("created", datetime.now().isoformat(timespec="seconds"))
        self.updated: Optional[str] = data.get("updated")
        self.game: str = data.get("game") or ""
        self.game_hwnd: Optional[int] = data.get("game_hwnd")
        self.game_resolution = data.get("game_resolution") or [0, 0]
        self.samples: int = int(data.get("samples") or 0)
        self.allow_new_keys: bool = bool(data.get("allow_new_keys", True))
        self.default_mouse_turn: int = int(data.get("default_mouse_turn")
                                           or MOUSE_TURN_PIXELS)
        self.mouse_sensitivity = data.get("mouse_sensitivity") or {}
        self.actions: List[dict] = list(data.get("actions") or [])
        self.actions_capped: bool = bool(data.get("actions_capped", False))

        self.keys: Dict[str, dict] = {}
        for name, entry in (data.get("keys") or {}).items():
            if isinstance(entry, int):                      # tolerate a bare map
                entry = {"vk": entry}
            self.keys[str(name).upper()] = {
                "vk": int(entry.get("vk", 0)),
                "hold": entry.get("hold", "hold" if str(name).upper() in
                                  DEFAULT_HOLD_KEYS else "tap"),
                "source": entry.get("source", "calibrated"),
                "count": int(entry.get("count") or 0),
                "held_seconds": float(entry.get("held_seconds") or 0.0),
                "max_hold": float(entry.get("max_hold") or 0.0),
            }

        self.mouse_buttons: Dict[str, dict] = {}
        for name, entry in (data.get("mouse_buttons") or {}).items():
            self.mouse_buttons[str(name).upper()] = {
                "count": int(entry.get("count") or 0),
                "source": entry.get("source", "calibrated"),
            }

    # ---- construction ----
    @classmethod
    def load(cls, path: str, required: bool = False) -> Optional["Keymap"]:
        """Read a keymap JSON. Returns None when it is missing (unless required)."""
        if not path or not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"No keymap at '{path}'")
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"[Keymap] Could not read '{path}': {exc}")
            if required:
                raise
            return None
        if not isinstance(data, dict):
            print(f"[Keymap] '{path}' is not a keymap object; ignoring it.")
            return None
        data["_path"] = os.path.abspath(path)
        keymap = cls(data)
        print(f"[Keymap] Loaded '{os.path.abspath(path)}' "
              f"({len(keymap.keys)} keys, {len(keymap.actions)} actions, "
              f"origin={keymap.origin})")
        return keymap

    @classmethod
    def default(cls) -> "Keymap":
        """The built-in whitelist, used when no calibrate run has happened yet."""
        keymap = cls({
            "origin": "default",
            "game": "generic defaults (FPS-style layout)",
            "allow_new_keys": True,
            "default_mouse_turn": MOUSE_TURN_PIXELS,
        })
        for name in DEFAULT_HOLD_KEYS + DEFAULT_TAP_KEYS:
            keymap.allow(name, source="default", hold="hold" if name in
                         DEFAULT_HOLD_KEYS else "tap")
        for button in ("left", "right", "middle"):
            keymap.allow(_MOUSE_BUTTON_NAMES[button], source="default")
        keymap.rebuild_actions()
        return keymap

    # ---- lookups ----
    def vk(self, name: str) -> Optional[int]:
        entry = self.keys.get(str(name).upper())
        if entry and entry.get("vk"):
            return int(entry["vk"])
        return BackgroundInput.VK.get(str(name).upper())

    def hold_name(self, vk: int) -> str:
        return f"HOLD_{vk:02X}"

    def is_hold(self, name: str) -> bool:
        """True for a hold key, a tracked-hold token, or a HOLD_xx placeholder."""
        upper = str(name).upper()
        if upper == TRACKED_HOLD.upper() or upper.startswith("HOLD_"):
            return True
        entry = self.keys.get(upper)
        return bool(entry and entry.get("hold") == "hold")

    def hold_vk(self, name: str) -> Optional[int]:
        """Virtual key of a hold token; HOLD_xx tokens carry their own code."""
        upper = str(name).upper()
        if upper.startswith("HOLD_"):
            try:
                return int(upper.split("_", 1)[1], 16)
            except (IndexError, ValueError):
                return None
        return self.vk(upper)

    def mouse_button_name(self, button: str) -> Optional[str]:
        name = _MOUSE_BUTTON_NAMES.get(str(button).lower())
        if name and name in self.mouse_buttons:
            return name
        return None

    def button_vk(self, name: str) -> Optional[int]:
        """Mouse buttons are real VKs too (VK_LBUTTON..) so holds work."""
        return {"MOUSE_LEFT": 0x01, "MOUSE_RIGHT": 0x02,
                "MOUSE_MIDDLE": 0x04}.get(str(name).upper())

    # ---- mutation ----
    def allow(self, name: str, source: str = "default", vk: Optional[int] = None,
              hold: Optional[str] = None) -> bool:
        """
        Add (or refresh) a key or mouse button in the whitelist.

        Returns True when this call added something new, which is what makes
        the human-takeover discovery path easy to report.
        """
        upper = str(name).upper()
        if upper in _MOUSE_BUTTON_NAMES.values():
            if upper in self.mouse_buttons:
                return False
            self.mouse_buttons[upper] = {"count": 0, "source": source}
            return True
        if upper in self.keys:
            if vk and not self.keys[upper].get("vk"):
                self.keys[upper]["vk"] = int(vk)
            return False
        code = vk if vk is not None else calibratable_keys().get(upper)
        if not code:
            return False
        if hold is None:
            hold = "hold" if upper in DEFAULT_HOLD_KEYS else "tap"
        self.keys[upper] = {"vk": int(code), "hold": hold, "source": source,
                            "count": 0, "held_seconds": 0.0, "max_hold": 0.0}
        return True

    def note_usage(self, name: str, held_seconds: float = 0.0,
                   source: str = "calibrated"):
        """Fold one observed press into a key's statistics."""
        upper = str(name).upper()
        if upper in _MOUSE_BUTTON_NAMES.values():
            if self.allow(upper, source=source):
                pass
            self.mouse_buttons[upper]["count"] += 1
            return
        self.allow(upper, source=source)
        entry = self.keys.get(upper)
        if entry is None:
            return
        entry["count"] += 1
        entry["held_seconds"] += max(0.0, float(held_seconds))
        entry["max_hold"] = max(entry["max_hold"], float(held_seconds))

    def note_mouse_turn(self, pixels: float):
        """Running mean of how far the human turned the view in one step."""
        stats = self.mouse_sensitivity or {}
        count = int(stats.get("samples") or 0) + 1
        mean = float(stats.get("pixels_per_step") or 0.0)
        mean += (float(pixels) - mean) / count
        self.mouse_sensitivity = {
            "samples": count,
            "pixels_per_step": mean,
            "max_pixels_per_step": max(float(stats.get("max_pixels_per_step") or 0.0),
                                       float(pixels)),
        }

    def set_actions(self, actions: List[dict], capped: bool = False):
        self.actions = list(actions)
        self.actions_capped = bool(capped)

    def rebuild_actions(self) -> List[dict]:
        """Derive the action list from the current whitelist (see build_action_set)."""
        actions, capped = build_action_set(self)
        self.set_actions(actions, capped)
        return self.actions

    # ---- persistence ----
    def to_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "origin": self.origin,
            "created": self.created,
            "updated": datetime.now().isoformat(timespec="seconds"),
            "game": self.game,
            "game_hwnd": self.game_hwnd,
            "game_resolution": list(self.game_resolution),
            "samples": self.samples,
            "allow_new_keys": self.allow_new_keys,
            "default_mouse_turn": self.default_mouse_turn,
            "mouse_sensitivity": self.mouse_sensitivity,
            "keys": self.keys,
            "mouse_buttons": self.mouse_buttons,
            "actions": self.actions,
            "actions_capped": self.actions_capped,
        }

    def save(self, path: Optional[str] = None) -> str:
        path = os.path.abspath(path or self.path or self.DEFAULT_PATH)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2, sort_keys=False)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        self.path = path
        return path

    # ---- reporting ----
    def describe_keys(self) -> str:
        holds = [n for n, e in self.keys.items() if e.get("hold") == "hold"]
        taps = [n for n, e in self.keys.items() if e.get("hold") != "hold"]
        buttons = list(self.mouse_buttons)
        return (f"hold=[{', '.join(sorted(holds)) or '-'}]  "
                f"tap=[{', '.join(sorted(taps)) or '-'}]  "
                f"mouse=[{', '.join(sorted(buttons)) or '-'}]")


def _key_priority(name: str) -> int:
    """Ordering for action lists: the keys a bot needs most come first."""
    table = {"W": 0, "A": 1, "S": 2, "D": 3, "SPACE": 4, "LSHIFT": 5,
             "LCTRL": 6, "RCTRL": 6, "RSHIFT": 5, "MOUSE_LEFT": 7,
             "MOUSE_RIGHT": 8, "MOUSE_MIDDLE": 9}
    return table.get(str(name).upper(), 50)


def _hold_tokens(keymap: Keymap) -> List[str]:
    """
    Hold tokens allowed by the keymap.

    Defaults come first so the core movement actions always exist; every
    calibrated hold key is appended after them.
    """
    tokens: List[str] = []
    for name in DEFAULT_HOLD_KEYS:
        if keymap.is_hold(name):
            tokens.append(keymap.hold_name(keymap.vk(name) or 0))
    for name, entry in sorted(keymap.keys.items(),
                              key=lambda kv: (-kv[1].get("count", 0), kv[0])):
        if entry.get("hold") != "hold":
            continue
        token = keymap.hold_name(keymap.vk(name) or 0)
        if token not in tokens:
            tokens.append(token)
    for name in keymap.mouse_buttons:
        token = keymap.hold_name(keymap.button_vk(name) or 0)
        if token not in tokens:
            tokens.append(token)
    return tokens


def build_action_set(keymap: Optional[Keymap] = None,
                     cap: int = MAX_ACTIONS) -> Tuple[List[dict], bool]:
    """
    Turn a keymap into the discrete action list the policy chooses from.

    Each action is a bundle of things the bot does for one control step:

        * ``held``  - buttons held down for the duration of the step. These are
                      real keys that stay pressed, so the bot can walk, sprint,
                      sneak or swim continuously instead of re-tapping.
        * ``uses``  - the same thing, for a combination that came out of your
                      own play (a recorded "replay:" action). Both fields are
                      resolved to virtual keys the same way when the action is
                      applied.
        * ``tap``   - a button pressed and released inside the step.
        * ``click`` - a mouse button clicked inside the step.
        * ``look``  - a relative mouse turn, which is the only thing games'
                      mouse-look responds to.

    One action covers buttons + a click + a turn at once, so the bot can learn
    combinations like "hold W, jump" or "hold W and D, click" as single
    decisions, and it can hold up to four buttons at once when you demonstrated
    such a combination. The list is capped; when it overflows, the least
    important combinations are dropped and ``capped`` comes back True.
    """
    keymap = keymap or Keymap.default()

    ordered: List[Tuple[int, Tuple, dict]] = []
    seen = set()
    covered_button_sets = {("noop",)}

    def add(priority: int, signature: Tuple, action: dict):
        if signature in seen or len(ordered) >= cap:
            return
        seen.add(signature)
        ordered.append((priority, signature, action))

    def add_hold(tokens: List[str], priority: int):
        """Register a plain hold action and remember its button set."""
        sortable = tuple(sorted(str(t) for t in tokens))
        if sortable in covered_button_sets:
            return
        covered_button_sets.add(sortable)
        add(priority, ("hold",) + tuple(tokens),
            {"label": "hold:" + "+".join(_hold_label(keymap, t) for t in tokens),
             "held": list(tokens), "uses": [], "tap": None,
             "click": None, "look": None})

    add(0, ("noop",), {"label": "noop", "held": [], "uses": [],
                       "tap": None, "click": None, "look": None})

    holds = _hold_tokens(keymap)

    def hold_tokens_for(names) -> List[str]:
        tokens = [keymap.hold_name(keymap.vk(n) or 0) for n in names
                  if keymap.is_hold(n)]
        return [t for t in tokens if t in holds]

    movement = hold_tokens_for(("W", "A", "S", "D"))
    timed = hold_tokens_for(("SPACE", "LSHIFT", "LCTRL"))

    # Continuous holds on their own - the core of "hold the button down".
    for token in holds:
        add_hold([token], 2)

    # Diagonals (W+A, W+D, ...) and the pairs inside the timed group, so the
    # bot can move at an angle, or jump and sprint, as one decision.
    for group in (movement, timed):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                add_hold([group[i], group[j]], 3)

    # Movement crossed with the timed buttons - "run forward", "sprint at an
    # angle", "jump while strafing" - which neither group can express on its
    # own, and which a player does constantly.
    for move in movement:
        for extra in timed:
            add_hold([move, extra], 4)

    # Momentary taps of non-hold keys, if the keymap allows them.
    for name in sorted(keymap.keys, key=_key_priority):
        if keymap.keys[name].get("hold") == "hold":
            continue
        add(10, ("tap", name), {"label": f"tap:{name}", "held": [], "uses": [],
                                "tap": name, "click": None, "look": None})

    # Looking around and clicking, which any first-person game needs.
    turn = max(4, int(keymap.default_mouse_turn))
    for label, dx, dy in (("look_left", -turn, 0), ("look_right", turn, 0),
                          ("look_up", 0, -turn), ("look_down", 0, turn)):
        add(5, ("look", label), {"label": label, "held": [], "uses": [],
                                 "tap": None, "click": None,
                                 "look": [dx, dy]})
    for button in ("left", "right", "middle"):
        if keymap.mouse_button_name(button):
            add(6, ("click", button), {"label": f"click:{button}", "held": [],
                                       "uses": [], "tap": None,
                                       "click": button, "look": None})
    if keymap.mouse_button_name("left"):
        forward = movement[:1]
        for label, dx, dy in (("look_left", -turn, 0), ("look_right", turn, 0)):
            add(8, ("hold", tuple(forward), "look", label),
                {"label": (f"hold:{_hold_label(keymap, forward[0])}+{label}"
                           if forward else label),
                 "held": list(forward), "uses": [], "tap": None,
                 "click": None, "look": [dx, dy]})

    # Button combinations from your own play, so the policy can reproduce
    # something the defaults never imagined. Each stored button is resolved to
    # a real key of *this* keymap: a combination that this keymap cannot press
    # is skipped outright rather than quietly pressed in part, which is what
    # used to turn "hold W and space and ctrl" into "hold W".
    for spec in keymap.actions:
        stored = list(spec.get("uses") or []) + list(spec.get("held") or [])
        if not stored:
            continue
        buttons: List[str] = []
        for token in stored:
            canon = _canonical_hold_token(keymap, token)
            if canon is None:
                buttons = []
                break
            if canon not in buttons:
                buttons.append(canon)     # keep the order you pressed them in
        if not buttons:
            continue                      # a combination this keymap cannot press
        if tuple(sorted(buttons)) in covered_button_sets:
            continue                      # already covered by a generated action
        covered_button_sets.add(tuple(sorted(buttons)))
        look = tuple(spec["look"]) if spec.get("look") else ()
        add(1, ("uses", tuple(buttons), look),
            {"label": "replay:" + "+".join(_hold_label(keymap, t) for t in buttons),
             "held": [], "uses": list(buttons), "tap": None, "click": None,
             "look": list(look) if look else None})

    ordered.sort(key=lambda item: item[0])
    dropped = max(0, len(seen) - len(ordered)) if len(ordered) >= cap else 0
    actions = []
    for index, (_priority, _signature, action) in enumerate(ordered):
        entry = dict(action)
        entry["id"] = index
        actions.append(entry)

    if not actions:
        actions = [{"id": 0, "label": "noop", "held": [], "uses": [],
                    "tap": None, "click": None, "look": None}]
    return actions, bool(dropped)


def _hold_label(keymap: Keymap, token: Optional[str]) -> str:
    """Readable name for a HOLD_xx token, e.g. HOLD_57 -> W."""
    if not token:
        return "?"
    vk = keymap.hold_vk(token)
    if not vk:
        return str(token)
    for name, entry in keymap.keys.items():
        if int(entry.get("vk") or 0) == vk:
            return name
    for name in keymap.mouse_buttons:
        if keymap.button_vk(name) == vk:
            return name
    return f"0x{vk:02X}"


# A button name an older build wrote down because it could not name the key:
# the low-level hooks report the *sided* modifier codes (VK_LSHIFT == 0xA0,
# VK_LCONTROL == 0xA2, ...) and the name table used to have no entry for them,
# so a recorded "hold shift" was stored as the literal string "VK_A0". Those
# names match no key in keymap.json, so they were dropped at press time.
_STALE_VK_TOKEN = re.compile(r"^VK_([0-9A-F]{1,2})$")


def _keymap_vks(keymap: Keymap) -> set:
    """Every virtual key this keymap is allowed to press."""
    vks = {int(entry.get("vk") or 0) for entry in keymap.keys.values()}
    vks |= {keymap.button_vk(name) or 0 for name in keymap.mouse_buttons}
    vks.discard(0)
    return vks


def _canonical_hold_token(keymap: Keymap, token) -> Optional[str]:
    """
    The canonical HOLD_xx token for a stored button name, or None when this
    keymap cannot press it.

    Handles the three shapes that turn up in a saved action list: a HOLD_xx
    token, a plain key name ("W", "SPACE", "MOUSE_LEFT"), and a stale "VK_xx"
    name from an older recording, which is decoded and looked up by code so
    that "VK_A2" becomes the LCTRL the keymap already knows about.
    """
    text = str(token or "").upper().strip()
    if not text:
        return None
    known = _keymap_vks(keymap)

    if text.startswith("HOLD_"):
        vk = keymap.hold_vk(text)
        return keymap.hold_name(vk) if vk in known else None

    vk = keymap.vk(text) or keymap.button_vk(text)
    if vk:
        return keymap.hold_name(int(vk))

    match = _STALE_VK_TOKEN.match(text)
    if match and int(match.group(1), 16) in known:
        return keymap.hold_name(int(match.group(1), 16))
    return None


class ActionRepair:
    """What normalize_action_set changed, so it can be reported and saved."""

    def __init__(self):
        self.renamed: Dict[str, str] = {}
        self.repaired = 0            # button names translated to real keys
        self.dropped_buttons = 0     # buttons this keymap cannot press
        self.dropped_unpressable = 0  # actions dropped because of those
        self.dropped_actions = 0     # actions that duplicated an earlier one
        self.kept = 0

    @property
    def changed(self) -> bool:
        return bool(self.repaired or self.dropped_buttons
                    or self.dropped_actions or self.dropped_unpressable)

    def summary(self) -> str:
        parts = []
        if self.repaired:
            shown = ", ".join(f"{old} -> {new}"
                              for old, new in sorted(self.renamed.items())[:6])
            parts.append(f"translated {self.repaired} stored button name(s) "
                         f"({shown})")
        if self.dropped_unpressable:
            parts.append(f"removed {self.dropped_unpressable} action(s) this "
                         f"keymap cannot press")
        if self.dropped_buttons and not self.dropped_unpressable:
            parts.append(f"dropped {self.dropped_buttons} button(s) this keymap "
                         f"cannot press")
        if self.dropped_actions:
            parts.append(f"removed {self.dropped_actions} action(s) that did "
                         f"exactly what another one does")
        return "; ".join(parts) if parts else "no changes"


def _action_signature(keymap: Keymap, action: dict) -> tuple:
    """
    What an action actually does, as a comparable key: the buttons it presses
    (by virtual key, so a name and its HOLD_xx token compare equal), the turn,
    the tap and the click.
    """
    vks = {keymap.hold_vk(token) or 0
           for token in list(action.get("held") or []) + list(action.get("uses") or [])}
    vks.discard(0)
    tap = action.get("tap")
    return (tuple(sorted(vks)),
            tuple(action.get("look") or ()),
            str(tap).upper() if tap else "",
            str(action.get("click") or ""))


def normalize_action_set(keymap: Keymap, actions: List[dict]
                         ) -> Tuple[List[dict], ActionRepair]:
    """
    Make a stored action list pressable, and free of clones.

    Two things happen here:

      * every button name is resolved against this keymap, so a name an older
        build could not interpret ("VK_20" for space) becomes the real button
        instead of being skipped when the action is applied. An action whose
        buttons this keymap does not have at all is removed rather than
        silently weakened - "hold W, space and ctrl" must not quietly become
        "hold W";
      * actions that would do exactly the same thing are collapsed, so the
        policy is not choosing between clones and splitting its probability
        across them.

    Ids are renumbered, because the id *is* the policy's action index.
    """
    out: List[dict] = []
    seen = set()
    report = ActionRepair()

    for action in actions:
        spec = dict(action)
        unpressable = False
        for field in ("held", "uses"):
            resolved = []
            for token in (spec.get(field) or []):
                canon = _canonical_hold_token(keymap, token)
                if canon is None:
                    report.dropped_buttons += 1
                    unpressable = True
                    continue
                if canon != str(token).upper():
                    report.repaired += 1
                    report.renamed[str(token)] = _hold_label(keymap, canon)
                    spec["label"] = str(spec.get("label") or "").replace(
                        str(token), _hold_label(keymap, canon))
                resolved.append(canon)
            spec[field] = resolved
        if unpressable:
            # Deliberately not kept as a weakened version of itself: "hold W,
            # space and ctrl" must not quietly become "hold W".
            report.dropped_unpressable += 1
            continue

        tap = spec.get("tap")
        if tap:
            if not (keymap.vk(tap) or BackgroundInput.VK.get(str(tap).upper())):
                canon = _canonical_hold_token(keymap, tap)
                if canon:
                    report.repaired += 1
                    report.renamed[str(tap)] = _hold_label(keymap, canon)
                    spec["label"] = str(spec.get("label") or "").replace(
                        str(tap), _hold_label(keymap, canon))
                    spec["tap"] = _hold_label(keymap, canon)
                else:
                    report.dropped_unpressable += 1
                    continue

        signature = _action_signature(keymap, spec)
        if signature in seen:
            report.dropped_actions += 1
            continue
        seen.add(signature)
        out.append(spec)

    if not out:
        out = [{"label": "noop", "held": [], "uses": [], "tap": None,
                "click": None, "look": None}]
    for index, spec in enumerate(out):
        spec["id"] = index
    report.kept = len(out)
    return out, report


def action_list_signature(actions: List[dict]) -> tuple:
    """Comparable form of a whole action list, for 'did the rebuild change it?'"""
    return tuple((a.get("label"),
                  tuple(a.get("held") or []),
                  tuple(a.get("uses") or []),
                  tuple(a.get("look") or ()),
                  a.get("tap") or "",
                  a.get("click") or "")
                 for a in actions)


def describe_actions(actions: List[dict]) -> str:
    lines = [f"  {a['id']:>3}  {a['label']}" for a in actions]
    return "\n".join(lines)


class RecordedInput:
    """
    A timeline of real keyboard/mouse activity, no matter where it came from.

    Events are ``(t, kind, value, pressed)`` with ``t`` in seconds since the
    recorder started. ``kind`` is 'key' (value = key name), 'mouse_btn'
    (value = left/right/middle) or 'mouse_move' (value = (dx, dy)).

    Both the calibration recorder and the watch recorder produce these, which
    is why the analysis code below only has to be written once.
    """

    def __init__(self):
        self.t0 = time.perf_counter()
        self.events: List[Tuple[float, str, object, bool]] = []
        self.frame_times: List[float] = []
        self.lock = threading.Lock()

    def stamp(self) -> float:
        return time.perf_counter() - self.t0

    def add(self, kind: str, value, pressed: bool = True, t: Optional[float] = None):
        event = (self.stamp() if t is None else float(t), kind, value, bool(pressed))
        with self.lock:
            self.events.append(event)
        return event

    def snapshot(self) -> List[Tuple[float, str, object, bool]]:
        with self.lock:
            return list(self.events)

    def duration(self) -> float:
        with self.lock:
            return self.events[-1][0] if self.events else 0.0

    def key_holds(self) -> Dict[str, dict]:
        """Per-key press counts and hold durations, for the keymap statistics."""
        stats: Dict[str, dict] = {}
        open_at: Dict[str, float] = {}
        with self.lock:
            events = list(self.events)
        for t, kind, value, pressed in events:
            if kind == "mouse_move":
                continue
            name = value if kind == "mouse_btn" else str(value)
            if kind == "mouse_btn":
                name = _MOUSE_BUTTON_NAMES.get(str(value), str(value))
            if pressed:
                stats.setdefault(name, {"count": 0, "held_seconds": 0.0,
                                        "max_hold": 0.0})
                stats[name]["count"] += 1
                open_at[name] = t
            else:
                if name in open_at:
                    held = max(0.0, t - open_at.pop(name))
                    stats.setdefault(name, {"count": 1, "held_seconds": 0.0,
                                            "max_hold": 0.0})
                    stats[name]["held_seconds"] += held
                    stats[name]["max_hold"] = max(stats[name]["max_hold"], held)
        return stats

    def active_at(self, times: List[float]) -> List[set]:
        """
        For each timestamp, the set of buttons that were held at that moment.

        Key-up is processed before key-down at identical timestamps so that a
        tap does not leak into the following frame.
        """
        with self.lock:
            events = sorted(self.events, key=lambda e: (e[0], e[3]))
        held: set = set()
        out: List[set] = []
        cursor = 0
        for t in times:
            while cursor < len(events) and events[cursor][0] <= t:
                _ts, kind, value, pressed = events[cursor]
                cursor += 1
                if kind == "mouse_move":
                    continue
                name = (_MOUSE_BUTTON_NAMES.get(str(value), str(value))
                        if kind == "mouse_btn" else str(value))
                if pressed:
                    held.add(name)
                else:
                    held.discard(name)
            out.append(set(held))
        return out

    def move_deltas(self, times: List[float]) -> List[Tuple[float, float]]:
        """Sum of (dx, dy) between consecutive timestamps."""
        with self.lock:
            moves = sorted([e for e in self.events if e[1] == "mouse_move"],
                           key=lambda e: e[0])
        out: List[Tuple[float, float]] = []
        cursor = 0
        prev = 0.0
        for t in times:
            dx = dy = 0.0
            while cursor < len(moves) and moves[cursor][0] <= t:
                mx, my = moves[cursor][2]
                dx += float(mx)
                dy += float(my)
                cursor += 1
            out.append((dx, dy))
            prev = t
        return out

    def button_sets(self, times: Optional[List[float]] = None
                    ) -> Dict[frozenset, int]:
        """
        How much time was spent with each set of simultaneously held buttons.

        This is what grows the action set from your play: the combinations you
        actually used, weighted by how long you held them. When `times` is
        given (the recorded frame timestamps) the weighting follows the frames
        the model will actually train on; otherwise the raw event stream is
        used. Only real multi-button sets and single holds are counted - the
        empty set means "nothing held", which is already the no-op action.
        """
        times = times or self.frame_times
        if times:
            samples = self.active_at(times)
        else:
            with self.lock:
                events = sorted(self.events, key=lambda e: (e[0], e[3]))
            held: set = set()
            samples = []
            for _t, kind, value, pressed in events:
                if kind == "mouse_move":
                    continue
                name = (_MOUSE_BUTTON_NAMES.get(str(value), str(value))
                        if kind == "mouse_btn" else str(value))
                if pressed:
                    held.add(name)
                else:
                    held.discard(name)
                samples.append(set(held))

        counts: Dict[frozenset, int] = {}
        for buttons in samples:
            if buttons:
                key = frozenset(buttons)
                counts[key] = counts.get(key, 0) + 1
        return counts


class InputRecorder:
    """
    Listens to the real keyboard and mouse with Win32 low-level hooks.

    Two jobs, selected by the flags passed in:

      * calibration - listen for F8 and record *which keys exist* for you;
      * watching    - record keys, mouse buttons and raw mouse deltas onto a
                      shared timeline while you play, for imitation learning.

    Hooks must be installed and removed from the same thread, and that thread
    must pump a Win32 message loop, which is why start()/stop() bracket the
    whole capture session rather than being called per event.
    """

    def __init__(self, keymap: Optional["Keymap"] = None,
                 target_hwnd: Optional[int] = None,
                 toggle_vk: int = 0x77,
                 record_only_when_focused: bool = True,
                 allow_new_keys: bool = True,
                 timeline: Optional[RecordedInput] = None,
                 verbose: bool = True):
        self.keymap = keymap
        self.target_hwnd = target_hwnd
        self.toggle_vk = int(toggle_vk)
        self.record_only_when_focused = bool(record_only_when_focused)
        self.allow_new_keys = bool(allow_new_keys)
        self.timeline = timeline or RecordedInput()
        self.verbose = bool(verbose)

        self.running = False
        self.recording = False
        self.ignored_focus = False
        self.toggles = 0
        self.last_toggle = 0.0
        self.key_events = 0
        self.mouse_events = 0
        self.ignored_events = 0
        self.new_keys: List[str] = []
        self.errors: List[str] = []
        self.pressed: set = set()
        self._key_down_at: Dict[str, float] = {}
        self._blocked: set = set()
        self._mouse_pos: Optional[Tuple[int, int]] = None

        self._kb_proc = None
        self._ms_proc = None
        # Hook handles are named distinctly from the _kb_hook/_ms_hook methods
        # below; reusing those names would overwrite the methods with an int.
        self._kb_hook_handle = None
        self._ms_hook_handle = None
        self._user32 = ctypes.windll.user32

    # ---- lifecycle ----
    def start(self) -> bool:
        """Install both hooks. Returns True when at least the keyboard hook is up."""
        if self.running:
            return True
        self.running = True

        # Wrap the bound methods in plain functions: a ctypes callback type
        # wants a callable, and passing a bound method is not portable. The
        # wrapper also makes sure a Python exception can never escape into the
        # OS hook chain - an exception there makes Windows drop the hook and
        # the recorder silently stops seeing anything.
        def keyboard_proc(code, wparam, lparam):
            try:
                return self._kb_hook(code, wparam, lparam)
            except Exception:
                return 0

        def mouse_proc(code, wparam, lparam):
            try:
                return self._ms_hook(code, wparam, lparam)
            except Exception:
                return 0

        self._kb_proc = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t,
            ctypes.c_ssize_t)(keyboard_proc)
        self._ms_proc = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t,
            ctypes.c_ssize_t)(mouse_proc)

        for label, hook_id, proc, attr in (
            ("keyboard", WH_KEYBOARD_LL, self._kb_proc, "_kb_hook_handle"),
            ("mouse", WH_MOUSE_LL, self._ms_proc, "_ms_hook_handle"),
        ):
            try:
                handle = self._user32.SetWindowsHookExW(hook_id, proc, None, 0)
            except Exception as exc:
                handle = None
                self.errors.append(f"{label} hook raised {exc}")
            if not handle:
                self.errors.append(
                    f"{label} hook could not be installed (error "
                    f"{ctypes.get_last_error() if hasattr(ctypes, 'get_last_error') else '?'})")
            else:
                setattr(self, attr, handle)
        return self._kb_hook_handle is not None

    def stop(self):
        self.flush_pressed()
        self.running = False
        self.recording = False
        for attr in ("_kb_hook_handle", "_ms_hook_handle"):
            handle = getattr(self, attr)
            if handle:
                try:
                    self._user32.UnhookWindowsHookEx(handle)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._kb_proc = None
        self._ms_proc = None

    def pump(self, timeout_ms: int = 20) -> None:
        """
        Drain pending messages so Windows keeps calling the hook procedures.

        A thread that installs a low-level hook and then blocks without pumping
        gets its hook silently dropped by the OS after a short timeout.
        """
        class _MSG(ctypes.Structure):
            _fields_ = [("hwnd", ctypes.c_void_p), ("message", ctypes.c_uint),
                        ("wParam", ctypes.c_void_p), ("lParam", ctypes.c_void_p),
                        ("time", ctypes.c_uint), ("pt_x", ctypes.c_long),
                        ("pt_y", ctypes.c_long)]

        msg = _MSG()
        while self._user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
            self._user32.TranslateMessage(ctypes.byref(msg))
            self._user32.DispatchMessageW(ctypes.byref(msg))

    # ---- recording state ----
    def toggle(self):
        self.toggles += 1
        self.last_toggle = time.perf_counter()
        self.recording = not self.recording
        if self.recording:
            # Forget the cursor position from before recording started, or the
            # first mouse delta would be whatever you did while reading the
            # instructions.
            self._mouse_pos = None
        else:
            self.flush_pressed()
        return self.recording

    def flush_pressed(self):
        """Close out any keys still held when recording stops."""
        for name in list(self.pressed):
            self._note_release(name, self.timeline.stamp())

    def _note_release(self, name: str, t: float):
        # Guard against a second release for the same press (a duplicated hook
        # call would otherwise double-count the hold time and write a stray
        # key-up into the recorded timeline).
        if name not in self.pressed and name not in self._key_down_at:
            return
        started = self._key_down_at.pop(name, None)
        if name.startswith("MOUSE_"):
            self.timeline.add("mouse_btn", _MOUSE_BUTTON_NAMES_REV.get(name, name),
                              False, t=t)
        else:
            self.timeline.add("key", name, False, t=t)
        if started is not None and self.keymap is not None:
            held = max(0.0, t - started)
            entry = self.keymap.keys.get(name.upper())
            if entry is not None:
                entry["held_seconds"] += held
                entry["max_hold"] = max(entry["max_hold"], held)
        self.pressed.discard(name)

    def _wants_input(self) -> bool:
        if not self.record_only_when_focused or not self.target_hwnd:
            return True
        try:
            return self._user32.GetForegroundWindow() == self.target_hwnd
        except Exception:
            return True

    # ---- hooks ----
    def _kb_hook(self, code, wparam, lparam):
        try:
            if code == 0:
                vk = int(ctypes.cast(lparam, ctypes.POINTER(ctypes.c_uint32))[0]) & 0xFF
                pressed = wparam in (WM_KEYDOWN, WM_SYSKEYDOWN)
                if vk == self.toggle_vk and pressed:
                    # The mode toggle must work even while we ignore focus, or
                    # you could never stop a recording.
                    if time.perf_counter() - self.last_toggle > 0.35:
                        self.toggle()
                    return self._user32.CallNextHookEx(None, code, wparam, ctypes.c_void_p(lparam))
                if not self.recording:
                    return self._user32.CallNextHookEx(None, code, wparam, ctypes.c_void_p(lparam))
                if not self._wants_input():
                    self.ignored_focus = True
                    return self._user32.CallNextHookEx(None, code, wparam, ctypes.c_void_p(lparam))
                if vk in self._blocked:
                    return self._user32.CallNextHookEx(None, code, wparam, ctypes.c_void_p(lparam))
                name = _vk_to_name(vk)
                if self._block(vk, name):
                    return self._user32.CallNextHookEx(None, code, wparam, ctypes.c_void_p(lparam))
                t = self.timeline.stamp()
                if pressed:
                    if name in self.pressed:      # OS auto-repeat, not a new press
                        return self._user32.CallNextHookEx(None, code, wparam, ctypes.c_void_p(lparam))
                    self.key_events += 1
                    self._key_down_at[name] = t
                    self._learn(name)
                    self.timeline.add("key", name, True, t=t)
                else:
                    self.key_events += 1
                    self.timeline.add("key", name, False, t=t)
                    self._note_release(name, t)
        except Exception as exc:
            if len(self.errors) < 5:
                self.errors.append(f"keyboard hook: {exc}")
        return self._user32.CallNextHookEx(None, code, wparam, ctypes.c_void_p(lparam))

    def _ms_hook(self, code, wparam, lparam):
        try:
            if code == 0:
                if not self.recording:
                    return self._user32.CallNextHookEx(None, code, wparam, ctypes.c_void_p(lparam))
                if not self._wants_input():
                    self.ignored_focus = True
                    return self._user32.CallNextHookEx(None, code, wparam, ctypes.c_void_p(lparam))
                msg = int(wparam)
                if msg == WM_MOUSEMOVE:
                    # The hook hands us an MSLLHOOKSTRUCT whose first two fields
                    # are the cursor position in *screen* coordinates. The game
                    # turns by the raw relative delta, so the delta is computed
                    # here from consecutive positions. (Reading those fields as
                    # though they were dx/dy would record the absolute cursor
                    # position as the mouse movement.)
                    cx = ctypes.cast(lparam, ctypes.POINTER(ctypes.c_long))[0]
                    cy = ctypes.cast(lparam, ctypes.POINTER(ctypes.c_long))[1]
                    if self._mouse_pos is not None:
                        dx = cx - self._mouse_pos[0]
                        dy = cy - self._mouse_pos[1]
                        if dx or dy:
                            self.mouse_events += 1
                            self.timeline.add("mouse_move", (dx, dy), True,
                                              t=self.timeline.stamp())
                    self._mouse_pos = (cx, cy)
                    return self._user32.CallNextHookEx(None, code, wparam, ctypes.c_void_p(lparam))
                button = {WM_LBUTTONDOWN: "left", WM_LBUTTONUP: "left",
                          WM_RBUTTONDOWN: "right", WM_RBUTTONUP: "right",
                          WM_MBUTTONDOWN: "middle", WM_MBUTTONUP: "middle"}.get(msg)
                if button:
                    name = _MOUSE_BUTTON_NAMES[button]
                    pressed = msg in (WM_LBUTTONDOWN, WM_RBUTTONDOWN, WM_MBUTTONDOWN)
                    t = self.timeline.stamp()
                    self.mouse_events += 1
                    if pressed:
                        self._learn(name)
                        self._key_down_at[name] = t
                    self.timeline.add("mouse_btn", button, pressed, t=t)
                    if pressed:
                        self.pressed.add(name)
                    else:
                        self._key_down_at.pop(name, None)
                        self.pressed.discard(name)
                elif msg == WM_MOUSEWHEEL:
                    self.ignored_events += 1
        except Exception as exc:
            if len(self.errors) < 5:
                self.errors.append(f"mouse hook: {exc}")
        return self._user32.CallNextHookEx(None, code, wparam, ctypes.c_void_p(lparam))

    def _block(self, vk: int, name: str) -> bool:
        """
        Keys the bot must never press and calibration must never learn.

        Diagnostics keys live here: F1 and F3 are debug-overlay keys in a very
        large number of games (and in some of them F3 is a modifier for
        further debug toggles), so treating them as bot output would change
        the game's own settings behind your back. See RECORDER_BLOCKED_VKS.
        """
        if vk in RECORDER_BLOCKED_VKS:
            self._blocked.add(vk)
            return True
        return False

    def _learn(self, name: str):
        """
        Add a newly-seen key to the whitelist (calibration discovery).

        New keys default to *hold* rather than *tap*: if you pressed something
        while playing, holding it is the behaviour the bot needs to be able to
        reproduce, and a tap can still be expressed by holding it for one step.
        """
        if self.keymap is None or not self.allow_new_keys:
            return
        if name.startswith("MOUSE_"):
            if self.keymap.allow(name, source="discovered"):
                self.new_keys.append(name)
                if self.verbose:
                    print(f"[Discover] New mouse button allowed: {name}")
            return
        if self.keymap.allow(name, source="discovered", hold="hold"):
            self.new_keys.append(name)
            if self.verbose:
                print(f"[Discover] New key allowed: {name} (as a hold) - "
                      f"it will be saved to the keymap")

    # ---- convenience ----
    def arm(self):
        """Start recording immediately (no waiting for the first toggle)."""
        self.recording = True

    def wait_for_toggle(self, poll_hook: bool = True, sleep: float = 0.01) -> None:
        """Block until F8 is pressed, while keeping the hooks alive."""
        self.last_toggle = time.perf_counter()
        while not self.recording and self.running:
            if poll_hook:
                self.pump()
            time.sleep(sleep)

    def hz(self, window: float = 2.0) -> float:
        """Approximate event rate over the recent past, for status lines."""
        return (self.key_events + self.mouse_events) / max(1e-6, window)


_MOUSE_BUTTON_NAMES_REV = {v: k for k, v in _MOUSE_BUTTON_NAMES.items()}


class MiniStatusThread:
    """
    Prints a live one-line status while recording, without stealing the
    console from whatever the user is actually doing.
    """

    def __init__(self, text_fn, interval: float = 2.0):
        self.text_fn = text_fn
        self.interval = max(0.25, float(interval))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, name="StatusLine",
                                        daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                text = self.text_fn()
            except Exception:
                continue
            if text:
                print(text, flush=True)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class ScreenGrabber:
    """
    Frame grabber that reuses its device contexts and bitmap.

    ``capture_window_printwindow`` above allocates a device context and a bitmap
    on every call, which costs milliseconds - fine for a diagnostic, but it caps
    a training loop at a handful of frames per second. This class keeps one
    compatible DC/bitmap alive for the window's current client size and only
    re-creates them when the window is resized, so the capture budget is spent
    on PrintWindow itself.

    It also measures the real capture cost, because "how fast can this thing
    see" is the biggest single limit on how quickly the bot can react.
    """

    PW_RENDERFULLCONTENT = 0x00000002

    def __init__(self, hwnd: int):
        self.hwnd = hwnd
        self._size = (0, 0)
        self._hwnd_dc = None
        self._mfc_dc = None
        self._save_dc = None
        self._bitmap = None
        self.last_error: Optional[str] = None
        self.capture_seconds = 0.0
        self.calls = 0

    def _release(self):
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

    def close(self):
        self._release()

    def grab(self, render_full_content: bool = True) -> Optional[np.ndarray]:
        """Capture the client area. Returns BGR (H, W, 3) or None."""
        started = time.perf_counter()
        try:
            _left, _top, right, bottom = win32gui.GetClientRect(self.hwnd)
            width, height = int(right), int(bottom)
            if width <= 0 or height <= 0:
                self.last_error = "window has no client area"
                return None
            if (width, height) != self._size:
                self._release()
                self._hwnd_dc = win32gui.GetWindowDC(self.hwnd)
                self._mfc_dc = win32ui.CreateDCFromHandle(self._hwnd_dc)
                self._save_dc = self._mfc_dc.CreateCompatibleDC()
                self._bitmap = win32ui.CreateBitmap()
                self._bitmap.CreateCompatibleBitmap(self._mfc_dc, width, height)
                self._save_dc.SelectObject(self._bitmap)
                self._size = (width, height)

            flags = self.PW_RENDERFULLCONTENT if render_full_content else 0
            result = ctypes.windll.user32.PrintWindow(
                self.hwnd, self._save_dc.GetSafeHdc(), flags)
            if result != 1 and render_full_content:
                result = ctypes.windll.user32.PrintWindow(
                    self.hwnd, self._save_dc.GetSafeHdc(), 0)
            if result != 1:
                self.last_error = f"PrintWindow returned {result}"
                return None

            info = self._bitmap.GetInfo()
            bits = self._bitmap.GetBitmapBits(True)
            img = np.frombuffer(bits, dtype=np.uint8)
            img = img.reshape((info["bmHeight"], info["bmWidth"], 4))
            self.last_error = None
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as exc:
            self.last_error = str(exc)
            return None
        finally:
            elapsed = time.perf_counter() - started
            self.calls += 1
            self.capture_seconds += elapsed

    @property
    def mean_capture_ms(self) -> float:
        return 1000.0 * self.capture_seconds / max(1, self.calls)


class FrameClock:
    """
    Paces the control loop at a target rate and reports the rate it achieved.

    Reaction time is set by how often the bot can grab a frame and act on it,
    so this is the number to watch: if the achieved rate sits well under the
    target, the capture path is the bottleneck and the perf line says so.
    """

    def __init__(self, target_fps: float = TARGET_FPS, extra_sleep: float = 0.0,
                 report_every: float = 10.0):
        self.target_fps = max(0.0, float(target_fps))
        self.extra_sleep = max(0.0, float(extra_sleep))
        self.report_every = max(1.0, float(report_every))
        self._step = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self._next = time.perf_counter()
        self._count = 0
        self._last_report = self._next
        self.achieved_hz = 0.0
        self.worst_hz = float("inf")

    def tick(self):
        """Sleep until the next slot, then advance it."""
        if self._step > 0:
            self._next += self._step
            wait = self._next - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            elif wait < -2 * self._step:
                # We fell far behind; resynchronise rather than sprinting to
                # catch up, which would flood the game with input.
                self._next = time.perf_counter()
        if self.extra_sleep > 0:
            time.sleep(self.extra_sleep)
        self._count += 1

    def report(self, force: bool = False) -> Optional[str]:
        now = time.perf_counter()
        if not force and now - self._last_report < self.report_every:
            return None
        window = now - self._last_report
        hz = self._count / max(1e-6, window)
        self.achieved_hz = hz
        self.worst_hz = min(self.worst_hz, hz)
        self._count = 0
        self._last_report = now
        if self.target_fps <= 0:
            return f"[Perf] {hz:5.1f} steps/s"
        line = f"[Perf] {hz:5.1f} steps/s (target {self.target_fps:.0f})"
        if hz < self.target_fps * 0.5:
            line += ("  <- capture is the bottleneck; raise CAPTURE_DELAY or "
                     "lower IMG_SIZE / TARGET_FPS in SECTION 0")
        return line

    def reset(self):
        self._next = time.perf_counter()
        self._last_report = self._next
        self._count = 0


# ---------------------------------------------------------------------------
# Calibration: learn the whitelist of keys you actually play with
# ---------------------------------------------------------------------------
def _print_key_report(keymap: Keymap, allowed: Dict[str, dict]):
    """Human-readable table of what calibration found."""
    print()
    print("-" * 72)
    print("  KEYS THE BOT MAY NOW USE")
    print("-" * 72)
    holds = [(n, e) for n, e in allowed.items() if keymap.is_hold(n)]
    taps = [(n, e) for n, e in allowed.items() if not keymap.is_hold(n)]
    for title, rows in (("hold (can be kept down)", holds),
                        ("tap (pressed and released)", taps)):
        if not rows:
            continue
        print(f"  {title}:")
        for name, entry in sorted(rows, key=lambda kv: _key_priority(kv[0])):
            count = entry.get("count", 0)
            total = entry.get("held_seconds", 0.0)
            longest = entry.get("max_hold", 0.0)
            print(f"    {name:<10} vk=0x{int(entry.get('vk') or 0):02X}  "
                  f"presses={count:<4} held={total:6.2f}s  longest={longest:5.2f}s")
    if keymap.mouse_buttons:
        print("  mouse buttons:")
        for name, entry in sorted(keymap.mouse_buttons.items()):
            print(f"    {name:<10} presses={entry.get('count', 0)}")
    sens = keymap.mouse_sensitivity or {}
    if sens:
        print(f"  mouse look: {sens.get('pixels_per_step', 0.0):.1f} px per "
              f"control step on average "
              f"(max {sens.get('max_pixels_per_step', 0.0):.0f})")
        print(f"              -> MOUSE_TURN_PIXELS set to "
              f"{keymap.default_mouse_turn}")
    print("-" * 72)


def run_calibration(target: dict, args) -> int:
    """
    --calibrate: press F8, play for a bit using only the keys the bot may use,
    press F8 again, and the whitelist is written to a JSON keymap.

    Input injection is disabled for the whole run (see InjectionLock), so
    nothing the bot does can interfere while you are demonstrating.

    Everything you press while recording is logged with how long you held it.
    Presses shorter than CALIBRATE_MIN_HOLD are reported but left out, so a
    stray tap on an unrelated key does not end up in the bot's action space.

    The same run also measures your mouse-look: the raw deltas the game sees
    while you turn are averaged into the per-step turn size the bot will use.
    """
    with InjectionLock("calibration"):
        return _run_calibration_locked(target, args)


def _run_calibration_locked(target: dict, args) -> int:
    hwnd = target["hwnd"]
    path = os.path.abspath(args.keymap)
    _mods, vk = parse_hotkey(args.calibrate_key)

    existing = Keymap.load(path) if os.path.exists(path) else None
    if existing is not None and not args.fresh_keymap:
        keymap = existing
        print(f"[Calibrate] Merging into the existing keymap at '{path}'.")
        print(f"[Calibrate] It already allows: {keymap.describe_keys()}")
    else:
        keymap = Keymap.default()
        keymap.origin = "calibrated"
        keymap.game_hwnd = hwnd
        keymap.game = target.get("title") or ""
        try:
            rect = win32gui.GetClientRect(hwnd)
            keymap.game_resolution = [int(rect[2]), int(rect[3])]
        except Exception:
            keymap.game_resolution = [0, 0]
        if existing is not None:
            print("[Calibrate] CALIBRATE_FRESH_KEYMAP is on: starting from the "
                  "built-in defaults, ignoring the old file.")

    print()
    print("=" * 72)
    print("  CALIBRATION")
    print("=" * 72)
    print(f"  Target window : {target.get('title')!r} (hwnd={hwnd})")
    print(f"  Keymap file   : {path}")
    print(f"  Start/stop key: {args.calibrate_key.upper()}")
    print()
    print("  1. Click the game window so it has focus.")
    print("  2. Press the start/stop key to BEGIN recording.")
    print("  3. Play normally for a while. Use every key and mouse button you")
    print("     want the bot to be allowed to use, and hold the movement keys")
    print("     the way you really play - hold times are recorded. Turn the")
    print("     view with the mouse so the bot learns how far one turn step")
    print("     should be.")
    print("  4. Press the start/stop key again to STOP and save the JSON.")
    print()
    print("  Input is only recorded while the game window is focused.")
    print("  (F1/F3 are skipped - they toggle debug overlays in most games.)")
    print("=" * 72)
    print()

    recorder = InputRecorder(
        keymap=keymap,
        target_hwnd=hwnd,
        toggle_vk=vk,
        record_only_when_focused=True,
        allow_new_keys=True,
        verbose=True,
    )
    if not recorder.start():
        print("[Calibrate] ERROR: could not install the input hooks:")
        for err in recorder.errors:
            print(f"            {err}")
        print("[Calibrate] Low-level hooks can be blocked by security software.")
        return 1
    for err in recorder.errors:
        print(f"[Calibrate] Warning: {err}")

    status = MiniStatusThread(lambda: _calibration_status(recorder), interval=2.0)
    status.start()
    aborted = False

    try:
        print(f"[Calibrate] Waiting for {args.calibrate_key.upper()} "
              f"(up to {CALIBRATE_WAIT_SECONDS / 60:.0f} min)...")
        waited = 0.0
        while not recorder.recording and waited < CALIBRATE_WAIT_SECONDS:
            recorder.pump()
            time.sleep(0.01)
            waited += 0.01
        if not recorder.recording:
            print("[Calibrate] Timed out waiting for the start key; nothing saved.")
            aborted = True
        else:
            print("[Calibrate] RECORDING. Play now. "
                  f"Press {args.calibrate_key.upper()} again when you are done.")
            while recorder.recording:
                recorder.pump()
                time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n[Calibrate] Interrupted; saving whatever was recorded.")
    finally:
        status.stop()
        recorder.stop()

    if aborted:
        return 1

    events = recorder.timeline
    holds = events.key_holds()
    duration = max(events.duration(), 1e-6)
    min_hold = max(0.0, float(getattr(args, "calibrate_min_hold",
                                      CALIBRATE_MIN_HOLD)))

    # Which keys survived the minimum-hold filter.
    allowed: Dict[str, dict] = {}
    rejected: List[str] = []
    known = calibratable_keys()
    for name, stats in holds.items():
        longest = stats.get("max_hold", 0.0)
        upper = name.upper()
        if longest < min_hold:
            rejected.append(f"{name} ({longest * 1000:.0f} ms)")
            continue
        if name.startswith("MOUSE_"):
            keymap.allow(name, source="calibrated")
            keymap.mouse_buttons[upper]["count"] = stats.get("count", 0)
            allowed[name] = {"vk": keymap.button_vk(upper) or 0,
                             "count": stats.get("count", 0),
                             "held_seconds": stats.get("held_seconds", 0.0),
                             "max_hold": longest}
            continue
        keymap.allow(name, source="calibrated", vk=known.get(upper))
        entry = keymap.keys.get(upper)
        if entry is None:
            continue
        entry.update(source="calibrated", count=stats.get("count", 0),
                     held_seconds=stats.get("held_seconds", 0.0),
                     max_hold=longest)
        # Anything you actually press during calibration is something the bot
        # may need to hold down, so allow it as a hold. A tap is just a hold
        # that lasts one step.
        entry["hold"] = "hold"
        allowed[name] = dict(entry)

    # Mouse-look sensitivity. The hooks deliver one event per raw mouse
    # message (often 100+ per second) while the bot turns once per control
    # step, so scale the observed per-event delta up to one step at the
    # capture rate - roughly what the game sees in that time.
    deltas = [e[2] for e in events.snapshot() if e[1] == "mouse_move"]
    if deltas:
        magnitudes = [abs(float(d[0])) + abs(float(d[1])) for d in deltas]
        per_event = float(np.mean(magnitudes))
        duration = max(1e-6, events.duration())
        events_per_second = len(deltas) / duration
        # If the user barely moved the mouse, do not shrink the turn to nothing.
        per_step = per_event * max(1.0, events_per_second / max(1.0, TARGET_FPS))
        per_step = float(np.clip(per_step, 10.0, 400.0))
        keymap.default_mouse_turn = int(round(per_step))
        keymap.note_mouse_turn(per_step)
        print(f"[Calibrate] Mouse: {len(deltas)} move events over "
              f"{duration:.1f}s = {events_per_second:.0f}/s, "
              f"{per_event:.0f} px each -> {keymap.default_mouse_turn} px per "
              f"control step.")
    keymap.samples += max(1, int(duration * max(1.0, TARGET_FPS)))
    keymap.updated = datetime.now().isoformat(timespec="seconds")
    keymap.origin = "calibrated"

    actions, capped = build_action_set(keymap)
    keymap.set_actions(actions, capped)

    print()
    print(f"[Calibrate] Recorded {duration:.1f}s of play, "
          f"{recorder.key_events} key events, {recorder.mouse_events} mouse events.")
    if recorder.ignored_focus:
        print("[Calibrate] Note: some input was skipped while the game window "
              "was not focused.")
    if rejected:
        print(f"[Calibrate] Ignored {len(rejected)} too-short press(es): "
              f"{', '.join(rejected)}")
        print(f"[Calibrate] (threshold {min_hold * 1000:.0f} ms; set "
              f"CALIBRATE_MIN_HOLD = 0 to keep everything)")
    if not allowed:
        print("[Calibrate] No usable keys were recorded; keeping the defaults.")
    _print_key_report(keymap, allowed)
    print(f"[Keymap] {len(actions)} actions available"
          + (" (capped at MAX_ACTIONS; the least useful combinations were "
             "dropped)" if capped else ""))
    print(describe_actions(actions[:24])
          + (f"\n  ... and {len(actions) - 24} more" if len(actions) > 24 else ""))

    try:
        saved = keymap.save(path)
    except Exception as exc:
        print(f"[Calibrate] ERROR: could not write '{path}': {exc}")
        return 1
    print()
    print(f"[Calibrate] Saved keymap -> {saved}")
    print("[Calibrate] The bot now only ever presses those keys. Re-run "
          "--calibrate any time you rebind something.")
    print("[Calibrate] Next: python bot1.py   (train, and it will watch you "
          "whenever you take the controls)")
    return 0


def _calibration_status(recorder: InputRecorder) -> Optional[str]:
    if not recorder.recording:
        return None
    return (f"[Calibrate] recording... {recorder.key_events} key events, "
            f"{recorder.mouse_events} mouse events")


# =============================================================================
# SECTION 1.7: HUMAN TAKEOVER - YIELD THE CONTROLS, WATCH, AND LEARN FROM YOU
#
# The bot has to be able to tell *your* input from its own, and it can: every
# event SendInput generates arrives at the Win32 low-level hooks with
# LLKHF_INJECTED / LLMHF_INJECTED set, while hardware input does not. That one
# fact is what makes this feature safe - the bot can watch the raw input stream
# continuously without ever mistaking its own keystrokes or mouse movement for
# yours.
#
# What happens when you touch the controls:
#   1. the bot lets go of every button it is holding and stops injecting,
#   2. it records the frames and exactly what you pressed and how you turned,
#   3. HUMAN_RELEASE_GRACE_SECONDS (3 s) after your last input it takes the
#      controls back and imitates what it saw in its next PPO updates.
#
# This is the old separate "--watch mode" folded into the training loop: there
# is no recording session to start, no countdown and no handoff prompt.
# =============================================================================

# KBDLLHOOKSTRUCT.flags bit meaning "this event was injected". (The mouse
# equivalent, LLMHF_INJECTED, is defined with the other hook constants above.)
LLKHF_INJECTED = 0x10


def build_action_set_for_args(args) -> Tuple[Optional[Keymap], List[dict]]:
    """
    Keymap -> action-set pipeline, used by both training and the diagnostics,
    so they always agree on what the bot is allowed to do.

    The set is rebuilt from the keymap on every launch rather than replayed
    from the list that was saved last time. The generator improves (new button
    combinations, better names), and a stored list can hold button names an
    older build spelled differently; the combinations that came out of your own
    play are folded back in from keymap.actions by build_action_set() itself,
    so nothing you demonstrated is lost. The rebuilt list is then repaired and
    de-duplicated, and written back if that changed anything.
    """
    path = os.path.abspath(args.keymap)
    keymap = Keymap.load(path, required=False)
    if keymap is None:
        keymap = Keymap.default()
        if os.path.exists(path):
            print(f"[Keymap] '{path}' unusable; falling back to the built-in "
                  f"default keys. Run --calibrate to create a real one.")
        else:
            print(f"[Keymap] No keymap at '{path}'; using the built-in default "
                  f"keys. Run --calibrate to choose your own.")

    original = list(keymap.actions)
    if original:
        # Repair what was saved *before* rebuilding, so the log can say which
        # stored button names were translated, and so the rebuild folds in
        # combinations this keymap can actually press.
        repaired_stored, stored_report = normalize_action_set(keymap, original)
        keymap.set_actions(repaired_stored, keymap.actions_capped)
    else:
        stored_report = ActionRepair()

    actions, capped = build_action_set(keymap)
    actions, report = normalize_action_set(keymap, actions)
    keymap.set_actions(actions, capped)

    for outcome in (stored_report, report):
        if outcome.changed:
            print(f"[Actions] {outcome.summary()}")
    if original and len(actions) != len(original):
        print(f"[Actions] Action set rebuilt from the keymap: "
              f"{len(original)} -> {len(actions)} stored action(s)"
              + (" (capped at MAX_ACTIONS)" if capped else ""))

    changed = (stored_report.changed or report.changed
               or action_list_signature(actions) != action_list_signature(original))
    # Persist the rebuilt list, but only when there was already a keymap file: a
    # run that fell back to the built-in defaults should not create one.
    if keymap.path and changed:
        try:
            keymap.save()
            print(f"[Actions] Saved the updated keymap -> {keymap.path}")
        except Exception as exc:
            print(f"[Actions] Could not save the updated keymap: {exc}")
    return keymap, actions


def describe_action(action: dict, keymap: Optional[Keymap] = None) -> str:
    """Human-readable one-liner for a calibrated action."""
    parts = []
    held = action.get("held") or []
    uses = action.get("uses") or []
    if held:
        parts.append("hold " + "+".join(
            _hold_label(keymap, h) if keymap else str(h) for h in held))
    if uses:
        parts.append("hold " + "+".join(
            _hold_label(keymap, u) if keymap else str(u) for u in uses))
    if action.get("tap"):
        parts.append(f"tap {action['tap']}")
    if action.get("click"):
        parts.append(f"click {action['click']}")
    if action.get("look"):
        parts.append(f"turn {action['look'][0]:+d},{action['look'][1]:+d}")
    return ", ".join(parts) if parts else "do nothing"


def _extract_look(dx: float, dy: float, turn: int) -> Optional[List[int]]:
    """Map a recorded mouse delta onto one of the four look actions."""
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return None
    if abs(dx) >= abs(dy):
        return [-turn, 0] if dx < 0 else [turn, 0]
    return [0, -turn] if dy < 0 else [0, turn]


def _uses_tokens(keymap: Keymap, uses) -> List[str]:
    """
    Normalise an action's ``uses`` entry to hold tokens.

    Recorded button combinations are stored under readable names ("W", "LCTRL")
    while the action spec compares HOLD_57 / HOLD_A2 tokens, so everything is
    converted to tokens before matching or comparing.
    """
    tokens = []
    for item in (uses or []):
        upper = str(item).upper()
        if upper.startswith("HOLD_") or upper == TRACKED_HOLD.upper():
            tokens.append(upper)
            continue
        vk = keymap.vk(upper) or keymap.button_vk(upper)
        tokens.append(keymap.hold_name(vk) if vk else upper)
    return sorted(tokens)


class _ButtonLookup:
    """
    Maps a recorded (held buttons, turn) pair onto the closest action.

    Exact matches win. When the action set was capped and the exact
    combination is missing, the closest subset is used (dropping one button,
    then two, ...), which keeps the turn and the important movement key instead
    of throwing the whole demonstration away as "do nothing".
    """

    def __init__(self, keymap: Keymap, actions: List[dict]):
        self.known_tokens: set = set()
        self.exact: Dict[tuple, int] = {}
        self.by_size: Dict[int, Dict[tuple, int]] = {}
        self.noop_id = 0
        for action in actions:
            empty = not (action.get("held") or action.get("uses")
                         or action.get("look") or action.get("tap")
                         or action.get("click"))
            if empty:
                self.noop_id = action["id"]
                continue
            uses = tuple(_uses_tokens(keymap,
                                      (action.get("uses") or [])
                                      + (action.get("held") or [])))
            if not uses:
                continue
            self.known_tokens.update(uses)
            look = tuple(action["look"]) if action.get("look") else ()
            # setdefault (first registration wins), not assignment: a
            # "hold W and turn" action also answers for the no-turn key so that
            # a demonstration of a plain W hold is still matched when no
            # turn-only action exists - but it must never *replace* the plain
            # hold action that was registered before it, or every W press would
            # be imitated as W plus a turn.
            for look_variant in ({look, ()} if look else {()}):
                self.exact.setdefault((uses, look_variant), action["id"])
                self.by_size.setdefault(len(uses), {}).setdefault(
                    (uses, look_variant), action["id"])

        # Every subset of every known button set, ordered by size, so a match
        # can fall back to "hold most of what the human held".
        self.subsets: List[Dict[tuple, int]] = [{}]
        if self.by_size:
            max_size = max(self.by_size)
            for size in range(1, max_size + 1):
                table: Dict[tuple, int] = {}
                for bigger in range(size, max_size + 1):
                    for (uses, look), action_id in self.by_size.get(bigger, {}).items():
                        for combo in itertools.combinations(uses, size):
                            key = (tuple(combo), look)
                            table.setdefault(key, action_id)
                            table.setdefault((tuple(combo), ()), action_id)
                self.subsets.append(table)

    def match(self, uses: tuple, look: Optional[List[int]]) -> Tuple[int, bool]:
        """Return (action_id, exact)."""
        look_key = tuple(look) if look else ()
        for key in ((uses, look_key), (uses, ())):
            found = self.exact.get(key)
            if found is not None:
                return found, True
        if not uses:
            return self.noop_id, False
        # No exact action; take the largest subset that does exist.
        for size in range(min(len(uses), len(self.subsets) - 1), 0, -1):
            table = self.subsets[size]
            if size == len(uses):
                for key in ((uses, look_key), (uses, ())):
                    found = table.get(key)
                    if found is not None:
                        return found, False
                continue
            for combo in itertools.combinations(uses, size):
                for key in ((tuple(combo), look_key), (tuple(combo), ())):
                    found = table.get(key)
                    if found is not None:
                        return found, False
        return self.noop_id, False


class HumanSegment:
    """Frames, action labels and statistics from one stretch of your play."""

    def __init__(self):
        self.frames: List[np.ndarray] = []
        self.action_ids: List[int] = []
        self.steps = 0
        self.distinct = 0
        self.skipped = 0            # frames whose buttons we cannot express
        self.inexact = 0            # frames rounded to a subset of what you did
        self.held_presses = 0
        self.turn_steps = 0
        self.labels: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.action_ids)

    def top_labels(self, count: int = 4) -> str:
        ranked = sorted(self.labels.items(), key=lambda kv: -kv[1])[:count]
        return ", ".join(f"{name} {100.0 * hits / max(1, self.steps):.0f}%"
                         for name, hits in ranked)


def build_human_replay(frames: List[np.ndarray],
                       timestamps: List[float],
                       timeline: RecordedInput,
                       keymap: Keymap,
                       actions: List[dict],
                       min_steps: int = HUMAN_SEGMENT_MIN_STEPS,
                       ) -> Optional["HumanSegment"]:
    """
    Label one stretch of your play: for every recorded frame, which of the
    bot's actions reproduces the buttons you were holding and how far you
    turned.

    Frames whose buttons this run's action set cannot express at all are
    dropped rather than rounded to the nearest wrong action; the counts come
    back in the segment so the caller can report them.
    """
    min_steps = max(2, int(min_steps))
    if not frames or len(frames) < min_steps:
        return None
    times = list(timestamps[:len(frames)])
    frames = frames[:len(times)] if len(times) != len(frames) else frames
    if not times:
        return None

    matcher = _ButtonLookup(keymap, actions)
    known = matcher.known_tokens

    def expressible(name: str) -> bool:
        tokens = _uses_tokens(keymap, [name])
        return bool(tokens) and tokens[0] in known

    button_sets = timeline.active_at(times)
    deltas = timeline.move_deltas(times)
    turn = max(4, int(keymap.default_mouse_turn))

    segment = HumanSegment()
    for i, frame in enumerate(frames):
        held = button_sets[i] if i < len(button_sets) else set()
        if any(not expressible(name) for name in held):
            segment.skipped += 1
            continue
        dx, dy = deltas[i] if i < len(deltas) else (0.0, 0.0)
        look = _extract_look(dx, dy, turn)
        uses = tuple(_uses_tokens(keymap, held))
        action_id, exact = matcher.match(uses, look)
        if held and not exact:
            segment.inexact += 1
        segment.frames.append(frame)
        segment.action_ids.append(int(action_id))
        label = (actions[action_id].get("label", "?")
                 if 0 <= action_id < len(actions) else "?")
        segment.labels[label] = segment.labels.get(label, 0) + 1
    if not segment.action_ids:
        return None
    segment.steps = len(segment.action_ids)
    segment.distinct = len(segment.labels)
    segment.held_presses = len(button_sets)
    segment.turn_steps = sum(1 for d in deltas if abs(d[0]) > 0.5 or abs(d[1]) > 0.5)
    return segment


class HumanReplay:
    """
    Rolling buffer of your play, used to nudge the policy towards what you did.

    Frames are kept at the model's frame size in RGB. `max_frames` bounds
    memory: 1500 frames of 160x160 RGB is about 115 MB. Nothing here is
    checkpointed - this is a short-term "do what I just showed you" signal, not
    part of the policy.
    """

    def __init__(self, frame_size: int, seq_len: int, num_actions: int,
                 max_frames: int = HUMAN_REPLAY_MAX_FRAMES):
        self.frame_size = int(frame_size)
        self.seq_len = max(2, int(seq_len))
        self.num_actions = max(1, int(num_actions))
        self.max_frames = max(self.seq_len + 1, int(max_frames))
        self.frames: List[np.ndarray] = []
        self.action_ids: List[int] = []
        self.segments = 0
        self.steps_added = 0
        self.steps_skipped = 0
        self.updates = 0
        self.last_loss: Optional[float] = None
        self.last_match: Optional[float] = None

    def __len__(self) -> int:
        return len(self.frames)

    def add(self, segment: HumanSegment) -> int:
        """Fold one labelled stretch of play into the buffer."""
        for frame, action_id in zip(segment.frames, segment.action_ids):
            if (frame.shape[0] != self.frame_size
                    or frame.shape[1] != self.frame_size):
                frame = cv2.resize(frame, (self.frame_size, self.frame_size),
                                   interpolation=cv2.INTER_AREA)
            self.frames.append(frame)
            self.action_ids.append(int(action_id))
        over = len(self.frames) - self.max_frames
        if over > 0:
            del self.frames[:over]
            del self.action_ids[:over]
        self.segments += 1
        self.steps_added += len(segment)
        self.steps_skipped += segment.skipped
        return len(segment)

    def ready(self) -> bool:
        return len(self.frames) >= self.seq_len + 1

    def _class_weights(self, device) -> torch.Tensor:
        """Class balance for the imitation loss: no-op must not win by default."""
        counts = np.bincount(np.asarray(self.action_ids, dtype=np.int64),
                             minlength=self.num_actions).astype(np.float64)
        weight = np.ones(self.num_actions, dtype=np.float32)
        present = counts > 0
        if present.any():
            weight[present] = (counts[present].sum() / counts[present]) ** 0.5
            weight[present] /= weight[present].mean()
        return torch.tensor(weight, dtype=torch.float32, device=device)

    def sample(self, batch_size: int, device):
        """A minibatch of (sequence, action) pairs ending on random frames."""
        n = len(self.frames)
        last = np.random.randint(self.seq_len - 1, n, size=int(batch_size))
        window = np.stack([
            np.stack(self.frames[i - self.seq_len + 1:i + 1], axis=0)
            for i in last
        ], axis=0)                        # (B, T, H, W, 3) RGB uint8
        visual = torch.from_numpy(
            np.ascontiguousarray(window.transpose(0, 1, 4, 2, 3))
        ).float().to(device)              # (B, T, 3, H, W)
        actions = torch.tensor([self.action_ids[i] for i in last],
                               dtype=torch.long, device=device)
        return visual, actions, self._class_weights(device)

    def imitation_step(self, model, optimizer, device,
                       weight: float = HUMAN_IMITATION_WEIGHT,
                       batch_size: int = HUMAN_IMITATION_BATCH_SIZE,
                       audio_samples: int = 16000) -> bool:
        """
        One gradient step towards "do what the human did".

        Deliberately not folded into the PPO epochs: this is a supervised nudge
        on a small minibatch, scaled by HUMAN_IMITATION_WEIGHT, that keeps the
        policy pointed at your behaviour without taking the update over.
        """
        if not self.ready():
            return False
        visual, actions, class_weights = self.sample(batch_size, device)
        audio = torch.zeros((visual.shape[0], visual.shape[1], audio_samples),
                            device=device)
        logits, _value = model(visual, audio)
        loss = F.cross_entropy(logits, actions, weight=class_weights)
        optimizer.zero_grad()
        (float(weight) * loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        with torch.no_grad():
            match = float((logits.argmax(dim=-1) == actions).float().mean().item())
        self.updates += 1
        self.last_loss = float(loss.item())
        self.last_match = match
        return True

    def describe(self) -> str:
        return (f"{len(self.frames)} frame(s) from {self.segments} takeover(s), "
                f"{self.steps_skipped} skipped as inexpressible")


class HumanWatcher:
    """
    Watches the real keyboard and mouse for *you*, on its own thread.

    The Win32 low-level hooks must be installed and pumped by the same thread,
    and that thread has to stay responsive or Windows silently drops them - so
    they live on a dedicated thread running a message pump, not on the training
    loop, which can spend seconds inside a PPO update.

    Every event carries an "injected" flag; SendInput sets it and your hardware
    does not, so the bot's own traffic is filtered out here and a single
    unflagged key press, click or mouse movement is what hands you the
    controls.
    """

    def __init__(self, target_hwnd: int = 0, keymap: Optional[Keymap] = None,
                 ignored_vks=(), max_frames: int = HUMAN_REPLAY_MAX_FRAMES,
                 verbose: bool = True):
        self.target_hwnd = int(target_hwnd or 0)
        self.keymap = keymap
        self.verbose = bool(verbose)
        self.ignored_vks = {int(v) for v in ignored_vks if v}
        self.max_frames = max(2, int(max_frames))
        self.events = 0
        self.new_keys: List[str] = []
        self.errors: List[str] = []

        self._lock = threading.RLock()
        self._last_activity = 0.0
        self._human_pressed: Dict[str, float] = {}
        self._recent: deque = deque(maxlen=512)
        self._segment: Optional[RecordedInput] = None
        self._frames: List[np.ndarray] = []
        self._frame_times: List[float] = []
        self._cap_hit = False
        self._mouse_pos: Optional[Tuple[int, int]] = None

        self._running = False
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._msg_type = None
        self._kb_proc = None
        self._ms_proc = None
        self._kb_hook_handle = None
        self._ms_hook_handle = None
        self._user32 = ctypes.windll.user32
        self._watch_vks = self._build_watch_vks()

    # ---- lookup ----
    def _build_watch_vks(self) -> List[int]:
        """
        Every key whose live state is worth polling.

        This is the fallback for the case where the OS quietly removes a hook
        (it does that when the owning thread is slow): if a watched key is
        physically down and the bot is not the one holding it, it is you.
        """
        names = set(DEFAULT_HOLD_KEYS) | set(DEFAULT_TAP_KEYS)
        if self.keymap is not None:
            names |= set(self.keymap.keys)
            names |= set(self.keymap.mouse_buttons)
        vks = set()
        for name in names:
            vk = None
            if self.keymap is not None:
                vk = self.keymap.vk(name) or self.keymap.button_vk(name)
            vk = vk or calibratable_keys().get(str(name).upper())
            if vk:
                vks.add(int(vk))
        for vk in (0x01, 0x02, 0x04):              # mouse buttons
            vks.add(vk)
        vks -= self.ignored_vks
        vks.discard(0)
        return sorted(vks)

    def _vk_of(self, name: str) -> Optional[int]:
        if name.startswith("MOUSE_"):
            return {"MOUSE_LEFT": 0x01, "MOUSE_RIGHT": 0x02,
                    "MOUSE_MIDDLE": 0x04}.get(name)
        if self.keymap is not None:
            vk = self.keymap.vk(name)
            if vk:
                return int(vk)
        return calibratable_keys().get(str(name).upper())

    def _focused(self) -> bool:
        if not self.target_hwnd:
            return True
        try:
            return self._user32.GetForegroundWindow() == self.target_hwnd
        except Exception:
            return True

    # ---- lifecycle ----
    def start(self) -> bool:
        """Install the hooks on their own thread. True if the keyboard hook is up."""
        if self._thread is not None:
            return self._kb_hook_handle is not None
        self._running = True
        self._ready.clear()
        self._thread = threading.Thread(target=self._thread_main,
                                        name="HumanWatcher", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3.0)
        return self._kb_hook_handle is not None

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _thread_main(self):
        try:
            self._install_hooks()
        except Exception as exc:
            self.errors.append(f"hooks: {exc}")
        finally:
            self._ready.set()
        try:
            while self._running:
                self._pump()
                time.sleep(0.002)
        finally:
            self._remove_hooks()

    def _install_hooks(self):
        def keyboard_proc(code, wparam, lparam):
            try:
                return self._kb_hook(code, wparam, lparam)
            except Exception:
                return 0

        def mouse_proc(code, wparam, lparam):
            try:
                return self._ms_hook(code, wparam, lparam)
            except Exception:
                return 0

        self._kb_proc = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t,
            ctypes.c_ssize_t)(keyboard_proc)
        self._ms_proc = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t,
            ctypes.c_ssize_t)(mouse_proc)

        for label, hook_id, proc, attr in (
            ("keyboard", WH_KEYBOARD_LL, self._kb_proc, "_kb_hook_handle"),
            ("mouse", WH_MOUSE_LL, self._ms_proc, "_ms_hook_handle"),
        ):
            try:
                handle = self._user32.SetWindowsHookExW(hook_id, proc, None, 0)
            except Exception as exc:
                handle = None
                self.errors.append(f"{label} hook raised {exc}")
            if not handle:
                self.errors.append(f"{label} hook could not be installed")
            else:
                setattr(self, attr, handle)

    def _remove_hooks(self):
        for attr in ("_kb_hook_handle", "_ms_hook_handle"):
            handle = getattr(self, attr)
            if handle:
                try:
                    self._user32.UnhookWindowsHookEx(handle)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._kb_proc = None
        self._ms_proc = None

    def _pump(self):
        """Drain messages so Windows keeps calling the hook procedures."""
        msg_type = self._msg_type
        if msg_type is None:
            class _MSG(ctypes.Structure):
                _fields_ = [("hwnd", ctypes.c_void_p), ("message", ctypes.c_uint),
                            ("wParam", ctypes.c_void_p), ("lParam", ctypes.c_void_p),
                            ("time", ctypes.c_uint), ("pt_x", ctypes.c_long),
                            ("pt_y", ctypes.c_long)]
            msg_type = _MSG
            self._msg_type = msg_type

        msg = msg_type()
        while self._user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
            self._user32.TranslateMessage(ctypes.byref(msg))
            self._user32.DispatchMessageW(ctypes.byref(msg))

    # ---- hooks (these run on the watcher thread) ----
    def _kb_hook(self, code, wparam, lparam):
        if code == 0:
            data = ctypes.cast(lparam, ctypes.POINTER(ctypes.c_uint32))
            vk = int(data[0]) & 0xFF
            flags = int(data[2])          # vkCode, scanCode, flags, ...
            if not (flags & LLKHF_INJECTED):
                self._on_key(vk, wparam in (WM_KEYDOWN, WM_SYSKEYDOWN))
        return self._user32.CallNextHookEx(None, code, wparam,
                                           ctypes.c_void_p(lparam))

    def _ms_hook(self, code, wparam, lparam):
        if code == 0:
            data = ctypes.cast(lparam, ctypes.POINTER(ctypes.c_long))
            injected = bool(int(data[3]) & LLMHF_INJECTED)   # pt, mouseData, flags
            if injected:
                if int(wparam) == WM_MOUSEMOVE:
                    # Keep the position baseline current even for our own
                    # injected moves, so the next real delta is measured from
                    # where the cursor actually is.
                    self._note_position((int(data[0]), int(data[1])))
                return self._user32.CallNextHookEx(None, code, wparam,
                                                   ctypes.c_void_p(lparam))
            msg = int(wparam)
            if msg == WM_MOUSEMOVE:
                self._on_move((int(data[0]), int(data[1])))
            else:
                button = {WM_LBUTTONDOWN: ("left", True),
                          WM_LBUTTONUP: ("left", False),
                          WM_RBUTTONDOWN: ("right", True),
                          WM_RBUTTONUP: ("right", False),
                          WM_MBUTTONDOWN: ("middle", True),
                          WM_MBUTTONUP: ("middle", False)}.get(msg)
                if button:
                    self._on_button(button[0], button[1])
        return self._user32.CallNextHookEx(None, code, wparam,
                                           ctypes.c_void_p(lparam))

    # ---- event handling ----
    def _note_position(self, point):
        with self._lock:
            self._mouse_pos = point

    def _on_key(self, vk: int, pressed: bool):
        if vk in self.ignored_vks or vk in RECORDER_BLOCKED_VKS:
            return
        name = _vk_to_name(vk)
        now = time.perf_counter()
        learn = False
        with self._lock:
            self._last_activity = now
            if pressed:
                repeat = name in self._human_pressed
                self._human_pressed[name] = now
                if not repeat:
                    self._record("key", name, True, now)
                    # Only grow the keymap from keys pressed *in the game*;
                    # what you type into another window is not the bot's
                    # business.
                    learn = self._focused()
            else:
                self._human_pressed.pop(name, None)
                self._record("key", name, False, now)
        if learn:
            self._learn(name)

    def _on_button(self, button: str, pressed: bool):
        name = _MOUSE_BUTTON_NAMES.get(button)
        if not name:
            return
        now = time.perf_counter()
        learn = False
        with self._lock:
            self._last_activity = now
            if pressed:
                repeat = name in self._human_pressed
                self._human_pressed[name] = now
                if not repeat:
                    self._record("mouse_btn", button, True, now)
                    learn = self._focused()
            else:
                self._human_pressed.pop(name, None)
                self._record("mouse_btn", button, False, now)
        if learn:
            self._learn(name)

    def _on_move(self, point):
        with self._lock:
            previous = self._mouse_pos
            self._mouse_pos = point
        if previous is None or not HUMAN_MOUSE_TAKEOVER:
            return
        dx = point[0] - previous[0]
        dy = point[1] - previous[1]
        if not dx and not dy:
            return
        now = time.perf_counter()
        with self._lock:
            self._last_activity = now
            self._record("mouse_move", (dx, dy), True, now)

    def _record(self, kind: str, value, pressed: bool, t: float):
        """Append one of your events (caller holds the lock)."""
        event = (float(t), kind, value, bool(pressed))
        self._recent.append(event)
        if self._segment is not None:
            self._segment.events.append(event)
        self.events += 1

    def _learn(self, name: str):
        """
        Remember a button the calibrated keymap does not know about yet.

        It goes into keymap.json now, and the action set is rebuilt from the
        keymap at the start of the next launch, so it becomes something the bot
        can actually press - at the cost of a fresh policy, because the action
        space (and so the policy head) has changed size.
        """
        if self.keymap is None:
            return
        if name.startswith("MOUSE_"):
            if self.keymap.allow(name, source="discovered"):
                self.new_keys.append(name)
                if self.verbose:
                    print(f"[Human] New mouse button: {name} - it is in the "
                          f"keymap now and joins the action set at the next "
                          f"launch.")
            return
        if self.keymap.allow(name, source="discovered", hold="hold"):
            self.new_keys.append(name)
            if self.verbose:
                print(f"[Human] New key: {name} - it is in the keymap now and "
                      f"joins the action set at the next launch.")

    # ---- state the training loop reads ----
    def human_active(self, bot_held_vks=(),
                     grace: float = HUMAN_RELEASE_GRACE_SECONDS) -> bool:
        """
        True while the controls are yours: any real input within the last
        `grace` seconds, or any watched key still physically held that the bot
        is not itself holding.
        """
        now = time.perf_counter()
        with self._lock:
            last = self._last_activity
            held = list(self._human_pressed)
        if now - last < max(0.0, float(grace)):
            return True

        bot_held = {int(vk) for vk in bot_held_vks}
        # A key you hold down sends no further events, so ask the OS directly:
        # first about the keys we watched you press...
        for name in held:
            vk = self._vk_of(name)
            if vk and vk not in bot_held and (win32api.GetAsyncKeyState(vk) & 0x8000):
                with self._lock:
                    self._last_activity = now
                return True
        # ...then about every key the bot could press. That sweep is also what
        # catches a hook the OS quietly dropped, so it runs even when we think
        # we know what is held - a stale held set must not mask a dead hook.
        for vk in self._watch_vks:
            if vk in bot_held:
                continue
            if win32api.GetAsyncKeyState(vk) & 0x8000:
                with self._lock:
                    self._last_activity = now
                return True
        return False

    def pressed_now(self) -> List[str]:
        with self._lock:
            return sorted(self._human_pressed)

    # ---- recording one takeover ----
    def begin_segment(self) -> RecordedInput:
        """
        Start recording. A second of recent events is carried over so the
        frames captured just after the first key press still line up with it.
        """
        now = time.perf_counter()
        with self._lock:
            segment = RecordedInput()
            segment.events = [event for event in self._recent
                              if now - event[0] <= 1.0]
            self._segment = segment
            self._frames = []
            self._frame_times = []
            self._cap_hit = False
            self._mouse_pos = None
        return segment

    def capture(self, frame: Optional[np.ndarray]) -> bool:
        """Store one frame of your play (only while the game window is focused)."""
        if frame is None or not self._focused():
            return False
        with self._lock:
            if self._segment is None:
                return False
            if len(self._frames) >= self.max_frames:
                self._cap_hit = True
                return False
            self._frames.append(np.ascontiguousarray(frame))
            self._frame_times.append(time.perf_counter())
        return True

    def end_segment(self) -> dict:
        """Stop recording and hand back the frames plus their input timeline."""
        with self._lock:
            segment = self._segment
            out = {"frames": self._frames,
                   "timestamps": self._frame_times,
                   "timeline": segment or RecordedInput(),
                   "cap_hit": self._cap_hit,
                   "focused": self._focused()}
            self._segment = None
            self._frames = []
            self._frame_times = []
            self._cap_hit = False
            self._mouse_pos = None
        return out


class HumanTakeover:
    """
    Runs the handover in both directions, one call per training iteration.

    service() returns "human" while the controls are yours - the caller must
    not collect a PPO transition then - "resumed" on the iteration where the
    bot takes them back, and "idle" the rest of the time.
    """

    NOTICE_EVERY = 10.0

    def __init__(self, env, watcher: HumanWatcher):
        self.env = env
        self.watcher = watcher
        self.replay = HumanReplay(env.frame_size, env.seq_len,
                                  env.action_space.n)
        self.active = False
        self.takeovers = 0
        self.steps_watched = 0
        self._next_notice = 0.0

    # ---- the one call the training loop makes ----
    def service(self) -> str:
        if self.watcher.human_active(self.env.held_vks):
            if not self.active:
                self._begin()
            frame = self.env.observe_human_step()
            if self.watcher.capture(frame):
                self.steps_watched += 1
            now = time.perf_counter()
            if now >= self._next_notice:
                self._next_notice = now + self.NOTICE_EVERY
                held = self.watcher.pressed_now()
                print(f"[Human] watching you play: {self.steps_watched} frame(s) "
                      f"this takeover"
                      + (f", holding {', '.join(held)}" if held else ""))
            return "human"
        if self.active:
            self._end()
            return "resumed"
        return "idle"

    # ---- transitions ----
    def _begin(self):
        self.active = True
        self.takeovers += 1
        self.steps_watched = 0
        self._next_notice = time.perf_counter() + self.NOTICE_EVERY
        # Let go of everything *before* the injection switch closes, so no key
        # is left pressed in the game while you are playing.
        self.env.suspend_bot()
        self.watcher.begin_segment()
        print()
        print("[Human] You have the controls. The bot has let go of every button "
              "it was holding and is sending nothing at all.")
        print(f"[Human] It is watching what you press, and takes the controls "
              f"back {HUMAN_RELEASE_GRACE_SECONDS:.0f}s after your last input.")

    def _end(self):
        self.active = False
        self.env.resume_bot()
        recorded = self.watcher.end_segment()
        frames = recorded["frames"]
        if not frames:
            print("[Human] Hands off - taking the controls back. Nothing was "
                  "recorded: the game window was not focused.")
            return

        segment = build_human_replay(frames, recorded["timestamps"],
                                     recorded["timeline"], self.env.keymap,
                                     self.env.actions)
        if segment is None:
            print(f"[Human] Hands off - taking the controls back. Recorded "
                  f"{len(frames)} frame(s), too few to learn anything from.")
            return

        self.replay.add(segment)
        notes = []
        if segment.inexact:
            notes.append(f"{segment.inexact} rounded to the closest action")
        if segment.skipped:
            notes.append(f"{segment.skipped} skipped (buttons this run's action "
                         f"set cannot express)")
        if recorded.get("cap_hit"):
            notes.append(f"stopped at the {self.watcher.max_frames}-frame buffer "
                         f"limit")
        print(f"[Human] Hands off - taking the controls back. Recorded "
              f"{segment.steps} step(s) of your play, "
              f"{segment.distinct} distinct action(s), "
              f"{segment.turn_steps} turn step(s)"
              + (f": {segment.top_labels()}" if segment.labels else "")
              + (f" [{'; '.join(notes)}]" if notes else ""))
        print(f"[Human] Imitation buffer now holds {len(self.replay)} frame(s) "
              f"({self.replay.describe()}); the next PPO updates will copy them.")
        if self.watcher.new_keys:
            saved = None
            try:
                saved = self.env.keymap.save()
            except Exception as exc:
                print(f"[Human] Could not save the keymap: {exc}")
            print(f"[Human] New buttons learned this session: "
                  f"{', '.join(sorted(set(self.watcher.new_keys)))}"
                  + (f" -> {saved}" if saved else ""))


# =============================================================================
# SECTION 2: AUDIO CAPTURE
# =============================================================================

# ---------------------------------------------------------------------------
# 2A. Per-process audio (WASAPI application loopback via ProcTap)
#
# Captures ONLY the audio a chosen process renders, instead of the whole
# default output device. Backed by the 'proc-tap' package, whose native C++
# extension performs the WASAPI application-loopback activation.
#
# A pure-Python/ctypes implementation of the same activation was tried first
# and was consistently refused by Windows on this machine (CO_E_OBJNOTREG,
# 0x8000000E) regardless of apartment, handler agility, or caller language, so
# the proven native backend is used instead.
#
# Requirements / caveats:
#   * pip install proc-tap  (Windows 11 / Server 2022+, prebuilt wheels exist)
#   * Captures the target process tree (the game itself plus any children).
#   * ProcTap delivers 48 kHz stereo float32; this wrapper resamples and
#     downmixes to the rate/channel count the model expects.
#   * Activation can still be refused on some systems. start() returns False
#     in that case and make_audio_capture() falls back - see below.
# ---------------------------------------------------------------------------

try:
    from proctap import ProcessAudioCapture as _ProcTapNative
    _HAS_PROCTAP = True
except Exception as _proctap_error:  # noqa: BLE001
    _ProcTapNative = None
    _HAS_PROCTAP = False
    _PROCTAP_ERROR = _proctap_error
else:
    _PROCTAP_ERROR = None


class ProcessAudioCapture:
    """
    Per-process audio capture with the same interface as AudioCapture.

    The native backend always produces 48 kHz stereo float32; this class
    downmixes to mono and resamples to `sample_rate` (the model works at
    16 kHz, which keeps the observation small).
    """

    def __init__(self, pid: int, sample_rate: int = 16000,
                 buffer_seconds: float = 1.0,
                 resample_quality: str = "fast"):
        self.pid = int(pid)
        self.sample_rate = int(sample_rate)
        self.buffer_len = int(sample_rate * buffer_seconds)
        self.latest_audio = np.zeros(self.buffer_len, dtype=np.float32)
        self.resample_quality = resample_quality
        self.last_error = None
        self.native_format = None

        self._running = False
        self._thread = None
        self._tap = None
        self._lock = threading.Lock()

        # streaming resampler state
        self._src_rate = 48000
        self._resample_pos = 0.0
        self._leftover = np.zeros(0, dtype=np.float32)

    # ---- lifecycle ----
    def start(self) -> bool:
        if sys.platform != "win32":
            self.last_error = "per-process capture is Windows-only"
            return False
        if not _HAS_PROCTAP:
            self.last_error = (f"'proc-tap' not available ({_PROCTAP_ERROR}); "
                               f"install with: python -m pip install proc-tap")
            return False

        try:
            self._tap = _ProcTapNative(self.pid,
                                       resample_quality=self.resample_quality)
            fmt = {}
            try:
                fmt = self._tap.get_format() or {}
            except Exception:
                fmt = {}
            self.native_format = fmt
            self._src_rate = int(fmt.get("sample_rate") or 48000)
            self._tap.start()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._tap = None
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name="ProcessAudioCapture", daemon=True
        )
        self._thread.start()
        return True

    def _capture_loop(self):
        while self._running:
            try:
                chunk = self._tap.read(timeout=0.5)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"read failed: {exc}"
                break
            if not chunk:
                continue
            block = np.frombuffer(chunk, dtype=np.float32)
            mono = self._to_mono(block)
            resampled = self._resample(mono)
            n = min(len(resampled), self.buffer_len)
            if n > 0:
                with self._lock:
                    self.latest_audio = np.roll(self.latest_audio, -n)
                    self.latest_audio[-n:] = resampled[-n:]

    # ---- format conversion ----
    def _to_mono(self, block: np.ndarray) -> np.ndarray:
        """ProcTap output is interleaved stereo float32; average the channels."""
        channels = int((self.native_format or {}).get("channels") or 2)
        if channels <= 1:
            return block
        usable = (block.size // channels) * channels
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        return block[:usable].reshape(-1, channels).mean(axis=1)

    def _resample(self, mono: np.ndarray) -> np.ndarray:
        """
        Streaming linear resample from the native rate down to sample_rate.

        Position is carried across calls so consecutive blocks stay
        phase-continuous (a naive per-block resample would click every chunk).
        """
        if mono.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self._src_rate == self.sample_rate:
            return mono.astype(np.float32, copy=False)

        data = np.concatenate((self._leftover, mono))
        step = self._src_rate / float(self.sample_rate)
        if data.size < 2:
            self._leftover = data
            return np.zeros(0, dtype=np.float32)

        # How many output samples can we safely produce from this buffer?
        n_out = int((data.size - 1 - self._resample_pos) / step) + 1
        if n_out <= 0:
            self._leftover = data
            return np.zeros(0, dtype=np.float32)

        positions = self._resample_pos + step * np.arange(n_out)
        out = np.interp(positions, np.arange(data.size), data).astype(np.float32)

        consumed = int(np.floor(positions[-1])) + 1
        self._leftover = data[consumed:]
        self._resample_pos = positions[-1] + step - consumed
        return out

    # ---- data access ----
    def get_audio(self) -> np.ndarray:
        with self._lock:
            return self.latest_audio.copy()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._tap is not None:
            for method in ("stop", "close"):
                try:
                    getattr(self._tap, method)()
                except Exception:
                    pass
            self._tap = None


AUDIO_MODES = ("auto", "process", "system", "exclude-self", "off")


class AudioCapture:
    """
    Captures loopback audio using the 'soundcard' library.
    If soundcard is unavailable, produces silence so the rest of the
    pipeline still runs.

    IMPORTANT: this captures the whole DEFAULT OUTPUT DEVICE, not the target
    process (the earlier pid argument was a no-op and is gone). Per-process
    capture needs WASAPI process loopback, which 'soundcard' does not expose.
    Practical consequence: whatever you are listening to while the bot trains
    becomes part of its observation. Set AUDIO_MODE = "off" for unattended runs.
    """

    def __init__(self, sample_rate: int = 16000, buffer_seconds: float = 1.0,
                 enabled: bool = True):
        self.sample_rate = sample_rate
        self.enabled = enabled
        self.buffer_len = int(sample_rate * buffer_seconds)
        self.latest_audio = np.zeros(self.buffer_len, dtype=np.float32)
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        if not _HAS_SOUNDCARD or not self.enabled:
            return  # silence fallback
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        try:
            speaker = sc.default_speaker()
            mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)

            # soundcard may not accept arbitrary sample rates on every device.
            # Try the requested rate; fall back to the device default.
            try:
                rec_ctx = mic.recorder(samplerate=self.sample_rate)
            except Exception:
                rec_ctx = mic.recorder(samplerate=48000)
                actual_rate = 48000
            else:
                actual_rate = self.sample_rate

            block = 1024
            with rec_ctx as rec:
                while self._running:
                    data = rec.record(numframes=block)
                    # shape (frames, channels) -> mono float32
                    audio = data.mean(axis=1).astype(np.float32)

                    # If the device ignored our rate, resample to what the
                    # rest of the pipeline expects.
                    if actual_rate != self.sample_rate and len(audio) > 1:
                        audio = np.interp(
                            np.linspace(0, len(audio) - 1, self.sample_rate * block // actual_rate),
                            np.arange(len(audio)),
                            audio,
                        ).astype(np.float32)

                    n = min(len(audio), self.buffer_len)
                    with self._lock:
                        self.latest_audio = np.roll(self.latest_audio, -n)
                        self.latest_audio[-n:] = audio[-n:]
        except Exception as e:
            print(f"[AudioCapture] Error: {e}")

    def get_audio(self) -> np.ndarray:
        with self._lock:
            return self.latest_audio.copy()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# 2B. Audio backend selection
# ---------------------------------------------------------------------------

def make_audio_capture(
    pid: int,
    sample_rate: int = 16000,
    mode: str = "auto",
    enabled: bool = True,
):
    """
    Build the audio capture backend.

    mode:
      auto          - try per-process loopback for `pid`, fall back to
                      system-wide if Windows refuses the activation
      process       - per-process loopback only
      system        - whole default output device (soundcard), reliable
      exclude-self  - everything EXCEPT our own process tree, so the bot does
                      not learn from sounds this script itself makes

    Returns (capture_object, human_readable_description).
    """
    if not enabled or mode == "off":
        return AudioCapture(sample_rate=sample_rate, enabled=False), \
            "disabled (silence)"

    if mode == "system":
        return AudioCapture(sample_rate=sample_rate), "system-wide loopback"

    if mode == "exclude-self":
        cap = ProcessAudioCapture(pid=os.getpid(), sample_rate=sample_rate)
        if cap.start():
            return cap, "everything except this process"
        print(f"[Audio] Per-process exclusion unavailable: {cap.last_error}")
        print("[Audio] Falling back to system-wide loopback.")
        return AudioCapture(sample_rate=sample_rate), "system-wide (fallback)"

    # auto / process: target the game's own process tree
    cap = ProcessAudioCapture(pid=pid, sample_rate=sample_rate)
    if cap.start():
        return cap, f"process loopback (pid {pid})"

    if mode == "process":
        print(f"[Audio] Per-process capture failed: {cap.last_error}")
        print("[Audio] Running with SILENCE. Set AUDIO_MODE = \"system\" "
              "instead if you want sound.")
        return AudioCapture(sample_rate=sample_rate, enabled=False), \
            "silence (process mode requested)"

    print(f"[Audio] Per-process capture unavailable ({cap.last_error}).")
    print("[Audio] Falling back to SYSTEM-WIDE audio: anything you play while")
    print("[Audio] training becomes part of the observation.")
    return AudioCapture(sample_rate=sample_rate), "system-wide (fallback)"


# =============================================================================
# SECTION 2.5: HOTKEYS, PAUSE STATE & ROLLING CHECKPOINTS
# =============================================================================

# Win32 bits needed to register system-wide hotkeys from a worker thread.
WM_HOTKEY = 0x0312
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

_MODIFIERS = {
    "ctrl": MOD_CONTROL, "control": MOD_CONTROL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN, "windows": MOD_WIN, "super": MOD_WIN,
}

_NAMED_KEYS = {
    "SPACE": 0x20, "ESC": 0x1B, "ESCAPE": 0x1B, "TAB": 0x09, "ENTER": 0x0D,
    "BACKSPACE": 0x08, "INSERT": 0x2D, "DELETE": 0x2E, "HOME": 0x24,
    "END": 0x23, "PAGEUP": 0x21, "PAGEDOWN": 0x22,
    "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27,
}


def parse_hotkey(spec: str) -> Tuple[int, int]:
    """
    Parse 'ctrl+shift+f8' / 'f8' into (modifiers, virtual_key).
    Raises ValueError with a readable message on a bad spec.
    """
    modifiers = 0
    key_vk = None

    for part in str(spec).lower().replace(" ", "").split("+"):
        if not part:
            continue
        if part in _MODIFIERS:
            modifiers |= _MODIFIERS[part]
            continue
        upper = part.upper()
        if upper in _NAMED_KEYS:
            key_vk = _NAMED_KEYS[upper]
            continue
        if part.startswith("f") and part[1:].isdigit() and 1 <= int(part[1:]) <= 24:
            key_vk = 0x70 + (int(part[1:]) - 1)   # VK_F1 == 0x70
            continue
        if len(part) == 1 and part.isalnum():
            key_vk = ord(part.upper())
            continue
        raise ValueError(f"Unrecognised hotkey part '{part}' in '{spec}'")

    if key_vk is None:
        raise ValueError(f"Hotkey '{spec}' has no key (only modifiers)")
    return modifiers, key_vk


class HotkeyController:
    """
    Registers global hotkeys on a dedicated thread that owns a Win32 message
    loop, and exposes thread-safe flags for the training loop to poll.

    Hotkeys fire regardless of which window has focus, so the bot can be
    paused, checkpointed or stopped while you are working in another app.
    """

    # Windows returns ERROR_HOTKEY_ALREADY_REGISTERED when another app owns
    # the combination; we surface that as a clear startup warning.
    def __init__(
        self,
        pause_spec: str = HOTKEY_PAUSE,
        save_spec: str = HOTKEY_SAVE,
        quit_spec: str = HOTKEY_QUIT,
    ):
        self.specs = {"pause": pause_spec, "save": save_spec, "quit": quit_spec}
        self.actions: "queue.Queue[str]" = queue.Queue()
        self.messages: "queue.Queue[str]" = queue.Queue()
        self._registered: Dict[str, str] = {}
        self._id_to_action: Dict[int, str] = {}
        self._thread: Optional[threading.Thread] = None
        self._next_id = 0xB001

    @property
    def available(self) -> bool:
        return bool(self._registered)

    def start(self) -> bool:
        """Start the hotkey thread. Returns True if at least one key bound."""
        try:
            self._thread = threading.Thread(
                target=self._message_loop, name="HotkeyController", daemon=True
            )
            self._thread.start()
            # Registration happens on the listener thread; wait for its verdict.
            self._thread.join(timeout=3.0)
        except Exception as exc:
            self.messages.put(f"Hotkeys unavailable: {exc}")
            return False
        return self.available

    def _message_loop(self):
        try:
            for action, spec in self.specs.items():
                if not spec:
                    continue
                try:
                    modifiers, key_vk = parse_hotkey(spec)
                except ValueError as exc:
                    self.messages.put(str(exc))
                    continue

                hotkey_id = self._next_id
                self._next_id += 1
                ok = ctypes.windll.user32.RegisterHotKey(
                    None, hotkey_id, modifiers | MOD_NOREPEAT, key_vk
                )
                if ok:
                    self._registered[action] = spec
                    self._id_to_action[hotkey_id] = action
                else:
                    err = ctypes.get_last_error() if hasattr(ctypes, "get_last_error") else 0
                    self.messages.put(
                        f"Could not register {spec.upper()} for '{action}' "
                        f"(error {err}) - another app may already use it."
                    )

            if not self._registered:
                return

            # Classic message pump; RegisterHotKey delivers WM_HOTKEY here.
            # The struct is built by hand because wintypes is not guaranteed
            # to be imported alongside ctypes.
            class _MSG(ctypes.Structure):
                _fields_ = [
                    ("hwnd", ctypes.c_void_p),
                    ("message", ctypes.c_uint),
                    ("wParam", ctypes.c_void_p),
                    ("lParam", ctypes.c_void_p),
                    ("time", ctypes.c_uint),
                    ("pt_x", ctypes.c_long),
                    ("pt_y", ctypes.c_long),
                ]

            msg = _MSG()
            while ctypes.windll.user32.GetMessageW(
                ctypes.byref(msg), None, 0, 0
            ) > 0:
                if msg.message == WM_HOTKEY:
                    action = self._id_to_action.get(int(msg.wParam or 0))
                    if action:
                        self.actions.put(action)
        except Exception as exc:
            self.messages.put(f"Hotkey listener stopped: {exc}")
        finally:
            self._unregister_all()

    def _unregister_all(self):
        for hid in list(self._id_to_action):
            try:
                ctypes.windll.user32.UnregisterHotKey(None, hid)
            except Exception:
                pass
        self._id_to_action.clear()
        self._registered.clear()

    def poll(self) -> List[str]:
        """Drain pending actions. Non-blocking."""
        actions = []
        while True:
            try:
                actions.append(self.actions.get_nowait())
            except queue.Empty:
                break
        return actions

    def drain_messages(self) -> List[str]:
        messages = []
        while True:
            try:
                messages.append(self.messages.get_nowait())
            except queue.Empty:
                break
        return messages

    def describe(self) -> str:
        if not self._registered:
            return "none registered"
        order = ["pause", "save", "quit"]
        parts = [f"{a.upper()}={self._registered[a].upper()}"
                 for a in order if a in self._registered]
        return ", ".join(parts)


class CheckpointManager:
    """
    Writes rollout/training state to disk atomically, keeps only the newest
    N timestamped checkpoints, and maintains a rolling 'last.pt' that always
    points at the freshest state (safe to use as a resume file).

    Ordering uses a monotonic sequence number embedded in each filename, not
    the training step. That matters because a resumed run restarts its step
    counter, so step numbers alone can go backwards and would make an older
    checkpoint look newer.
    """

    def __init__(self, directory: str = CHECKPOINT_DIR, keep: int = CHECKPOINT_KEEP):
        self.directory = os.path.abspath(directory)
        self.keep = max(1, int(keep))
        os.makedirs(self.directory, exist_ok=True)
        self.last_path = os.path.join(self.directory, "last.pt")
        self.index_path = os.path.join(self.directory, "index.json")
        self._entries: List[dict] = self._load_index()
        self._seq = self._highest_seq_on_disk()

    # ---- sequence numbers ----
    def _highest_seq_on_disk(self) -> int:
        """Highest checkpoint number already present, so we never reuse one."""
        highest = 0
        try:
            for name in os.listdir(self.directory):
                if name.startswith("checkpoint_") and name.endswith(".pt"):
                    try:
                        highest = max(highest, int(name.split("_")[1]))
                    except (IndexError, ValueError):
                        continue
        except OSError:
            pass
        return highest

    # ---- index bookkeeping ----
    def _load_index(self) -> List[dict]:
        try:
            with open(self.index_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return [e for e in data if isinstance(e, dict)]
        except Exception:
            pass
        return []

    def _save_index(self):
        tmp = self.index_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._entries, fh, indent=2)
            os.replace(tmp, self.index_path)
        except Exception as exc:
            print(f"[Checkpoint] Could not write index: {exc}")
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _discover_existing(self) -> List[dict]:
        """Rebuild the index from disk if index.json was lost or is empty."""
        found = []
        try:
            for name in os.listdir(self.directory):
                if not (name.startswith("checkpoint_") and name.endswith(".pt")):
                    continue
                path = os.path.join(self.directory, name)
                try:
                    seq = int(name.split("_")[1])
                except (IndexError, ValueError):
                    seq = 0
                found.append({
                    "path": path,
                    "seq": seq,
                    "step": seq,
                    "mtime": os.path.getmtime(path),
                    "reason": "discovered",
                })
        except OSError:
            pass
        return found

    # ---- writing ----
    def save(self, payload: dict, step: int, reason: str = "periodic") -> Optional[str]:
        """
        Serialise payload to a timestamped checkpoint plus 'last.pt', then
        prune old checkpoints. Returns the new checkpoint path, or None if the
        write failed.
        """
        self._seq += 1
        seq = self._seq
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{seq:09d}_step{int(step):09d}_{stamp}.pt"
        path = os.path.join(self.directory, filename)
        tmp = path + ".tmp"

        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)  # atomic: the timestamped file is complete
            # Keep a rolling 'last.pt' copy of the freshest state for easy resume.
            try:
                if not atomic_torch_save(payload, self.last_path):
                    print("[Checkpoint] Rolling last.pt copy failed (timestamped file is fine).")
            except Exception as exc:
                print(f"[Checkpoint] Rolling last.pt copy failed: {exc}")
        except Exception as exc:
            print(f"[Checkpoint] FAILED to save: {exc}")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            return None

        if not self._entries:
            self._entries = self._discover_existing()

        self._entries.append({
            "path": path,
            "seq": seq,
            "step": int(step),
            "mtime": time.time(),
            "reason": reason,
        })
        self._prune()
        self._save_index()
        return path

    def _prune(self):
        """
        Keep the newest `keep` checkpoints, ordered by the monotonic sequence
        number in the filename (with mtime and path as tie-breakers) so that a
        just-written file is never the one deleted.

        Files on disk that the index does not know about (a lost index.json, or
        checkpoints from an older naming scheme) are folded in here, so the
        directory cannot silently grow past the limit.
        """
        known = {e.get("path", "") for e in self._entries}
        for orphan in self._discover_existing():
            if orphan.get("path") not in known:
                self._entries.append(orphan)
                known.add(orphan.get("path"))

        # De-duplicate by path, keeping the last entry seen.
        unique: Dict[str, dict] = {}
        for entry in self._entries:
            unique[entry.get("path", "")] = entry
        entries = list(unique.values())

        def sort_key(entry: dict) -> tuple:
            seq = entry.get("seq")
            if seq is None:
                seq = entry.get("step", 0)
            try:
                seq = int(seq)
            except (TypeError, ValueError):
                seq = 0
            return (seq, float(entry.get("mtime", 0.0)), str(entry.get("path", "")))

        entries.sort(key=sort_key)
        survivors = entries[-self.keep:] if len(entries) > self.keep else entries
        doomed = entries[: max(0, len(entries) - self.keep)]

        for entry in doomed:
            path = entry.get("path", "")
            if not path or os.path.basename(path) == "last.pt":
                continue
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                print(f"[Checkpoint] Could not delete {path}: {exc}")

        self._entries = survivors

    def list_kept(self) -> List[dict]:
        try:
            alive = [e for e in self._entries if os.path.exists(e.get("path", ""))]
            return sorted(alive, key=lambda e: int(e.get("seq", e.get("step", 0))))
        except Exception:
            return []


def atomic_torch_save(payload: dict, path: str) -> bool:
    """torch.save that cannot leave a truncated file behind."""
    tmp = path + ".tmp"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return True
    except Exception as exc:
        print(f"[Save] FAILED for {path}: {exc}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def model_has_nonfinite(model: nn.Module) -> List[str]:
    """Return the names of parameters holding NaN/Inf (empty list if healthy)."""
    bad = []
    try:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not torch.isfinite(param).all().item():
                    bad.append(name)
    except Exception:
        return []
    return bad


class SignalController:
    """
    Owns everything the user can press to steer a running training session:

      * global hotkeys (pause / save / quit) via HotkeyController
      * Ctrl+C (SIGINT) and Ctrl+Break (SIGBREAK) -> clean stop
      * wall-clock checkpoint scheduling, driven by the training loop

    The training loop polls it once per step; the handlers themselves never
    touch the model, so there is no risk of saving from inside a signal
    handler while a forward pass is mid-flight.
    """

    def __init__(
        self,
        interval_sec: float = CHECKPOINT_INTERVAL_SEC,
        enable_hotkeys: bool = True,
        pause_hotkey: str = HOTKEY_PAUSE,
        save_hotkey: str = HOTKEY_SAVE,
        quit_hotkey: str = HOTKEY_QUIT,
    ):
        self.interval_sec = max(0.05, float(interval_sec))
        self._paused = False
        self._stop_requested = False
        self._interrupt_count = 0
        self._last_time = time.monotonic()
        self._accum = 0.0
        self._manual_save_requested = False
        self.stop_reason = "running"
        self._pause_message_printed = False

        self.hotkeys: Optional[HotkeyController] = None
        if enable_hotkeys:
            self.hotkeys = HotkeyController(pause_hotkey, save_hotkey, quit_hotkey)

    # ---- lifecycle ----
    def start(self) -> None:
        self._last_time = time.monotonic()
        self._accum = 0.0
        if self.hotkeys is not None:
            if self.hotkeys.start():
                for msg in self.hotkeys.drain_messages():
                    print(f"[Hotkeys] {msg}")
                print(f"[Hotkeys] Active -> {self.hotkeys.describe()}  "
                      f"(these work from any window)")
            else:
                for msg in self.hotkeys.drain_messages():
                    print(f"[Hotkeys] {msg}")
                print("[Hotkeys] No global hotkeys active. Ctrl+C still saves and quits.")
        self._install_signal_handlers()

    def _install_signal_handlers(self):
        for sig_name in ("SIGINT", "SIGBREAK"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                # Not running in the main thread; Ctrl+C still works normally.
                pass

    def _on_signal(self, signum, frame):
        self._interrupt_count += 1
        name = "Ctrl+C" if signum == getattr(signal, "SIGINT", 2) else "Ctrl+Break"
        if self._interrupt_count == 1:
            self.stop_reason = "ctrl_c"
            self._stop_requested = True
            print(f"\n[Signal] {name} received. Saving a checkpoint and "
                  f"shutting down cleanly...")
            print("[Signal] (Press Ctrl+C again to abandon the save and exit now.)")
        else:
            print(f"\n[Signal] Second {name} - exiting immediately.")
            raise KeyboardInterrupt

    # ---- polling API used by the training loop ----
    def service(self) -> List[str]:
        """Note wall-clock progress. Returns messages worth printing."""
        now = time.monotonic()
        self._accum += now - self._last_time
        self._last_time = now

        for msg in (self.hotkeys.drain_messages() if self.hotkeys else []):
            print(f"[Hotkeys] {msg}")

        notes = []
        for action in (self.hotkeys.poll() if self.hotkeys else []):
            if action == "pause":
                self._paused = not self._paused
                self._pause_message_printed = False
                notes.append("PAUSED - the bot is no longer sending input."
                             if self._paused else "RESUMED - the bot is active again.")
            elif action == "save":
                self._manual_save_requested = True
                notes.append("Manual checkpoint requested.")
            elif action == "quit":
                self.stop_reason = "hotkey"
                self._stop_requested = True
                notes.append("Quit hotkey pressed - saving and shutting down.")
        return notes

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def claim_pause_notice(self) -> bool:
        """True exactly once per pause, so the message prints a single time."""
        if self._paused and not self._pause_message_printed:
            self._pause_message_printed = True
            return True
        return False

    def checkpoint_due(self) -> Optional[str]:
        """Return 'manual'/'periodic' when a save is due, else None."""
        if self._manual_save_requested:
            self._manual_save_requested = False
            return "manual"
        if self._accum >= self.interval_sec:
            self._accum -= self.interval_sec
            return "periodic"
        return None

    def wait_while_paused(self, sleep: float = 0.1) -> None:
        """Idle without consuming CPU or sending any input to the game."""
        while self._paused and not self._stop_requested:
            for msg in self.service():
                print(f"[Control] {msg}")
            time.sleep(sleep)


# =============================================================================
# SECTION 3: SwiGLU + RoPE TRANSFORMER MODEL
#
# Two small causal transformers: one over visual frame tokens, one over audio
# frame tokens. Each frame is compressed to a single token by a convolutional
# encoder, then the sequence is mixed by pre-norm blocks that use
#   * rotary position embeddings (RoPE) on queries and keys, so attention is
#     relative and position handling costs no parameters, and
#   * a SwiGLU feed-forward network, which is the gated MLP that replaced the
#     plain 2-layer MLP in modern transformers
#     (SwiGLU(x) = W_down( SiLU(W_gate x) * W_up x )).
#
# Everything here is deliberately shaped for a CPU: attention is O(T^2) in the
# *sequence of frames* (SEQ_LEN), never in pixels, and the CNN does the heavy
# lifting on the image.
# =============================================================================


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute the norm in float32 so a half-precision run stays stable.
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32.to(dtype)) * self.weight


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Rotary position embeddings (RoPE), Su-style: the rotation table is built
    once at init and sliced to whatever sequence length actually arrives, so a
    model built for one SEQ_LEN still runs on a shorter or longer window
    without re-allocating cache or crashing.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE needs an even attention head_dim")
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(self.max_seq_len).float()
        freqs = torch.outer(t, inv)               # (max_seq_len, head_dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)   # (max_seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q, k):
        """q, k: (B, H, T, head_dim). Returns the rotated pair."""
        t = q.shape[-2]
        if t > self.cos_cached.shape[0]:
            # Grow the table rather than dying: a longer window than the one
            # this model was built for is a supported (if slower) case.
            self._extend(t)
        cos = self.cos_cached[:t].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:t].to(dtype=q.dtype, device=q.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

    @torch.no_grad()
    def _extend(self, t: int):
        base = getattr(self, "_rope_base", 10000.0)
        inv = 1.0 / (base ** (torch.arange(
            0, self.head_dim, 2, device=self.cos_cached.device).float() / self.head_dim))
        freqs = torch.outer(
            torch.arange(t, device=self.cos_cached.device).float(), inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
        self.max_seq_len = int(t)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.

    Attention is computed in float32 even when the model runs in half
    precision: the softmax over a long window is where fp16 quietly turns into
    NaNs, and that is exactly the failure that makes a run look stalled.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model {d_model} is not divisible by num_heads {num_heads}")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = float(dropout)

    def forward(self, x, cos_sin=None):
        B, T, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, T, H, D) -> (B, H, T, D)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if cos_sin is not None:
            rope = cos_sin
            q, k = rope(q, k)

        # Causal mask: token t may only look at tokens <= t. Built on the fly
        # so it never needs to be stored in a checkpoint.
        att = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att.float(), dim=-1).to(v.dtype)
        if self.dropout > 0.0 and self.training:
            att = F.dropout(att, p=self.dropout)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network:
        SwiGLU(x) = W_down( SiLU(W_gate x) * (W_up x) )

    `hidden` defaults to the 2/3 * 4 * d_model width that keeps the parameter
    count level with a classic 4x MLP, rounded up to a multiple of 8. That is
    strictly more expressive than two plain linears for the same budget.
    """

    def __init__(self, d_model: int, hidden: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        if hidden is None:
            hidden = int(round(2.0 / 3.0 * 4.0 * d_model))
            hidden = max(8, ((hidden + 7) // 8) * 8)
        self.hidden = int(hidden)
        self.gate_up = nn.Linear(d_model, 2 * self.hidden, bias=False)
        self.down = nn.Linear(self.hidden, d_model, bias=False)
        self.dropout = float(dropout)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.down(x)
        if self.dropout > 0.0 and self.training:
            x = F.dropout(x, p=self.dropout)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm -> attention -> RMSNorm -> SwiGLU."""

    def __init__(self, d_model: int, num_heads: int, ffn_hidden: Optional[int] = None,
                 dropout: float = 0.0, norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLU(d_model, hidden=ffn_hidden, dropout=dropout)

    def forward(self, x, cos_sin=None):
        x = x + self.attn(self.norm1(x), cos_sin=cos_sin)
        x = x + self.ffn(self.norm2(x))
        return x


class MultiHeadSwiGLUTransformer(nn.Module):
    """
    SwiGLU + RoPE transformer with separate visual and audio encoders, fused
    for policy and value outputs.

    Shape contract (unchanged from the gMLP it replaces, so the environment,
    the human-imitation path and the rollout buffers all keep working):
        visual_seq: (B, T, 3, frame_size, frame_size)
        audio_seq:  (B, T, samples)
        -> logits (B, num_actions), value (B,)
    """

    def __init__(
        self,
        visual_dim: int = 128,
        audio_dim: int = 64,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        seq_len: int = 16,
        num_actions: int = 16,
        frame_size: int = 96,
        num_heads: int = 4,
        ffn_hidden: Optional[int] = None,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
        pool: str = "last",
        visual_pool: int = 3,
        audio_downsample: int = 8,
    ):
        super().__init__()
        self.frame_size = int(frame_size)
        self.seq_len = int(seq_len)
        self.pool = str(pool).lower()
        self.arch = ARCH_NAME
        self.audio_downsample = max(1, int(audio_downsample))

        # The three conv stages below consume 8+4+3 pixels of support, so tiny
        # frames produce a negative padded size. Reject that up front with a
        # readable message instead of a deep torch traceback.
        if self.frame_size < 40:
            raise ValueError(
                f"IMG_SIZE {self.frame_size} is too small for this encoder; "
                f"use 40 or larger (96 is the default)"
            )
        if visual_dim % num_heads != 0:
            raise ValueError(
                f"visual_dim {visual_dim} is not divisible by num_heads {num_heads}")

        # ---- visual encoder: one CNN token per frame ------------------------
        # The adaptive pool keeps the flattened size constant, so IMG_SIZE can
        # change without breaking the heads or forcing new weights.
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((visual_pool, visual_pool)),
            nn.Flatten(),
            nn.Linear(64 * visual_pool * visual_pool, visual_dim),
        )

        # ---- audio encoder: one 1D-conv token per audio chunk ---------------
        # `audio_downsample` average-pools the raw waveform before any
        # convolution. This is the single biggest CPU win in the model: a 16 kHz
        # chunk run through paired 3x3 convs costs about as much as a whole
        # 96x96 visual frame, and the fine detail being discarded is well below
        # what a game's mix actually carries. It runs inside forward() rather
        # than in the environment so old checkpoints and the human-imitation
        # path keep feeding raw audio.
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=4), nn.ReLU(),
            nn.Conv1d(64, audio_dim, kernel_size=5, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # ---- transformer trunks ---------------------------------------------
        # Visual heads must divide visual_dim; the audio width is chosen as the
        # largest head count that divides audio_dim, capped to keep attention
        # cheap.
        audio_heads = max(
            (h for h in range(min(4, audio_dim), 0, -1) if audio_dim % h == 0),
            default=1,
        )
        self.visual_blocks = nn.ModuleList([
            TransformerBlock(visual_dim, num_heads, ffn_hidden=ffn_hidden,
                             dropout=dropout, norm_eps=norm_eps)
            for _ in range(num_blocks)
        ])
        self.audio_blocks = nn.ModuleList([
            TransformerBlock(audio_dim, audio_heads,
                             ffn_hidden=(ffn_hidden if ffn_hidden and audio_dim == visual_dim
                                         else None),
                             dropout=dropout, norm_eps=norm_eps)
            for _ in range(num_blocks)
        ])
        self.visual_norm = RMSNorm(visual_dim, eps=norm_eps)
        self.audio_norm = RMSNorm(audio_dim, eps=norm_eps)
        self.visual_rope = RotaryEmbedding(visual_dim // num_heads,
                                           max(self.seq_len, 2), base=rope_base)
        self.visual_rope._rope_base = float(rope_base)
        self.audio_rope = RotaryEmbedding(audio_dim // audio_heads,
                                          max(self.seq_len, 2), base=rope_base)
        self.audio_rope._rope_base = float(rope_base)

        # ---- fusion + heads -------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)
        # Put the policy head where PPO needs it: a near-uniform Categorical.
        # Without this an untrained net can start saturated, and a policy that
        # is already certain of one action never explores its way out.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _pool(self, x):
        """(B, T, D) -> (B, D). Causal, so 'last' is the summary token."""
        if self.pool == "mean":
            return x.mean(dim=1)
        if self.pool == "max":
            return x.max(dim=1).values
        return x[:, -1, :]

    def _downsample_audio(self, audio_seq):
        """
        (B, T, samples) -> (B, T, samples // audio_downsample).

        Average-pooling the raw waveform is what makes the audio branch cheap;
        uses a reshape rather than a conv so it costs no parameters and stays
        fast on a CPU. A tail shorter than one window is dropped.
        """
        f = self.audio_downsample
        if f <= 1:
            return audio_seq
        samples = audio_seq.shape[-1]
        usable = (samples // f) * f
        if usable <= 0:
            return audio_seq
        trimmed = audio_seq[..., :usable]
        return trimmed.reshape(*trimmed.shape[:-1], usable // f, f).mean(dim=-1)

    def forward(self, visual_seq, audio_seq):
        B, T = visual_seq.shape[0], visual_seq.shape[1]

        vis = self.visual_cnn(
            visual_seq.reshape(B * T, 3, self.frame_size, self.frame_size)
        ).view(B, T, -1)
        for block in self.visual_blocks:
            vis = block(vis, cos_sin=self.visual_rope)
        vis_feat = self._pool(self.visual_norm(vis))

        aud = self._downsample_audio(audio_seq).reshape(B * T, 1, -1)
        aud = self.audio_cnn(aud).view(B, T, -1)
        for block in self.audio_blocks:
            aud = block(aud, cos_sin=self.audio_rope)
        aud_feat = self._pool(self.audio_norm(aud))

        fused = self.fusion(torch.cat([vis_feat, aud_feat], dim=-1))
        return self.policy_head(fused), self.value_head(fused).squeeze(-1)


# Kept as an alias so any external script that imported the old name still
# resolves to the current architecture.
MultiHeadgMLP = MultiHeadSwiGLUTransformer


# =============================================================================
# SECTION 4: BACKGROUND INPUT INJECTION
# =============================================================================

class InjectionLock:
    """
    While held, no SendInput can leave this process.

    --calibrate wraps its entire duration in one of these, so the game is
    guaranteed to be yours while you demonstrate: not just the actions the bot
    chooses, but any stray call anywhere in the program. It reports how many
    attempts were refused when it is released, which makes a hidden code path
    loud instead of silent.

    Training does not use this lock. There the bot is *supposed* to drive, and
    a human takeover is handled by BackgroundInput.suspended instead - a soft
    switch the training loop flips for as long as your hands are on the
    controls, which would otherwise be reported as a bug every three seconds.
    """

    def __init__(self, reason: str = "this mode"):
        self.reason = reason
        self._blocked_before = 0

    def __enter__(self) -> "InjectionLock":
        global INPUT_INJECTION_ENABLED
        self._previous = INPUT_INJECTION_ENABLED
        INPUT_INJECTION_ENABLED = False
        self._blocked_before = _INJECTION_BLOCKED_TOTAL[0]
        return self

    def __exit__(self, exc_type, exc, tb):
        global INPUT_INJECTION_ENABLED
        INPUT_INJECTION_ENABLED = self._previous
        blocked = _INJECTION_BLOCKED_TOTAL[0] - self._blocked_before
        if blocked:
            print(f"[Input] {blocked} injection attempt(s) were refused while "
                  f"{self.reason} ran. The bot did not press anything.")
        return False

    @staticmethod
    def note_block():
        _INJECTION_BLOCKED_TOTAL[0] += 1


# Counts every refused injection process-wide, so the lock can report totals.
_INJECTION_BLOCKED_TOTAL = [0]


class BackgroundInput:
    """
    Injects real keyboard and mouse input into the target window.

    Designed for a machine dedicated to the bot: the game window owns the
    foreground, so SendInput delivers keystrokes and cursor deltas straight to
    it. That is the only arrangement in which mouse-look works, because games
    read look from raw mouse deltas rather than from absolute cursor position.

    Which keys are legal comes from the calibrated Keymap; the VK table below is
    only the fallback used before calibration has run.

    Kept the name for continuity; it is no longer "background" in the sense of
    sharing the machine with you.
    """

    VK = {
        'W': 0x57, 'A': 0x41, 'S': 0x53, 'D': 0x44,
        'SPACE': 0x20, 'E': 0x45, 'Q': 0x51, 'ESC': 0x1B, 'TAB': 0x09,
        '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
        '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
        # Games bind the sided variants, so a left/right-specific code is stored
        # for modifiers; the generic VK_SHIFT (0x10) is a different key and many
        # games do not react to it.
        'LSHIFT': 0xA0, 'RSHIFT': 0xA1, 'SHIFT': 0xA0,
        'LCTRL': 0xA2, 'RCTRL': 0xA3, 'CTRL': 0xA2,
        'LALT': 0xA4, 'RALT': 0xA5, 'ALT': 0xA4,
    }

    # Keys that need the extended-key flag.
    _EXTENDED = {0xA1, 0xA3, 0xA5, 0x2D, 0x2E, 0x24, 0x23, 0x21, 0x22,
                 0x25, 0x26, 0x27, 0x28}

    def __init__(self, hwnd: int, keymap: Optional["Keymap"] = None):
        self.hwnd = hwnd
        self.keymap = keymap
        self.blocked_count = 0
        self.last_blocked: Optional[str] = None
        self._focused = False
        self.focus_warned = False
        self.last_send_count = None
        self.last_send_error = None
        # Soft switch for human takeovers: while True this object injects
        # nothing at all, and does so silently (it is expected, not a bug).
        # BackgroundGameEnv.suspend_bot()/resume_bot() own it.
        self.suspended = False
        # Keys/buttons this object is pressing right now as part of a tap or a
        # click (not a hold), so nothing reads them as somebody else's input.
        self.transient_vks: set = set()

    # ---- focus ----
    def _ensure_focus(self) -> bool:
        """
        Make sure the target window is foreground.

        On a dedicated machine this is normally already true and costs one
        cheap comparison. It exists so that a stray click on another window
        cannot silently send the bot's keystrokes somewhere else.
        """
        if win32gui.GetForegroundWindow() == self.hwnd:
            return True
        try:
            win32gui.SetForegroundWindow(self.hwnd)
        except Exception:
            try:
                fg = win32gui.GetForegroundWindow()
                fg_thread = win32process.GetWindowThreadProcessId(fg)[0]
                my_thread = ctypes.windll.kernel32.GetCurrentThreadId()
                ctypes.windll.user32.AttachThreadInput(my_thread, fg_thread, True)
                win32gui.SetForegroundWindow(self.hwnd)
                ctypes.windll.user32.AttachThreadInput(my_thread, fg_thread, False)
            except Exception:
                return False
        return win32gui.GetForegroundWindow() == self.hwnd

    def begin_action(self):
        """Assert focus once per action rather than once per key."""
        if self._focused:
            return
        self._focused = self._ensure_focus()
        if not self._focused and not self.focus_warned:
            self.focus_warned = True
            print("[Input] WARNING: could not focus the game window. Input will "
                  "not reach it. Click the game once and check that no other "
                  "window is stealing focus.")

    def end_action(self):
        """
        Deliberately keeps focus on the game.

        This is the dedicated-machine design: the game stays foreground so
        input keeps landing. Nothing is handed back to another window.
        """
        self._focused = False

    # ---- SendInput ----
    @staticmethod
    def _input_structs():
        ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 \
            else ctypes.c_ulong

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
            # The union must be as large as its biggest member (MOUSEINPUT),
            # which is what makes sizeof(INPUT) == 40 on x64. Sizing it by
            # KEYBDINPUT alone gives 32 and SendInput then fails silently with
            # ERROR_INVALID_PARAMETER.
            class _I(ctypes.Union):
                _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]
            _anonymous_ = ("i",)
            _fields_ = [("type", ctypes.c_ulong), ("i", _I)]

        return KEYBDINPUT, MOUSEINPUT, INPUT

    def _blocked_injection(self, what: str) -> bool:
        """
        Refuse to inject anything while input is globally disabled.

        This is the last line of defence. run_calibration disables injection for
        its whole duration, so even a code path that nobody remembered about
        cannot press a key while you are teaching. Every attempt is counted and
        reported.
        """
        if INPUT_INJECTION_ENABLED:
            return False
        self.blocked_count += 1
        self.last_blocked = what
        InjectionLock.note_block()
        if self.blocked_count <= 5:
            print(f"[Input] BLOCKED an injection attempt ({what}) - input is "
                  f"disabled in this mode. This is a bug; please report it.")
            if INJECTION_AUDIT:
                import traceback
                traceback.print_stack(limit=6)
        return True

    def _send_input_key(self, vk: int, key_up: bool):
        if self.suspended:
            # The controls are yours; this is deliberate and silent.
            self.last_send_count = 0
            return
        if self._blocked_injection(f"key 0x{int(vk):02X} "
                                   f"{'up' if key_up else 'down'}"):
            self.last_send_count = 0
            return
        KEYBDINPUT, _MOUSEINPUT, INPUT = self._input_structs()
        scan = win32api.MapVirtualKey(vk, 0)
        extended = vk in self._EXTENDED
        flags = 0x0002 if key_up else 0          # KEYEVENTF_KEYUP
        if extended:
            flags |= 0x0001                      # KEYEVENTF_EXTENDEDKEY
        inp = INPUT()
        inp.type = 1                             # INPUT_KEYBOARD
        inp.ki = KEYBDINPUT(vk, scan, flags, 0, 0)
        self.last_send_count = ctypes.windll.user32.SendInput(
            1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if not self.last_send_count:
            self.last_send_error = ctypes.get_last_error()

    def _send_input_mouse(self, dx: int, dy: int, flags: int):
        if self.suspended:
            self.last_send_count = 0
            return
        if self._blocked_injection(f"mouse dx={dx} dy={dy} flags=0x{flags:04X}"):
            self.last_send_count = 0
            return
        _KEYBDINPUT, MOUSEINPUT, INPUT = self._input_structs()
        inp = INPUT()
        inp.type = 0                             # INPUT_MOUSE
        inp.mi = MOUSEINPUT(int(dx), int(dy), 0, flags, 0, 0)
        self.last_send_count = ctypes.windll.user32.SendInput(
            1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if not self.last_send_count:
            self.last_send_error = ctypes.get_last_error()

    # ---- keys ----
    def set_keymap(self, keymap: "Keymap"):
        """Adopt the calibrated whitelist, so key names resolve through it."""
        self.keymap = keymap

    def vk_for(self, key: str) -> Optional[int]:
        """Look a key name up, preferring the calibrated keymap."""
        if self.keymap is not None:
            vk = self.keymap.vk(key)
            if vk:
                return vk
        return self.VK.get(str(key).upper())

    def press_key(self, key: str):
        vk = self.vk_for(key)
        if vk:
            self._send_input_key(vk, False)

    def release_key(self, key: str):
        vk = self.vk_for(key)
        if vk:
            self._send_input_key(vk, True)

    def press_vk(self, vk: int):
        """Press a raw virtual key. Mouse buttons are VKs too, so holds work."""
        if vk:
            self._send_input_key(int(vk), False)

    def release_vk(self, vk: int):
        if vk:
            self._send_input_key(int(vk), True)

    def tap_key(self, key: str, duration: float = 0.05):
        vk = self.vk_for(key)
        if not vk:
            return
        # A tap is down-then-up with a sleep in between, so the key reads as
        # "physically down" while the bot does not have it in a hold set. Note
        # it in transient_vks so the human-takeover watcher's key-state probe
        # never mistakes the bot's own tap for your hands.
        self.transient_vks.add(int(vk))
        try:
            self._send_input_key(vk, False)
            time.sleep(duration)
            self._send_input_key(vk, True)
        finally:
            self.transient_vks.discard(int(vk))

    def mouse_move(self, dx: int, dy: int):
        """
        Relative mouse movement, via SendInput.

        This is a real cursor delta, which is the only thing a game's
        mouse-look responds to - message-based approaches cannot turn the view
        at all. Requires the game window to be focused.
        """
        self._send_input_mouse(dx, dy, 0x0001)      # MOUSEEVENTF_MOVE

    def mouse_down(self, button: str = 'left'):
        flags = _MOUSE_DOWN_FLAG.get(button)
        if flags:
            self._send_input_mouse(0, 0, flags)

    def mouse_up(self, button: str = 'left'):
        flags = _MOUSE_UP_FLAG.get(button)
        if flags:
            self._send_input_mouse(0, 0, flags)

    def mouse_click(self, button: str = 'left', duration: float = 0.05):
        # MOUSEEVENTF_LEFTDOWN/UP, RIGHTDOWN/UP, MIDDLEDOWN/UP
        flags = {'left': (0x0002, 0x0004),
                 'right': (0x0008, 0x0010),
                 'middle': (0x0020, 0x0040)}.get(button)
        if not flags:
            return
        vk = {"left": 0x01, "right": 0x02, "middle": 0x04}.get(button)
        if vk:
            self.transient_vks.add(vk)
        try:
            self._send_input_mouse(0, 0, flags[0])
            time.sleep(duration)
            self._send_input_mouse(0, 0, flags[1])
        finally:
            if vk:
                self.transient_vks.discard(vk)


# =============================================================================
# SECTION 5: GAME-AGNOSTIC INTRINSIC REWARD
# =============================================================================

class IntrinsicReward:
    """
    Game-agnostic reward, shaped so that the only way to score is to keep
    interacting with the game and to keep finding situations the bot has not
    handled yet. Six signals:

      1. RND novelty over (view, action)
             Global curiosity. The RND MLPs see the frame embedding with a
             one-hot of the action that was taken glued on, so the prediction
             error is "I have never seen this screen while pressing this
             button". A policy that sits still stops earning this the moment
             the idle screen becomes predictable, which is the whole point.
      2. Place novelty
             This coarse view has never been seen before, this whole run.
      3. Attempt novelty
             This action has not been tried in this view yet, this whole run.
             This is the signal that pays for asking "what does this button
             do here?", and it decays with repeats so it cannot be farmed by
             mashing one key in one spot.
      4. Controllability
             The frame moved more than it moves with no input at all - "my
             press visibly did something". Gated by (3), so a single
             repeatable effect stops paying once it is understood.
      5. Idle cost
             A small charge for any action that presses nothing, growing while
             the bot keeps idling. Doing nothing can never be the free option.
      6. Disruption
             A large, sudden change the bot did not cause (a cutscene, a
             respawn, a menu it did not open). Deliberately *not* charged to
             actions the bot took: pressing things must never be the risky
             choice.

    Novelty is scaled by a running standard deviation instead of being
    normalised to zero mean, so it stays a bonus that decays as the bot gets
    used to something rather than a penalty for familiar screens.

    There is no true episode reset in this environment (an "episode" is just a
    segment of a game that keeps running), so novelty memory persists across
    resets; only the previous frame and the idle streak are forgotten.
    """

    def __init__(
        self,
        obs_dim: int,
        device: torch.device,
        num_actions: int = 16,
        rnd_dim: int = RND_EMBED_DIM,
        rnd_lr: float = 1e-4,
        w_rnd: float = W_RND,
        w_place: float = W_NOVEL_STATE,
        w_attempt: float = W_NOVEL_PAIR,
        w_control: float = W_CONTROL,
        w_idle: float = W_IDLE,
        w_disruption: float = W_DISRUPTION,
        novelty_bins: int = 32,
    ):
        self.device = device
        self.num_actions = max(1, int(num_actions))
        self.frame_dim = int(obs_dim)
        self.w_rnd = w_rnd
        self.w_place = w_place
        self.w_attempt = w_attempt
        self.w_control = w_control
        self.w_idle = w_idle
        self.w_disruption = w_disruption
        self.novelty_bins = novelty_bins

        in_dim = self.frame_dim + self.num_actions

        def make_net():
            return nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, rnd_dim),
            )

        self.rnd_target = make_net().to(device)
        for p in self.rnd_target.parameters():
            p.requires_grad_(False)
        self.rnd_pred = make_net().to(device)
        self.rnd_opt = torch.optim.Adam(self.rnd_pred.parameters(), lr=rnd_lr)

        self.rnd_mean, self.rnd_var, self.rnd_count = 0.0, 1e-4, 1e-4

        self.visited = set()          # coarse views seen at all this run
        self.attempts: Dict[Tuple[str, int], int] = {}   # (view, action) -> tries
        self.prev_obs = None
        self.ambient = 0.0            # how much the frame moves with no input
        self.idle_streak = 0
        self.last: Dict[str, float] = {}   # breakdown of the most recent step

    def _update_stats(self, x, mean, var, count):
        count += 1
        delta = x - mean
        mean += delta / count
        var += delta * (x - mean)
        return mean, var, count

    def _scale(self, x, mean, var, count):
        """
        Divide a novelty bonus by the running standard deviation of that
        signal, so it fades as the bot gets used to something - this is what
        the RND paper does, and it is why the bonus stays comparable over a
        long run.

        The divisor is floored relative to the signal's own mean (and at an
        absolute minimum). Without that floor a signal that settles on a
        *constant* value has a standard deviation of zero and the bonus would
        race off toward infinity instead of becoming the harmless constant it
        should be.
        """
        std = max((var / count) ** 0.5, 0.25 * abs(mean), RND_MIN_SCALE)
        return x / std

    def _hash_obs(self, obs_np: np.ndarray) -> str:
        """
        Coarse fingerprint of a frame: the "situation" key for novelty.

        Deliberately blunt - 8x8 grayscale, six levels. A sharper hash would
        make almost every frame a brand new situation, which turns attempt
        novelty into a flat per-step bonus instead of a reason to try a
        *different* button, and place novelty into noise. This size keeps
        "same room, roughly same view" together while still separating the
        world, a menu and a loading screen.
        """
        gray = cv2.cvtColor(obs_np, cv2.COLOR_RGB2GRAY)
        coarse = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        q = (coarse.astype(np.int16) * 6 // 256).astype(np.uint8)
        return hashlib.md5(q.tobytes()).hexdigest()

    def _forget_oldest(self):
        """Keep novelty memory bounded without throwing all of it away."""
        for key in list(self.visited)[:len(self.visited) // 2]:
            self.visited.discard(key)
        for key in list(self.attempts)[:len(self.attempts) // 2]:
            self.attempts.pop(key, None)

    def reset_episode(self):
        # An episode boundary does not restart the game, so the novelty memory
        # deliberately survives it; only per-step state is cleared.
        self.prev_obs = None
        self.idle_streak = 0

    def note_human(self, obs_np: np.ndarray):
        """
        Note a step the bot did not take, because you had the controls.

        The frame is adopted as the new reference, so the first reward after the
        bot takes over does not charge you a huge unexplained disruption, and
        the idle streak is cleared rather than left to ramp up while you play.
        """
        self.prev_obs = obs_np.copy()
        self.idle_streak = 0

    # ---- checkpointing -----------------------------------------------------
    def state_dict(self) -> dict:
        """
        The parts of the reward worth carrying across a resume: the RND nets
        and their statistics, plus the estimate of how much the game moves on
        its own. The hash memories are per-run and are not saved.
        """
        return {
            "rnd_pred": self.rnd_pred.state_dict(),
            "rnd_target": self.rnd_target.state_dict(),
            "rnd_opt": self.rnd_opt.state_dict(),
            "rnd_stats": (self.rnd_mean, self.rnd_var, self.rnd_count),
            "ambient": self.ambient,
        }

    def load_state_dict(self, state: dict, strict: bool = False) -> bool:
        """Restore a state_dict(); returns False (and changes nothing) on a
        shape mismatch, so an incompatible file cannot half-load."""
        if not state:
            return False
        try:
            self.rnd_pred.load_state_dict(state["rnd_pred"])
            self.rnd_target.load_state_dict(state["rnd_target"])
            self.rnd_opt.load_state_dict(state["rnd_opt"])
            self.rnd_mean, self.rnd_var, self.rnd_count = state["rnd_stats"]
            self.ambient = float(state.get("ambient", self.ambient))
            return True
        except Exception as exc:
            if strict:
                raise
            print(f"[Reward] Intrinsic state not restored ({exc}); "
                  f"curiosity starts from scratch.")
            return False

    def compute(self, obs_embed: torch.Tensor, obs_np: np.ndarray,
                action: int, engaged: bool = True,
                episode_done: bool = False) -> float:
        """Reward one transition. `engaged` is True when the action actually
        presses/holds/clicks/turns something - the environment knows this and
        passes it in, so the reward never has to guess from the action index.
        `episode_done` is accepted for callers that report episode boundaries
        but deliberately changes nothing: see reset_episode()."""
        if obs_embed.dim() == 1:
            obs_embed = obs_embed.unsqueeze(0)
        index = int(action) % self.num_actions

        # 1. RND novelty, conditioned on the action that was taken.
        onehot = torch.zeros(1, self.num_actions, device=obs_embed.device,
                             dtype=obs_embed.dtype)
        onehot[0, index] = 1.0
        x = torch.cat([obs_embed, onehot], dim=1)
        with torch.no_grad():
            target = self.rnd_target(x)
        pred = self.rnd_pred(x)
        loss = F.mse_loss(pred, target)
        rnd_err = float(loss.detach())
        self.rnd_opt.zero_grad()
        loss.backward()
        self.rnd_opt.step()
        self.rnd_mean, self.rnd_var, self.rnd_count = self._update_stats(
            rnd_err, self.rnd_mean, self.rnd_var, self.rnd_count
        )
        rnd_bonus = min(
            self.w_rnd * self._scale(rnd_err, self.rnd_mean, self.rnd_var,
                                     self.rnd_count),
            RND_CAP,
        )

        # 2 + 3. Place novelty and attempt novelty, both persistent.
        view = self._hash_obs(obs_np)
        place_bonus = 0.0
        if view not in self.visited:
            self.visited.add(view)
            place_bonus = self.w_place * (
                1.0 / (1.0 + len(self.visited) / self.novelty_bins)
            )
        key = (view, index)
        tries = self.attempts.get(key, 0)
        self.attempts[key] = tries + 1
        attempt_bonus = self.w_attempt * (1.0 / (1.0 + tries))

        if len(self.attempts) > NOVELTY_MEMORY:
            self._forget_oldest()

        # How much the frame moved at all.
        delta = 0.0
        if self.prev_obs is not None:
            delta = float(np.abs(
                obs_np.astype(np.float32) - self.prev_obs.astype(np.float32)
            ).mean() / 255.0)

        # 4. Controllability: movement beyond what the game does by itself.
        # A fixed scale plus a cap keeps this bounded - a curiosity bonus that
        # grows with the size of the change would just buy a bot that shakes
        # the camera forever - and the per-attempt decay means a repeatable
        # effect stops paying once the bot has seen what it does.
        control_bonus = 0.0
        if engaged and self.prev_obs is not None:
            excess = max(0.0, delta - self.ambient)
            control_bonus = self.w_control * min(
                excess / CONTROL_SCALE, CONTROL_CAP
            ) / (1.0 + tries)

        # 5. Idle cost, growing while the bot keeps pressing nothing.
        idle_cost = 0.0
        if engaged:
            self.idle_streak = 0
        else:
            self.idle_streak += 1
            ramp = min(1.0 + self.idle_streak / max(IDLE_RAMP_STEPS, 1.0),
                       IDLE_RAMP_MAX)
            idle_cost = self.w_idle * ramp
            if self.prev_obs is not None:
                # Learn what "nothing happening" looks like from idle steps.
                self.ambient += 0.05 * (delta - self.ambient)

        # 6. Disruption the bot did not cause.
        disruption = 0.0
        if (not engaged) and self.prev_obs is not None \
                and delta > DISRUPTION_THRESHOLD:
            disruption = self.w_disruption

        self.prev_obs = obs_np.copy()
        self.last = {
            "rnd": rnd_bonus,
            "place": place_bonus,
            "attempt": attempt_bonus,
            "control": control_bonus,
            "idle": -idle_cost,
            "disruption": -disruption,
        }
        return float(rnd_bonus + place_bonus + attempt_bonus
                     + control_bonus - idle_cost - disruption)


# =============================================================================
# SECTION 6: GYMNASIUM ENVIRONMENT
# =============================================================================

class BackgroundGameEnv(gym.Env):
    """
    Wraps a background window as a Gymnasium environment.
    Observations: sequences of (visual frame, audio chunk).
    Actions: discrete, but the list is built from the calibrated keymap, so an
             action can hold buttons down, tap one, click and turn the view.
    Reward: fully game-agnostic intrinsic reward.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        hwnd: int,
        pid: int,
        device: torch.device,
        seq_len: int = 8,
        frame_size: int = IMG_SIZE,
        audio_samples: int = 16000,
        max_steps: int = 1000,
        rnd_embed_dim: int = 256,
        capture_delay: float = CAPTURE_DELAY,
        target_fps: float = TARGET_FPS,
        preview: bool = False,
        preview_scale: int = 3,
        audio_enabled: bool = True,
        audio_mode: str = "auto",
        keymap: Optional[Keymap] = None,
        action_set: Optional[List[dict]] = None,
        dry_run: bool = False,
    ):
        super().__init__()
        self.hwnd = hwnd
        self.device = device
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.audio_samples = audio_samples
        self.max_steps = max_steps
        self.capture_delay = max(0.0, float(capture_delay))
        self.target_fps = max(0.0, float(target_fps))
        self.preview = preview
        self.preview_scale = max(1, int(preview_scale))
        self._preview_ready = False

        # The keymap is the contract for what the bot may press; the action set
        # is the discrete list derived from it.
        self.keymap = keymap or Keymap.default()
        if action_set is None:
            action_set, capped = build_action_set(self.keymap)
            self.keymap.set_actions(action_set, capped)
        self.actions: List[dict] = list(action_set) or [
            {"id": 0, "label": "noop", "held": [], "uses": [], "tap": None,
             "click": None, "look": None}
        ]
        self.action_labels = [a.get("label", f"action{i}")
                              for i, a in enumerate(self.actions)]

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Dict({
            "visual": spaces.Box(0, 255, (seq_len, 3, frame_size, frame_size), np.uint8),
            "audio": spaces.Box(-1.0, 1.0, (seq_len, audio_samples), np.float32),
        })

        self.visual_buffer = deque(maxlen=seq_len)
        self.audio_buffer = deque(maxlen=seq_len)

        self.grabber = ScreenGrabber(hwnd)
        self.clock = FrameClock(self.target_fps, self.capture_delay)
        self.last_capture_ms = 0.0
        # dry_run keeps everything else identical (observations, reward, action
        # ids) while making the actual injection a no-op. Nothing in the current
        # design needs it - human takeovers use suspend_bot() instead - but it
        # stays as a way to exercise the environment without a game.
        self.dry_run = bool(dry_run)
        self.input = BackgroundInput(hwnd)
        self.input.set_keymap(self.keymap)
        if self.dry_run:
            print("[Input] DRY RUN: the bot will not send any input.")
        else:
            print(f"[Input] Real keyboard + mouse into the focused window "
                  f"({len(self.actions)} calibrated actions).")
        self._held: set = set()
        self._action_override: Optional[set] = None
        self._tracked_hold: set = set()
        # True while a human has the controls: nothing is injected, and the
        # observation advances without any action being taken.
        self._suspended = False

        self.audio_capture, self.audio_description = make_audio_capture(
            pid=pid,
            sample_rate=audio_samples,
            mode=audio_mode,
            enabled=audio_enabled,
        )
        print(f"[Audio] Source: {self.audio_description}")
        self.audio_capture.start()

        self.intrinsic = IntrinsicReward(
            obs_dim=rnd_embed_dim,
            device=device,
            num_actions=self.action_space.n,
        )

        self.step_count = 0

        # Frozen-screen watchdog: a paused or dead window produces identical
        # frames, which is not worth training on.
        self._prev_frame = None
        self._frozen_since = None
        self._frozen_warned = False

    # ---- helpers ----
    def _check_frozen(self, frame_rgb: np.ndarray):
        """Warn (once per freeze) when the grabbed frames stop changing."""
        if FROZEN_WARN_SECONDS <= 0:
            return
        if self._prev_frame is not None:
            diff = float(np.abs(
                frame_rgb.astype(np.int16) - self._prev_frame.astype(np.int16)
            ).mean())
            if diff < 0.5:  # effectively unchanged
                if self._frozen_since is None:
                    self._frozen_since = time.monotonic()
                elif (not self._frozen_warned
                      and time.monotonic() - self._frozen_since >= FROZEN_WARN_SECONDS):
                    self._frozen_warned = True
                    print()
                    print("!" * 72)
                    print(f"[Watchdog] The captured image has not changed for "
                          f"{FROZEN_WARN_SECONDS:.0f}s.")
                    print("[Watchdog] The game has almost certainly stopped")
                    print("[Watchdog] simulating or stopped rendering while")
                    print("[Watchdog] unfocused. Training on frozen frames is")
                    print("[Watchdog] wasted time.")
                    print("[Watchdog] Fix it in the game: turn off 'pause when")
                    print("[Watchdog] unfocused' / enable 'run in background',")
                    print("[Watchdog] or run the game borderless on a machine")
                    print("[Watchdog] you are not using. Check whether the window")
                    print("[Watchdog] is minimized, too.")
                    print("!" * 72)
                    print()
            else:
                self._frozen_since = None
                self._frozen_warned = False
        self._prev_frame = frame_rgb
    def _get_visual_frame(self) -> np.ndarray:
        started = time.perf_counter()
        img = self.grabber.grab()
        self.last_capture_ms = 1000.0 * (time.perf_counter() - started)
        if img is None:
            return np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.frame_size, self.frame_size),
                         interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _show_preview(self, frame_rgb: np.ndarray):
        """
        Optional live view of exactly what the model sees. The window is
        created with WINDOW_NORMAL and never asked for focus, so you can park
        it on a second monitor while you keep working.
        """
        if not self.preview:
            return
        try:
            big = cv2.resize(
                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                (self.frame_size * self.preview_scale,
                 self.frame_size * self.preview_scale),
                interpolation=cv2.INTER_NEAREST,
            )
            if not self._preview_ready:
                cv2.namedWindow("bot1 preview (what the model sees)", cv2.WINDOW_NORMAL)
                self._preview_ready = True
            cv2.imshow("bot1 preview (what the model sees)", big)
            cv2.waitKey(1)  # pump the GUI without blocking
        except Exception as exc:
            print(f"[Preview] Disabled after error: {exc}")
            self.preview = False

    def _get_audio_chunk(self) -> np.ndarray:
        audio = self.audio_capture.get_audio()
        if len(audio) != self.audio_samples:
            audio = np.interp(
                np.linspace(0, len(audio) - 1, self.audio_samples),
                np.arange(len(audio)),
                audio,
            )
        return audio.astype(np.float32)

    def _rnd_embedding(self, frame: np.ndarray) -> torch.Tensor:
        """Downsampled frame embedding fed to RND. Only the visual half - the
        reward appends a one-hot of the action to make it a state-action
        novelty signal."""
        small = cv2.resize(frame, (16, 16)).flatten().astype(np.float32) / 255.0
        # Pad or truncate to the frame dimension the reward net expects
        target_dim = self.intrinsic.frame_dim
        if len(small) < target_dim:
            small = np.pad(small, (0, target_dim - len(small)))
        else:
            small = small[:target_dim]
        return torch.tensor(small, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _get_obs(self):
        visual = np.stack(list(self.visual_buffer), axis=0).transpose(0, 3, 1, 2)
        audio = np.stack(list(self.audio_buffer), axis=0)
        return {"visual": visual, "audio": audio}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.intrinsic.reset_episode()
        # Never carry a held button across an episode boundary.
        self.release_all()
        self.clock.reset()

        self.visual_buffer.clear()
        self.audio_buffer.clear()
        for _ in range(self.seq_len):
            self.visual_buffer.append(self._get_visual_frame())
            self.audio_buffer.append(self._get_audio_chunk())

        return self._get_obs(), {}

    # ---- calibrated actions ----
    def _resolve_buttons(self, action_spec: dict,
                         override: Optional[set] = None) -> set:
        """
        Work out which buttons an action leaves held down.

        An action spec can name hold tokens directly, or use the TRACKED_HOLD
        token, which means "whatever the human was holding at this moment".
        That second form is what behavioral cloning supervises, and it is exact
        for recorded play.
        """
        buttons: set = set()
        if override is not None:
            return set(override)
        for token in list(action_spec.get("uses") or []) + \
                list(action_spec.get("held") or []):
            upper = str(token).upper()
            if upper == TRACKED_HOLD.upper():
                buttons |= set(self._tracked_hold)
                continue
            vk = self.keymap.hold_vk(upper)
            if vk:
                buttons.add(int(vk))
        return buttons

    def set_held(self, buttons: set):
        """
        Apply a button set, pressing what is new and releasing what is gone.

        This is the whole point of hold support: the bot presses a button once
        and it stays down across steps, so it can walk, sprint or swim
        continuously instead of re-tapping the key every 33 ms.

        In dry_run the bookkeeping still happens, but nothing is injected.
        """
        target = {int(vk) for vk in buttons if vk}
        if self._suspended:
            # The controls are yours. suspend_bot() already lifted everything,
            # and nothing new may go down until resume_bot().
            self._held = set()
            return
        if not self.dry_run:
            if target != self._held:
                # Input only lands in the focused window, so make sure of it
                # before any key goes down or comes up.
                self.input.begin_action()
            for vk in sorted(self._held - target):
                self.input.release_vk(vk)
            for vk in sorted(target - self._held):
                self.input.press_vk(vk)
        self._held = target

    def release_all(self):
        """Let go of everything - used on reset, pause and shutdown."""
        self.set_held(set())

    # ---- human takeover ----
    @property
    def held_vks(self) -> set:
        """The buttons the bot is holding right now, so the watcher can tell
        the bot's own held keys apart from yours. Taps and clicks in flight
        count too - the key is physically down even though it is not a hold."""
        return set(self._held) | set(self.input.transient_vks)

    @property
    def suspended(self) -> bool:
        return self._suspended

    def suspend_bot(self):
        """
        Hand the controls over: let go of everything, then inject nothing.

        The release happens first and on purpose - while injection is still
        allowed - so the game is not left believing a key is held down when you
        start playing.
        """
        if self._suspended:
            return
        self.release_all()
        self.input.suspended = True
        self._suspended = True

    def resume_bot(self):
        """Take the controls back. Nothing is pressed until the next action."""
        if not self._suspended:
            return
        self.input.suspended = False
        self._suspended = False

    def observe_human_step(self) -> np.ndarray:
        """
        Advance the observation while you are playing.

        The bot sends nothing, but it keeps the frame and audio buffers current
        (and paces itself at TARGET_FPS) so the policy is not looking at a stale
        frame the moment it takes the controls back. The intrinsic reward is
        told too, so the world changing under it is not charged to the bot as
        disruption or as an idle streak.
        """
        self.release_all()
        self.clock.tick()
        frame = self._get_visual_frame()
        self.audio_buffer.append(self._get_audio_chunk())
        self.visual_buffer.append(frame)
        self._show_preview(frame)
        self._check_frozen(frame)
        self.intrinsic.note_human(frame)
        self.step_count += 1
        return frame

    def _apply_action(self, action: int):
        """
        Apply one calibrated action for one control step.

        An action bundles whatever it needs: buttons held down for the step, an
        optional tap, an optional click, and an optional relative mouse turn.
        Mouse movement is a real cursor delta, which is the only thing the
        game's mouse-look responds to, so the window must be focused.
        """
        spec = self.actions[action] if 0 <= action < len(self.actions) else {}
        self.set_held(self._resolve_buttons(spec, override=self._action_override))
        if self.dry_run:
            return

        tap = spec.get("tap")
        if tap:
            self.input.tap_key(tap, ACTION_TAP_SECONDS)
        click = spec.get("click")
        if click:
            self.input.mouse_click(click, ACTION_TAP_SECONDS)
        look = spec.get("look")
        if look:
            self.input.mouse_move(int(look[0]), int(look[1]))

    def step(self, action, override_held: Optional[set] = None):
        self._action_override = override_held
        if not self.dry_run:
            self.input.begin_action()
        try:
            self._apply_action(int(action))
        finally:
            if not self.dry_run:
                self.input.end_action()
            self._action_override = None

        # "Did this action actually touch the game?" - read after applying it,
        # so a TRACKED_HOLD action counts as engaged exactly when it held
        # something. The reward uses this instead of guessing from the index.
        spec = self.actions[int(action)] if 0 <= int(action) < len(self.actions) else {}
        engaged = bool(self._held) or bool(spec.get("tap")) \
            or bool(spec.get("click")) or bool(spec.get("look"))

        self.clock.tick()

        new_frame = self._get_visual_frame()
        new_audio = self._get_audio_chunk()
        self.visual_buffer.append(new_frame)
        self.audio_buffer.append(new_audio)
        self._show_preview(new_frame)
        self._check_frozen(new_frame)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        # Game-agnostic intrinsic reward
        obs_embed = self._rnd_embedding(new_frame)
        reward = self.intrinsic.compute(
            obs_embed, new_frame, int(action), engaged=engaged,
            episode_done=(terminated or truncated),
        )

        info = {"capture_ms": self.last_capture_ms,
                "steps_per_second": self.clock.achieved_hz,
                "action_label": self.action_labels[int(action)]
                if 0 <= int(action) < len(self.action_labels) else "?",
                "engaged": engaged,
                "reward_parts": dict(self.intrinsic.last),
                "held": sorted(self._held)}
        perf = self.clock.report()
        if perf:
            print(perf)
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        try:
            self.release_all()
        except Exception:
            pass
        self.grabber.close()
        self.audio_capture.stop()
        if self.preview:
            try:
                cv2.destroyWindow("bot1 preview (what the model sees)")
            except Exception:
                pass
        super().close()


# =============================================================================
# SECTION 7: PPO TRAINING LOOP
# =============================================================================

# Single source of truth for the model shape, so checkpoints can be rebuilt
# exactly as they were saved. `seq_len`, `num_actions` and `frame_size` are not
# listed: they come from the live environment and are recorded into the
# checkpoint at save time. Keeping a second copy here is how the old build
# ended up building a seq_len=8 model for a seq_len=128 observation and dying
# on the first forward pass.
MODEL_CONFIG = {
    "visual_dim": VISUAL_DIM,
    "audio_dim": AUDIO_DIM,
    "hidden_dim": HIDDEN_DIM,
    "num_blocks": NUM_BLOCKS,
    "num_heads": NUM_HEADS,
    "ffn_hidden": FFN_HIDDEN,
    "dropout": DROPOUT,
    "pool": SEQ_POOL,
    "audio_downsample": AUDIO_DOWNSAMPLE,
}

# Bumped whenever the model architecture changes in a way that makes old
# weights unusable. Checkpoints record it, and load_checkpoint refuses anything
# that does not match rather than half-loading it.
ARCH_NAME = "swiglu-rope-transformer-v1"


def describe_model_shape(seq_len: int, num_actions: int, frame_size: int,
                         cfg: Optional[dict] = None) -> str:
    """One readable line describing what is about to be trained."""
    c = dict(cfg or MODEL_CONFIG)
    algo = "SwiGLU FFN"
    return (f"{c.get('num_blocks')}x transformer blocks "
            f"(d_model {c.get('visual_dim')}, {c.get('num_heads')} heads, "
            f"{algo}, RoPE, {c.get('pool')} pool) over "
            f"{seq_len} frames at {frame_size}px, {num_actions} actions")


def build_model(seq_len: int = SEQ_LEN, num_actions: int = 16,
                frame_size: int = IMG_SIZE,
                device: Optional[torch.device] = None,
                **overrides) -> MultiHeadSwiGLUTransformer:
    cfg = dict(MODEL_CONFIG)
    cfg.update(overrides)
    cfg.update(seq_len=seq_len, num_actions=num_actions, frame_size=frame_size)
    model = MultiHeadSwiGLUTransformer(**cfg)
    return model.to(device) if device is not None else model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def lr_scale_for_update(update_index: int) -> float:
    """
    Warmup then cosine decay, as a multiplier on the optimizer's base LR.

    Prodigy (and Adam) both benefit from this: a raw constant LR either moves
    too fast at the start, when the value function is still random and the
    advantages are mostly noise, or never settles at the end, when the novelty
    bonus has been mined out and the policy should be consolidating. The decay
    is what turns "it is technically still training" into "it is converging".
    """
    i = max(0, int(update_index))
    warmup = max(1, int(LR_WARMUP_UPDATES))
    if i < warmup:
        return float(i + 1) / float(warmup)
    total = max(warmup + 1, int(LR_DECAY_UPDATES))
    progress = min(1.0, float(i - warmup) / float(max(1, total - warmup)))
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    lo = float(LR_MIN_FRACTION)
    return lo + (1.0 - lo) * cosine


def build_optimizer(params, kind: str = OPTIMIZER_KIND, lr: float = OPTIMIZER_LR):
    """
    Central constructor so training and checkpoint-resume can never disagree
    about which optimizer a run uses.

    Prodigy is kept available because it needs no learning rate at all, but for
    a long CPU run Adam plus the warmup/cosine schedule below is the more
    predictable choice: Prodigy estimates its own step size, which means a
    decay schedule applied on top of it is fighting the estimator.
    """
    kind = str(kind or "adam").strip().lower()
    if kind == "prodigy":
        if Prodigy is None:
            raise RuntimeError("prodigyopt is not installed (pip install prodigyopt)")
        return Prodigy(params, lr=float(lr))
    return torch.optim.Adam(params, lr=float(lr), eps=1e-5)


def set_optimizer_lr(optimizer, base_lrs: List[float], scale: float) -> float:
    """Apply `scale` to every stored base LR; returns the first resulting LR."""
    first = 0.0
    for idx, group in enumerate(optimizer.param_groups):
        base = base_lrs[idx] if idx < len(base_lrs) else group.get("lr", 1.0)
        group["lr"] = float(base) * float(scale)
        if idx == 0:
            first = group["lr"]
    return first



def report_training_status(step: int, loss: float, alpha: float,
                           entropy: float, target_entropy: float,
                           action_counts: np.ndarray, reward_sums: Dict[str, float],
                           engaged_steps: int, window_steps: int,
                           env: BackgroundGameEnv,
                           perf_line: str = "") -> None:
    """
    One status block every few hundred steps, reporting the two things that
    actually answer "is it doing anything?": how often it pressed something,
    and which actions it spent the window on.

    A policy that has quietly collapsed onto "press nothing" is obvious here -
    the engaged share falls, one label owns the histogram, and the reward
    breakdown is nothing but idle cost. Without this you only find out by
    watching the game, which is exactly how a run can waste hours.

    `perf_line` adds the throughput split, which is what tells you whether the
    machine is keeping up or whether the model needs to get smaller.
    """
    print(f"[Step {step}] loss={loss:.4f} "
          f"entropy={entropy:.3f}/{target_entropy:.3f} alpha={alpha:.4f} "
          f"rnd_mean={env.intrinsic.rnd_mean:.4f}")
    if perf_line:
        print(f"          {perf_line}")
    total = int(action_counts.sum())
    if total <= 0 or window_steps <= 0:
        return

    labels = env.action_labels
    engaged_pct = 100.0 * engaged_steps / window_steps
    distinct = int((action_counts > 0).sum())
    share = 100.0 * float(action_counts.max()) / total
    print(f"          actions: {distinct}/{len(action_counts)} tried, "
          f"pressed something on {engaged_pct:.0f}% of steps, "
          f"most-used action {share:.0f}% of steps")

    top = np.argsort(action_counts)[::-1][:6]
    parts = [f"{labels[i] if i < len(labels) else i} "
             f"{100.0 * action_counts[i] / total:.0f}%"
             for i in top if action_counts[i] > 0]
    if parts:
        print("          top: " + ", ".join(parts))
    if reward_sums:
        order = ("rnd", "place", "attempt", "control", "idle", "disruption")
        parts = [f"{k}={reward_sums[k] / window_steps:+.4f}"
                 for k in order if k in reward_sums]
        if parts:
            print("          reward/step: " + ", ".join(parts))

    if engaged_pct < 25.0:
        print("          [WARN] the policy is mostly idling. Raise W_IDLE and "
              "W_NOVEL_PAIR, or raise ENTROPY_COEF_MAX (SECTION 0).")
    elif share > 90.0:
        print("          [WARN] one action owns almost every step. Raise "
              "ENTROPY_COEF_MAX so the entropy bonus can push it apart.")


def train_ppo(
    env: BackgroundGameEnv,
    total_steps: int = TOTAL_STEPS,
    batch_size: int = BATCH_SIZE,
    checkpoint_interval: float = CHECKPOINT_INTERVAL_SEC,
    keep_checkpoints: int = CHECKPOINT_KEEP,
    checkpoint_dir: str = CHECKPOINT_DIR,
    enable_hotkeys: bool = True,
    final_model_path: str = "background_swiglu_model.pt",
    initial_model: Optional[MultiHeadSwiGLUTransformer] = None,
    initial_optimizer=None,
    initial_step: int = 0,
    control: Optional[SignalController] = None,
    entropy_coef: float = ENTROPY_COEF,
    human: Optional[HumanWatcher] = None,
    minibatch_size: int = MINIBATCH_SIZE,
    epochs_per_update: int = EPOCHS_PER_UPDATE,
    update_seconds_budget: float = UPDATE_SECONDS_BUDGET,
) -> MultiHeadSwiGLUTransformer:
    """
    PPO training loop with three escape hatches:

      * global hotkeys (default F8 pause, F9 save now, F10 quit)
      * Ctrl+C / Ctrl+Break -> checkpoint then exit cleanly
      * automatic checkpoint every `checkpoint_interval` seconds, keeping only
        the newest `keep_checkpoints` files

    Runs forever when `total_steps` is "forever" (or unreachably large); the
    loop then only ends on a hotkey or an interrupt, and it always checkpoints
    on the way out so the next launch continues from there.

    Every checkpoint contains the model, optimizer, reward nets, step counter,
    rollout buffers and RNG state, so 'checkpoints/last.pt' is a resumable file.

    When a `human` watcher is passed, the loop also gives the game back to you
    the moment you touch the keyboard or mouse: it collects no transitions, it
    sends no input, it records what you did, and HUMAN_RELEASE_GRACE_SECONDS
    after your last input it resumes and takes one imitation step towards your
    behaviour after each of the next PPO updates.

    The entropy bonus keeps the policy from collapsing onto a single action.
    With ADAPTIVE_ENTROPY it is raised automatically whenever the policy stops
    exploring, and the periodic status line reports the action histogram and
    the reward breakdown, so a policy that has quietly learned to press nothing
    is visible in the log instead of only in the game.

    Two things here exist specifically so a slow machine never looks dead:

      * Each PPO epoch runs in `minibatch_size` chunks instead of one giant
        tensor, which keeps every backward pass small and cache-friendly.
      * `update_seconds_budget` caps the total time an update may spend
        looping. If it is hit, the update ends early and the loop goes back to
        collecting experience. Progress is therefore bounded per update, so
        the run always advances and always reaches its next checkpoint/status
        line.
    """
    total_steps = resolve_steps(total_steps)
    device = env.device
    print(f"[Training] Device: {device}")
    if total_steps >= FOREVER_STEPS:
        print("[Training] Step limit: none (forever). Stop with F10 or Ctrl+C; "
              "both save a checkpoint first.")

    if initial_model is not None:
        model = initial_model
    else:
        model = build_model(
            seq_len=env.seq_len,
            num_actions=env.action_space.n,
            frame_size=env.frame_size,
            device=device,
        )
    if initial_optimizer is not None:
        optimizer = initial_optimizer
    else:
        optimizer = build_optimizer(model.parameters())
    # The base LR of each param group, captured before any schedule touches it,
    # so warmup/decay is always applied to the true base instead of compounding.
    base_lrs = [float(g.get("lr", 1.0)) for g in optimizer.param_groups]

    gamma, lam, clip_eps = 0.99, 0.95, 0.2
    epochs_per_update = max(1, int(epochs_per_update))
    minibatch_size = max(1, int(minibatch_size))
    update_seconds_budget = float(update_seconds_budget)
    update_index = 0
    print(f"[Training] Model: {count_parameters(model):,} parameters - "
          f"{describe_model_shape(env.seq_len, env.action_space.n, env.frame_size)}")
    print(f"[Training] PPO: {batch_size} steps/update x {epochs_per_update} epoch(s) "
          f"in minibatches of <= {minibatch_size}"
          + (f", {update_seconds_budget:.0f}s budget per update"
             if update_seconds_budget > 0 else ", no update time budget")
          + f"; LR warmup {LR_WARMUP_UPDATES} then cosine decay to "
            f"{LR_MIN_FRACTION:.0%} over {LR_DECAY_UPDATES} updates.")
    opt_name = type(optimizer).__name__
    opt_lr = ", ".join(f"{g.get('lr', 0.0):.2e}" for g in optimizer.param_groups)
    print(f"[Training] Optimizer: {opt_name} (base lr {opt_lr})")

    # Exploration pressure. `alpha` is the live entropy coefficient; it starts
    # at ENTROPY_COEF and is nudged up while the policy's entropy sits below
    # the target, which is what keeps "press nothing" from becoming a stable
    # fixed point of the update rule.
    alpha = max(0.0, float(entropy_coef))
    alpha_base = alpha
    num_actions = max(1, int(env.action_space.n))
    target_entropy = ENTROPY_TARGET_FRAC * float(np.log(max(2, num_actions)))
    print(f"[Training] Exploration: entropy target {target_entropy:.3f} of "
          f"{float(np.log(max(2, num_actions))):.3f} max"
          + (", coefficient self-tuning" if ADAPTIVE_ENTROPY else "")
          + f" (start {alpha:.3f}, cap {ENTROPY_COEF_MAX:.3f}).")

    rollout = {k: [] for k in
               ["visual", "audio", "actions", "log_probs",
                "rewards", "values", "dones", "dones_any"]}

    # Rolling wall-clock diagnostics. `env` prints its own per-step cost; these
    # cover the *update* side, which is the half that silently eats a run on a
    # CPU. step_times holds recent seconds-per-environment-step, update_times
    # recent seconds-per-PPO-update, and env_share the fraction of wall clock
    # spent actually interacting with the game (the number that matters: if it
    # collapses, the model is too big for this machine).
    step_times: deque = deque(maxlen=200)
    update_times: deque = deque(maxlen=50)
    total_step_seconds = 0.0
    total_update_seconds = 0.0
    last_step_start = time.time()
    last_update_report = 0.0
    minibatches_skipped_by_budget = 0

    # Windowed diagnostics, reset every time the status line is printed.
    status_every = 500
    action_counts = np.zeros(num_actions, dtype=np.int64)
    reward_sums: Dict[str, float] = {}
    engaged_steps = 0
    window_steps = 0
    last_entropy = 0.0
    next_status = status_every

    checkpoint_manager = CheckpointManager(checkpoint_dir, keep=keep_checkpoints)
    if control is None:
        control = SignalController(interval_sec=checkpoint_interval,
                                   enable_hotkeys=enable_hotkeys)
        control.start()

    # The handover to you and back, plus the buffer of your play it fills.
    takeover = HumanTakeover(env, human) if human is not None else None
    imitation_steps = 0

    obs, _ = env.reset()
    episode_reward = 0.0
    step = 0
    session_start_step = initial_step
    last_loss = 0.0

    def build_payload():
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": {
                **MODEL_CONFIG, "seq_len": env.seq_len,
                "num_actions": env.action_space.n,
                "frame_size": env.frame_size,
                "arch": ARCH_NAME,
                "keymap": {
                    "path": env.keymap.path,
                    "origin": env.keymap.origin,
                    "actions": [a.get("label") for a in env.actions],
                },
            },
            "step": session_start_step + step,
            "episode_reward": episode_reward,
            "rollout": {k: list(v) for k, v in rollout.items()},
            "reward_version": REWARD_VERSION,
            "intrinsic": env.intrinsic.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "wall_clock": time.time(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        return payload

    def save_checkpoint(reason: str, verbose: bool = True):
        absolute_step = session_start_step + step
        path = checkpoint_manager.save(build_payload(), step=absolute_step,
                                       reason=reason)
        if verbose:
            if path:
                print(f"[Checkpoint] Saved '{reason}' at step {absolute_step} -> {path}")
                kept = checkpoint_manager.list_kept()
                print(f"[Checkpoint] Keeping {len(kept)} checkpoint(s) in "
                      f"{checkpoint_manager.directory} (limit {checkpoint_manager.keep})")
            else:
                print(f"[Checkpoint] Save failed at step {absolute_step}; "
                      f"training continues.")
        return path

    print()
    print("-" * 72)
    if control.hotkeys is not None and control.hotkeys.available:
        print(f"  Hotkeys: {control.hotkeys.describe()}  (pause / save / quit)")
    print("  Ctrl+C = quit and save a checkpoint")
    print(f"  Auto-checkpoint every {checkpoint_interval / 60:.1f} min, "
          f"keeping the latest {keep_checkpoints} in "
          f"'{checkpoint_manager.directory}'")
    print(f"  Resume later with: python bot1.py  (it picks up "
          f"'{os.path.basename(checkpoint_manager.last_path)}' automatically)")
    print("-" * 72)
    print()
    print("[Training] Starting...")

    interrupted = False
    try:
        while session_start_step + step < total_steps:
            # ---- control checks, at most once per step ----
            for message in control.service():
                print(f"[Control] {message}")

            if control.paused:
                if control.claim_pause_notice():
                    print("[Control] PAUSED. The bot sends no input while paused. "
                          "Press the pause key again to resume.")
                # Stay in the pause, but keep servicing a takeover: pressing F8
                # and then playing should still record what you did.
                while control.paused and not control.stop_requested:
                    for message in control.service():
                        print(f"[Control] {message}")
                    if takeover is not None:
                        takeover.service()
                    time.sleep(0.05)

            due = control.checkpoint_due()
            if due:
                save_checkpoint(due)

            if control.stop_requested:
                break

            # ---- are you taking the controls back? ----
            if takeover is not None:
                state = takeover.service()
                if state == "human":
                    continue
                if state == "resumed":
                    # The buffers kept up with your play, so make sure the
                    # policy acts on the current view rather than the one from
                    # before it handed the game over.
                    obs = env._get_obs()

            for _ in range(batch_size):
                if takeover is not None:
                    state = takeover.service()
                    if state == "human":
                        # Drop the partial batch: the world moved on without us,
                        # so the queued rollouts are no longer a chain.
                        for key in rollout:
                            rollout[key].clear()
                        break
                    if state == "resumed":
                        obs = env._get_obs()

                vis = torch.tensor(obs["visual"], dtype=torch.float32).unsqueeze(0).to(device)
                aud = torch.tensor(obs["audio"], dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits, value = model(vis, aud)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                next_obs, reward, terminated, truncated, info = env.step(action.item())
                episode_done = bool(terminated or truncated)

                # Wall-clock split between "interacting with the game" and
                # "training on what we saw". See the [Perf] report below.
                now = time.time()
                step_times.append(now - last_step_start)
                total_step_seconds += now - last_step_start
                last_step_start = now

                rollout["visual"].append(obs["visual"])
                rollout["audio"].append(obs["audio"])
                rollout["actions"].append(action.item())
                rollout["log_probs"].append(log_prob.item())
                rollout["rewards"].append(reward)
                rollout["values"].append(value.item())
                # GAE masks the current step only on true termination; a time
                # limit is bootstrapped instead of treated as a real end.
                rollout["dones"].append(float(terminated))
                rollout["dones_any"].append(float(episode_done))

                # Windowed diagnostics: which buttons are being pressed, and
                # which part of the reward is paying for it.
                action_counts[action.item()] += 1
                window_steps += 1
                if info.get("engaged"):
                    engaged_steps += 1
                for name, part in (info.get("reward_parts") or {}).items():
                    reward_sums[name] = reward_sums.get(name, 0.0) + float(part)

                obs = next_obs
                episode_reward += reward
                step += 1

                if episode_done:
                    print(f"[Step {step}] Episode reward: {episode_reward:.2f}")
                    # Reset at the top of the next iteration: resetting here
                    # would leave obs and the rollout buffer out of sync.
                    obs, _ = env.reset()
                    episode_reward = 0.0

                for message in control.service():
                    print(f"[Control] {message}")
                if control.paused or control.stop_requested:
                    break
                due = control.checkpoint_due()
                if due:
                    save_checkpoint(due)

            if control.stop_requested:
                break
            if not rollout["rewards"]:
                continue

            # ---- GAE ----
            rewards = np.array(rollout["rewards"])
            values = np.array(rollout["values"])
            dones = np.array(rollout["dones"], dtype=np.float32)
            dones_any = np.array(rollout["dones_any"], dtype=np.float32)

            last_value = 0.0
            if not bool(dones_any[-1]) and win32gui.IsWindow(env.hwnd):
                try:
                    with torch.no_grad():
                        boot_obs = env._get_obs()
                        _, boot_v = model(
                            torch.tensor(boot_obs["visual"], dtype=torch.float32)
                            .unsqueeze(0).to(device),
                            torch.tensor(boot_obs["audio"], dtype=torch.float32)
                            .unsqueeze(0).to(device),
                        )
                        last_value = float(boot_v.item())
                except Exception:
                    last_value = 0.0

            adv = np.zeros_like(rewards)
            last = 0.0
            for t in reversed(range(len(rewards))):
                next_v = values[t + 1] if t < len(rewards) - 1 else last_value
                delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
                adv[t] = last = delta + gamma * lam * (1 - dones[t]) * last
            returns = adv + values
            if len(adv) > 1 and adv.std() > 1e-8:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            else:
                adv = adv - adv.mean()

            vis_t = torch.tensor(np.array(rollout["visual"]), dtype=torch.float32).to(device)
            aud_t = torch.tensor(np.array(rollout["audio"]), dtype=torch.float32).to(device)
            act_t = torch.tensor(rollout["actions"], dtype=torch.long).to(device)
            old_logp = torch.tensor(rollout["log_probs"], dtype=torch.float32).to(device)
            adv_t = torch.tensor(adv, dtype=torch.float32).to(device)
            ret_t = torch.tensor(returns, dtype=torch.float32).to(device)

            n_samples = int(vis_t.shape[0])
            mb_size = max(1, min(minibatch_size, n_samples))

            # Learning-rate schedule (warmup, then cosine decay). Set before the
            # first minibatch of this update so the LR is never left at whatever
            # the previous update happened to end on.
            lr_now = set_optimizer_lr(optimizer, base_lrs, lr_scale_for_update(update_index))

            update_started = time.time()
            epochs_done = 0
            minibatches_done = 0
            budget_hit = False

            for _epoch in range(epochs_per_update):
                # Shuffle once per epoch so minibatches differ between passes.
                perm = torch.randperm(n_samples, device=device)
                stop_epoch = False
                for start in range(0, n_samples, mb_size):
                    idx = perm[start:start + mb_size]
                    if idx.numel() == 0:
                        continue

                    logits, vpred = model(vis_t[idx], aud_t[idx])
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(act_t[idx])

                    ratio = torch.exp(new_logp - old_logp[idx])
                    surr1 = ratio * adv_t[idx]
                    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(vpred, ret_t[idx])
                    entropy = dist.entropy().mean()
                    loss = policy_loss + 0.5 * value_loss - alpha * entropy

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), GRAD_CLIP_NORM)
                    optimizer.step()
                    minibatches_done += 1

                    # A non-finite loss poisons every weight on the next step,
                    # so catch it here, on the minibatch that produced it, and
                    # stop the update instead of writing NaNs into the model.
                    if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                        print(f"[Training] WARNING: non-finite "
                              f"{'loss' if not torch.isfinite(loss) else 'gradient'} "
                              f"at step {step}; ending this update early.")
                        budget_hit = True
                        stop_epoch = True
                        break

                    last_loss = float(loss.item())
                    last_entropy = float(entropy.item())

                    # The anti-stall guarantee: no single update may run longer
                    # than the budget. Experience collection resumes immediately
                    # and the next update picks up from a fresher policy.
                    if (update_seconds_budget > 0
                            and time.time() - update_started >= update_seconds_budget):
                        budget_hit = True
                        stop_epoch = True
                        break
                if stop_epoch:
                    break
                epochs_done += 1

            update_seconds = time.time() - update_started
            update_times.append(update_seconds)
            total_update_seconds += update_seconds
            update_index += 1
            if budget_hit:
                minibatches_skipped_by_budget += 1

            # Keep the entropy bonus honest: if the policy has stopped
            # exploring, buy exploration back rather than letting the run sit
            # on a single action for hours. It decays again once entropy is
            # healthy, so the bonus never dominates a working policy.
            if ADAPTIVE_ENTROPY:
                if last_entropy < target_entropy:
                    alpha = min(alpha * ENTROPY_COEF_STEP, ENTROPY_COEF_MAX)
                else:
                    alpha = max(alpha_base, alpha * ENTROPY_COEF_DECAY)

            for k in rollout:
                rollout[k].clear()

            # ---- learn from anything you demonstrated while driving ----
            # One supervised step towards "do what the human just did", on a
            # small class-weighted minibatch. Separate from the PPO epochs and
            # scaled by HUMAN_IMITATION_WEIGHT, so it nudges the policy towards
            # your play without taking the update over.
            if takeover is not None and takeover.replay.ready():
                try:
                    if takeover.replay.imitation_step(model, optimizer, device):
                        imitation_steps += 1
                        if imitation_steps % 50 == 1:
                            print(f"[Human] Imitation step {imitation_steps}: "
                                  f"loss={takeover.replay.last_loss:.4f} "
                                  f"match={100.0 * takeover.replay.last_match:.0f}% "
                                  f"over {len(takeover.replay)} recorded frame(s).")
                except Exception as exc:
                    print(f"[Human] Imitation step skipped: {exc}")

            if step >= next_status:
                next_status = step + status_every
                sec_per_step = (sum(step_times) / len(step_times)) if step_times else 0.0
                sec_per_update = (sum(update_times) / len(update_times)) if update_times else 0.0
                wall = total_step_seconds + total_update_seconds
                env_share = (100.0 * total_step_seconds / wall) if wall > 0 else 100.0
                budget_note = (f", {minibatches_skipped_by_budget} update(s) hit the "
                               f"{update_seconds_budget:.0f}s budget"
                               if minibatches_skipped_by_budget else "")
                perf_line = (f"perf: {1.0 / sec_per_step:.1f} env steps/s "
                             f"({sec_per_step * 1000:.0f} ms/step), "
                             f"{sec_per_update:.1f} s/PPO update, "
                             f"{env_share:.0f}% of wall clock in the game"
                             f"{budget_note}")
                report_training_status(
                    step, last_loss, alpha, last_entropy, target_entropy,
                    action_counts, reward_sums, engaged_steps, window_steps,
                    env,
                    perf_line=perf_line,
                )
                if env_share < 50.0:
                    print("          [WARN] most of the wall clock is going into "
                          "PPO updates, so the bot is reacting to stale frames. "
                          "Lower SEQ_LEN first (it is the dominant cost), then "
                          "IMG_SIZE, or raise MINIBATCH_SIZE "
                          "for better cache use (SECTION 0).")
                if sec_per_update > max(5.0, update_seconds_budget or 0.0):
                    print("          [WARN] updates are slow for this machine. "
                          "SEQ_LEN and the audio downsampling factor are the "
                          "biggest levers; IMG_SIZE matters much less than it "
                          "looks (SECTION 0).")
                action_counts[:] = 0
                reward_sums.clear()
                engaged_steps = 0
                window_steps = 0

            # A NaN/Inf blow-up is easy to miss in a background run, and every
            # later checkpoint would be worthless, so catch it early.
            bad = model_has_nonfinite(model)
            if bad:
                print(f"[Training] WARNING: non-finite weights in {bad[:3]}"
                      f"{'...' if len(bad) > 3 else ''}.")
                print("[Training] Loading the newest healthy checkpoint and stopping.")
                save_checkpoint("before_nan_abort", verbose=False)
                break

    except KeyboardInterrupt:
        interrupted = True
        control.stop_reason = "ctrl_c"
        print("\n[Training] Hard interrupt - saving what we have.")
    finally:
        for message in control.service():
            print(f"[Control] {message}")

    # ---- always leave a usable checkpoint behind ----
    if control.stop_reason == "hotkey":
        print("[Training] Quit hotkey. Saving final checkpoint...")
    elif control.stop_reason == "ctrl_c" or interrupted:
        print("[Training] Interrupted. Saving final checkpoint...")
    elif total_steps < FOREVER_STEPS and session_start_step + step >= total_steps:
        print(f"[Training] Reached the {total_steps:,}-step target. "
              f"Saving final checkpoint...")
    else:
        print("[Training] Stopping early. Saving final checkpoint...")

    save_checkpoint("final")

    print(f"[Done] Training steps this run: {step} "
          f"(total {session_start_step + step})")
    print(f"[Done] Checkpoints: {checkpoint_manager.directory}")
    for entry in checkpoint_manager.list_kept():
        print(f"       step {entry['step']:>9}  ({entry['reason']})  {entry['path']}")
    print(f"[Done] Rolling resume file: {checkpoint_manager.last_path}")
    if takeover is not None and takeover.takeovers:
        print(f"[Done] You took the controls {takeover.takeovers} time(s); the "
              f"imitation buffer ended with {len(takeover.replay)} frame(s) and "
              f"was used in {imitation_steps} update(s).")

    bad = model_has_nonfinite(model)
    if bad:
        print(f"[WARN] Model contains non-finite weights ({bad[:3]}); "
              f"NOT writing {final_model_path}.")
    else:
        if atomic_torch_save(model.state_dict(), final_model_path):
            print(f"[OK] Final weights saved to {final_model_path}")
    return model


def load_checkpoint(path: str, device: torch.device,
                    expect_actions: Optional[int] = None,
                    expect_frame_size: Optional[int] = None) -> Optional[dict]:
    """
    Load a checkpoint file and rebuild the model, optimizer and reward nets.
    Returns a dict with the restored objects, or None if the file is unusable.

    A checkpoint is only reusable if it was trained with the same action space,
    the same frame size and the same reward; the action space comes from the
    calibrated keymap, so re-running --calibrate changes it. Any mismatch is
    reported as a readable message instead of a torch shape error, and the run
    starts fresh rather than half-resuming.
    """
    if not os.path.exists(path):
        print(f"[Resume] No checkpoint at '{path}'.")
        return None

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        print(f"[Resume] Could not read '{path}': {exc}")
        return None

    saved_reward = int(ckpt.get("reward_version", 1))
    if saved_reward != REWARD_VERSION:
        print(f"[Resume] '{os.path.basename(path)}' was trained with reward "
              f"version {saved_reward} but this build uses {REWARD_VERSION}.")
        print("[Resume] The objective changed, so that policy was chasing a "
              "different thing (and its curiosity nets no longer match). "
              "Starting fresh - delete the checkpoint directory if you want "
              "the old files gone.")
        return None

    cfg = dict(ckpt.get("model_config") or MODEL_CONFIG)
    cfg.pop("keymap", None)          # metadata, not a model argument

    # Architecture gate. Checkpoints written before the SwiGLU/RoPE rebuild
    # have no "arch" key at all, and one of those must never be partially
    # loaded into the new model - the state_dict errors that produces are
    # unreadable, and a half-loaded model trains on garbage.
    saved_arch = cfg.pop("arch", None)
    if saved_arch != ARCH_NAME:
        print(f"[Resume] '{os.path.basename(path)}' was trained by an older "
              f"model ({saved_arch or 'pre-SwiGLU gMLP build'}); this build "
              f"trains {ARCH_NAME}.")
        print("[Resume] Weights cannot be reused across architectures, so this "
              "run starts fresh. Delete the checkpoint directory to stop being "
              "reminded.")
        return None

    if expect_actions is not None:
        saved_actions = int(cfg.get("num_actions") or MODEL_CONFIG["num_actions"])
        if saved_actions != expect_actions:
            print(f"[Resume] '{os.path.basename(path)}' was trained with "
                  f"{saved_actions} actions but this keymap defines "
                  f"{expect_actions}.")
            print("[Resume] The action space follows the calibrated keymap, so "
                  "this checkpoint cannot be reused. Starting fresh.")
            return None
    if expect_frame_size is not None:
        saved_size = int(cfg.get("frame_size") or MODEL_CONFIG["frame_size"])
        if saved_size != expect_frame_size:
            print(f"[Resume] '{os.path.basename(path)}' used {saved_size}px "
                  f"frames but IMG_SIZE is {expect_frame_size}; starting fresh "
                  f"(set IMG_SIZE = {saved_size} to resume it).")
            return None

    try:
        model = MultiHeadSwiGLUTransformer(**cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as exc:
        print(f"[Resume] Model mismatch in '{path}': {exc}")
        return None

    optimizer = build_optimizer(model.parameters())
    try:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception as exc:
        print(f"[Resume] Optimizer state skipped: {exc}")

    print(f"[Resume] Loaded '{path}' at step {ckpt.get('step', '?')}")
    return {"checkpoint": ckpt, "model": model, "optimizer": optimizer}


# =============================================================================
# SECTION 8: MAIN
# =============================================================================

def resolve_steps(value, default: int = FOREVER_STEPS) -> int:
    """
    Turn a step target into a number the training loop can compare against.

    Accepts an int (a finite target, where 0 or negative also means forever), or
    the string "forever" (also "inf", "infinite", "none", "always"). "Forever"
    becomes FOREVER_STEPS, an unreachable count, so the loop keeps resuming,
    checkpointing and reporting just like a normal long run.
    """
    if value is None:
        return int(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in (FOREVER, "inf", "infinite", "infinity", "none", "always",
                    "unlimited"):
            return FOREVER_STEPS
        try:
            value = int(float(text))
        except (TypeError, ValueError):
            print(f"[Steps] Unrecognised TOTAL_STEPS value {value!r}; training "
                  f"forever instead.")
            return FOREVER_STEPS
    try:
        steps = int(value)
    except (TypeError, ValueError):
        return FOREVER_STEPS
    if steps <= 0:
        return FOREVER_STEPS
    return min(steps, FOREVER_STEPS)


def describe_steps(total_steps: int) -> str:
    """'forever' when the target is unreachable, else the plain number."""
    if total_steps >= FOREVER_STEPS:
        return "forever"
    return f"{total_steps:,}"


def lower_process_priority(level: str) -> None:
    """
    Keep frame capture from fighting the apps you are actually using.
    Windows maps below-normal priority to a lower CPU scheduling class, so the
    training loop yields automatically whenever something else wants the CPU.
    """
    try:
        import psutil
        proc = psutil.Process()
        if level == "below_normal":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print("[Priority] Torch/capture threads set to BELOW_NORMAL.")
        elif level == "high":
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
            print("[Priority] Process set to HIGH (may make the desktop sluggish).")
        else:
            print("[Priority] Left at NORMAL.")
    except ImportError:
        if level == "below_normal":
            print("[Priority] 'psutil' not installed - priority unchanged.")
            print("           Install with: python -m pip install psutil")
    except Exception as exc:
        print(f"[Priority] Could not change priority: {exc}")


def hotkey_vks(*specs: str) -> set:
    """Virtual keys the global hotkeys use, so they are never read as yours."""
    if not specs:
        specs = (HOTKEY_PAUSE, HOTKEY_SAVE, HOTKEY_QUIT)
    vks = set()
    for spec in specs:
        if not spec:
            continue
        try:
            _modifiers, vk = parse_hotkey(spec)
        except ValueError:
            continue
        vks.add(int(vk))
    return vks


def build_parser() -> argparse.ArgumentParser:
    """
    The command line, which is now exactly one option.

    Everything else the program can be told to do is a constant in SECTION 0.
    The flags that used to exist only ever overrode those constants for a single
    run, so they were removed: the constants are the single source of truth.
    """
    parser = argparse.ArgumentParser(
        prog="bot1.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Train a SwiGLU + RoPE transformer agent to play a windowed game\n"
            "from game-agnostic intrinsic rewards, and let it watch you\n"
            "whenever you take the controls back.\n\n"
            "  python bot1.py              train (and hand the controls to you\n"
            "                              whenever you touch the keyboard or\n"
            "                              mouse, then take them back 3s later)\n"
            "  python bot1.py --calibrate  teach it which keys it may press\n\n"
            "Every other setting is a constant in SECTION 0 of the script."),
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="learn which keys and mouse buttons the bot may use: pick the "
             "window, press the toggle key (F8), play, press it again, and a "
             "keymap JSON is written")
    return parser


def build_config(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Settle every run setting into one namespace.

    Only --calibrate comes from the command line; all of these values are the
    SECTION 0 constants, gathered here so the rest of the program keeps reading
    a single object.
    """
    parsed = build_parser().parse_args(argv)
    return argparse.Namespace(
        calibrate=bool(parsed.calibrate),
        # ---- keymap ----
        keymap=KEYMAP_PATH,
        calibrate_key=CALIBRATE_KEY,
        calibrate_min_hold=CALIBRATE_MIN_HOLD,
        fresh_keymap=CALIBRATE_FRESH_KEYMAP,
        # ---- window selection ----
        window=TARGET_WINDOW_HWND,
        select=False,
        pick=False,
        auto_window=False,
        no_auto_window=not PREFER_GAME_WINDOW,
        # ---- training / machine load ----
        steps=TOTAL_STEPS,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        seq_len=SEQ_LEN,
        capture_delay=CAPTURE_DELAY,
        target_fps=TARGET_FPS,
        device=DEVICE,
        preview=SHOW_PREVIEW,
        priority=PRIORITY,
        no_tests=not RUN_STARTUP_TESTS,
        # ---- checkpoints and global control ----
        checkpoint_interval=CHECKPOINT_INTERVAL_SEC,
        keep_checkpoints=CHECKPOINT_KEEP,
        checkpoint_dir=CHECKPOINT_DIR,
        pause_key=HOTKEY_PAUSE,
        save_key=HOTKEY_SAVE,
        quit_key=HOTKEY_QUIT,
        no_hotkeys=not ENABLE_HOTKEYS,
        resume="auto" if RESUME_ON_START else None,
        no_resume=not RESUME_ON_START,
        # ---- audio ----
        audio_mode=AUDIO_MODE,
        no_audio=(str(AUDIO_MODE).lower() == "off"),
        # ---- one-shot diagnostics (set DIAGNOSTIC in SECTION 0) ----
        diagnostic=str(DIAGNOSTIC or "").strip().lower(),
        diagnostic_seconds=DIAGNOSTIC_SECONDS,
        diagnostic_key=DIAG_INPUT_KEY,
    )


def run_diagnostic(args) -> int:
    """
    The old diagnostic flags, now driven by the DIAGNOSTIC constant.

    "windows" and "actions" need no window; the rest pick one first, exactly
    like a training run does.
    """
    kind = args.diagnostic
    if kind in ("windows", "list-windows", "list_windows"):
        windows = enumerate_windows(min_area=1)
        print(format_window_table(windows))
        print(f"\n{len(windows)} windows. Put the one you want in "
              f"TARGET_WINDOW_HWND in SECTION 0 to skip the picker.")
        return 0
    if kind in ("actions", "print-actions", "keymap", "show-keymap"):
        keymap, actions = build_action_set_for_args(args)
        print()
        print(f"Keymap   : {keymap.path or '(built-in defaults)'}")
        print(f"Origin   : {keymap.origin}")
        print(f"Keys     : {keymap.describe_keys()}")
        print(f"Mouse    : turn {keymap.default_mouse_turn} px per action")
        print(f"Actions  : {len(actions)}"
              + (" (capped at MAX_ACTIONS)" if keymap.actions_capped else ""))
        print("-" * 72)
        for action in actions:
            print(f"{action['id']:>4}  {action.get('label', '?'):<34} "
                  f"{describe_action(action, keymap)}")
        print("-" * 72)
        return 0

    known = ("capture", "input", "safety", "release")
    if kind not in known:
        print(f"[DIAGNOSTIC] Unrecognised value {kind!r}. Use one of "
              f"{', '.join(known + ('windows', 'actions'))}, or \"\" for none.")
        return 2

    target = choose_window(args, force_picker=False)
    if target is None:
        print("[Cancelled] No target window selected.")
        return 1
    if kind == "capture":
        return diag_capture(target["hwnd"], float(args.diagnostic_seconds))
    if kind == "input":
        return diag_input(target["hwnd"], args.diagnostic_key)
    if kind == "safety":
        return diag_watch_safety(target["hwnd"])
    return release_stuck_keys(target["hwnd"])


def diag_input(hwnd: int, key: str = "E") -> int:
    """
    Test whether the game actually reacts to injected input.

    Taps `key` (the inventory key in most survival games), watches for the big
    frame change that opening a menu causes, then taps Escape to close it again.
    This is the quickest way to confirm the whole input path works on a new
    machine; set DIAG_INPUT_KEY to a key your game reacts to. Reachable with
    DIAGNOSTIC = "input".
    """
    print(f"[input-diag] window: {win32gui.GetWindowText(hwnd)!r}")
    print(f"[input-diag] hwnd={hwnd} foreground="
          f"{win32gui.GetForegroundWindow() == hwnd} "
          f"minimized={is_minimized(hwnd)}")

    inp = BackgroundInput(hwnd)
    inp.begin_action()
    focus_ok = inp._focused
    print(f"[input-diag] focus acquired: {focus_ok}")
    if not focus_ok:
        print("[input-diag] The game window is not focused, so SendInput has")
        print("[input-diag] nowhere to land. Click the game once (or run this")
        print("[input-diag] on the machine that owns the desktop) and retry.")

    print(f"[input-diag] Pressing {key.upper()} should cause a noticeable change.")
    before = capture_window_printwindow(hwnd)
    if before is None:
        print("[input-diag] capture failed; cannot test.")
        inp.end_action()
        return 1

    inp.tap_key(key, 0.1)
    sent = inp.last_send_count
    err = inp.last_send_error
    time.sleep(1.4)
    after = capture_window_printwindow(hwnd)
    inp.end_action()
    if after is None:
        print("[input-diag] capture failed after the key press.")
        return 1

    print(f"[input-diag] SendInput reported events injected: {sent}"
          + (f" (error {err})" if err else ""))
    d = np.abs(after.astype(np.int16) - before.astype(np.int16))
    fraction = int((d > 8).sum()) / max(1, d.size)
    print(f"[input-diag] mean delta={d.mean():6.2f}  "
          f"changed={fraction * 100:5.1f}% of pixels")

    if fraction > 0.02:
        print("[input-diag] RESULT: INPUT WORKS. Pressing Escape to undo it.")
        inp.begin_action()
        inp.tap_key('ESC', 0.1)
        inp.end_action()
        time.sleep(0.9)
        return 0

    print("[input-diag] RESULT: NO REACTION TO THE KEY. Check, in order:")
    print("[input-diag]   1. The game window must be focused and in an active")
    print("[input-diag]      state (not a menu or a loading screen).")
    print("[input-diag]   2. Set DIAGNOSTIC = \"capture\": if the image is")
    print("[input-diag]      frozen, a real change could not show up.")
    print(f"[input-diag]   3. {key.upper()} may not do anything visible in this")
    print("[input-diag]      game - set DIAG_INPUT_KEY to one that does.")
    return 1


def release_stuck_keys(hwnd: int) -> int:
    """
    Let go of every key and mouse button this bot could ever have pressed.

    If a previous run was killed while holding a button - and especially if the
    game still had focus - the OS can be left believing that button is down, so
    the game keeps moving on its own even though no bot process is running.
    This sends the matching key-up/button-up for the whole calibrated whitelist
    plus the built-in defaults, which clears that state.

    Safe to run any time; it only releases.
    """
    keymap = Keymap.default()
    loaded = Keymap.load(KEYMAP_PATH)
    if loaded is not None:
        keymap = loaded
        for name in list(loaded.keys):
            keymap.allow(name, source="release")
        for name in list(loaded.mouse_buttons):
            keymap.allow(name, source="release")

    vks = set()
    for name in list(keymap.keys) + list(DEFAULT_HOLD_KEYS) + list(DEFAULT_TAP_KEYS):
        vk = keymap.vk(name) or calibratable_keys().get(str(name).upper())
        if vk:
            vks.add(int(vk))
    for name in list(keymap.mouse_buttons) + list(_MOUSE_BUTTON_NAMES.values()):
        vk = keymap.button_vk(name)
        if vk:
            vks.add(int(vk))
    # Also the generic modifier codes, in case a game saw those instead.
    vks.update({0x10, 0x11, 0x12, 0x5B, 0x5C})
    vks.discard(0)

    print(f"[release] Releasing {len(vks)} possible button(s) into "
          f"hwnd={hwnd} ({win32gui.GetWindowText(hwnd)!r}).")
    inp = BackgroundInput(hwnd, keymap)
    if not inp._ensure_focus():
        print("[release] Warning: could not focus the target window, so the "
              "key-up events may not reach the game.")
        print("[release] Click the game and run this again.")
    for vk in sorted(vks):
        inp.release_vk(vk)
    print("[release] Done. Every key the bot knows about has been lifted.")
    return 0


def diag_watch_safety(hwnd: int) -> int:
    """
    Prove that calibration cannot touch the game.

    Run this while you are in a game (or on the desktop) to check that this
    build refuses to inject input when it is supposed to. It holds the same
    injection lock --calibrate holds, deliberately tries to press keys, clicks
    and mouse movements, and reports that every one of them was refused. If
    this passes, nothing you do during calibration can be fighting the bot.
    Reachable with DIAGNOSTIC = "safety".
    """
    print("[watch-safety] Checking that input injection is refused...")
    attempts = [
        ("press W", lambda i: i.press_key("W")),
        ("tap SPACE", lambda i: i.tap_key("SPACE", 0.0)),
        ("hold LSHIFT", lambda i: i.press_vk(0xA0)),
        ("release LSHIFT", lambda i: i.release_vk(0xA0)),
        ("click left", lambda i: i.mouse_click("left", 0.0)),
        ("turn view", lambda i: i.mouse_move(80, 0)),
    ]
    input_iface = BackgroundInput(hwnd)
    inside = 0
    with InjectionLock("the safety check"):
        before = _INJECTION_BLOCKED_TOTAL[0]
        for label, call in attempts:
            try:
                call(input_iface)
            except Exception as exc:
                print(f"[watch-safety] {label}: raised {exc}")
        inside = _INJECTION_BLOCKED_TOTAL[0] - before
    # Each call may produce more than one SendInput (a tap is a down and an up,
    # a click likewise), so compare against refused calls, not labels.
    refused = input_iface.blocked_count
    print(f"[watch-safety] {inside} refused injection call(s) across "
          f"{len(attempts)} attempted behaviours.")
    after = _INJECTION_BLOCKED_TOTAL[0]
    # Outside the lock a real injection is allowed again, so do not actually
    # perform one - just check that the switch flipped back.
    from_switch = INPUT_INJECTION_ENABLED
    print(f"[watch-safety] injection re-enabled after the lock: {from_switch}")
    if inside > 0 and from_switch:
        print("[watch-safety] RESULT: SAFE - calibration cannot send input.")
        return 0
    print("[watch-safety] RESULT: UNSAFE - input got through. Please report this "
          "with the numbers above.")
    return 1


def diag_capture(hwnd: int, seconds: float = 5.0) -> int:
    """
    Report whether the target window's captured pixels actually update.

    This is the fastest way to tell a working setup from one where the game
    is paused, minimized, or not updating its framebuffer while unfocused.
    """
    print(f"[diag] grabbing hwnd={hwnd} for {seconds:.0f}s...")
    print(f"[diag] window: {win32gui.GetWindowText(hwnd)!r} "
          f"(foreground={win32gui.GetForegroundWindow() == hwnd}, "
          f"minimized={is_minimized(hwnd)})")

    prev = None
    frames = 0
    changed = 0
    deltas = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        frame = capture_window_printwindow(hwnd)
        if frame is None:
            print("[diag] capture returned None")
            return 1
        if prev is not None:
            delta = float(np.abs(frame.astype(np.int16)
                                 - prev.astype(np.int16)).mean())
            deltas.append(delta)
            if delta > 0.5:
                changed += 1
        prev = frame
        frames += 1
        time.sleep(0.06)

    mean = float(prev.mean()) if prev is not None else 0.0
    print(f"[diag] {frames} grabs, {changed} differed from the previous one")
    if deltas:
        print(f"[diag] mean abs diff per grab: "
              f"min={min(deltas):.3f} avg={sum(deltas) / len(deltas):.3f} "
              f"max={max(deltas):.3f}")
    print(f"[diag] final frame mean brightness: {mean:.1f}")

    if frames < 3:
        print("[diag] Not enough grabs; capture is failing.")
        return 1
    if changed == 0 and mean < 1.0:
        print("[diag] RESULT: BLACK / EMPTY - the window is not rendering "
              "anything the grabber can see.")
        return 1
    if changed == 0:
        print("[diag] RESULT: FROZEN - pixels never changed. The game has almost")
        print("[diag]         certainly stopped simulating or stopped rendering")
        print("[diag]         while unfocused. Fix it in the game (turn off pause")
        print("[diag]         on lost focus / enable run-in-background) or use a")
        print("[diag]         window mode that keeps rendering, then run this again.")
        return 1
    ratio = changed / max(1, frames - 1)
    print(f"[diag] RESULT: LIVE - pixels changed in {ratio * 100:.0f}% of grabs. "
          f"Capture is usable.")
    return 0


def _window_info(hwnd: int) -> dict:
    pid = get_window_pid(hwnd)
    return {"hwnd": hwnd, "title": win32gui.GetWindowText(hwnd), "pid": pid,
            "exe": get_process_name(pid) or "?", "area": get_client_area(hwnd),
            "minimized": is_minimized(hwnd)}


def choose_window(args, force_picker: bool = False) -> Optional[dict]:
    """
    Resolve the target window. No game is special-cased; any window works.

    Order: the pinned TARGET_WINDOW_HWND, then (when PREFER_GAME_WINDOW) the
    largest window that merely *looks* like a game, otherwise the interactive
    picker. The heuristic only decides what to try first - if your game is not
    recognised, pick it from the list and everything works the same.

    force_picker (used by --calibrate) always shows the numbered window list
    first, so calibration can be pointed at exactly the window you play in
    instead of whatever was guessed.
    """
    want_picker = force_picker or getattr(args, "select", False) \
        or getattr(args, "pick", False) \
        or getattr(args, "no_auto_window", False)

    if args.window is not None and not force_picker:
        hwnd = args.window
        if not win32gui.IsWindow(hwnd):
            print(f"[ERROR] hwnd {hwnd} is not a valid window.")
            return None
        print(f"[picker] Using TARGET_WINDOW_HWND {hwnd} "
              f"({win32gui.GetWindowText(hwnd)!r}).")
        return _window_info(hwnd)

    if want_picker:
        print("[picker] Choose the window this run should use.")
        chosen = pick_window_interactive()
        if chosen is None and args.window is not None:
            print(f"[picker] Nothing chosen; falling back to "
                  f"TARGET_WINDOW_HWND {args.window}.")
            return _window_info(args.window)
        return chosen

    prefer_auto = (PREFER_GAME_WINDOW or getattr(args, "auto_window", False)) \
        and not getattr(args, "no_auto_window", False)
    if prefer_auto:
        hwnd = find_game_window()
        if hwnd is not None:
            return _window_info(hwnd)

    return pick_window_interactive()


def main(argv: Optional[List[str]] = None) -> int:
    args = build_config(argv)

    print("=" * 72)
    print("Background RL - Proof of Concept")
    print("=" * 72)

    if sys.platform != "win32":
        print("[ERROR] This script relies on Win32 APIs (PrintWindow, "
              "RegisterHotKey, SendInput) and only runs on Windows.")
        return 2

    if args.diagnostic:
        return run_diagnostic(args)

    if args.calibrate:
        # Calibration records real input, so always let the user pick the
        # window from the list rather than trusting auto-detection: what you
        # calibrate must be the window you actually play in.
        print("[picker] Calibration records real input, so the target window "
              "is picked from the list.")
        target = choose_window(args, force_picker=True)
        if target is None:
            print("[Cancelled] No target window selected.")
            return 1
        return run_calibration(target, args)

    target = choose_window(args, force_picker=False)
    if target is None:
        print("[Cancelled] No target window selected.")
        return 1

    hwnd, pid = target["hwnd"], target["pid"]
    if not win32gui.IsWindow(hwnd):
        print(f"[ERROR] hwnd {hwnd} disappeared before training started.")
        return 1
    print(f"[OK] Window: hwnd={hwnd}, pid={pid}, title='{target['title']}'")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    lower_process_priority(args.priority)

    audio_mode = "off" if args.no_audio else args.audio_mode

    # On a dedicated machine the game owns the foreground. Say so up front,
    # because everything downstream assumes it.
    if win32gui.GetForegroundWindow() != hwnd:
        print("[Input] Note: the target window is not focused. The bot will "
              "try to focus it; if you are using this machine, input will not "
              "land reliably.")
    else:
        print("[Input] Target window is focused - real input will land.")

    # Everything the bot may press comes from the calibrated keymap; the action
    # list is derived from it.
    keymap, actions = build_action_set_for_args(args)

    env = BackgroundGameEnv(
        hwnd, pid, device,
        seq_len=args.seq_len,
        frame_size=args.img_size,
        capture_delay=args.capture_delay,
        target_fps=args.target_fps,
        preview=args.preview,
        audio_enabled=(audio_mode != "off"),
        audio_mode=audio_mode,
        keymap=keymap,
        action_set=actions,
    )
    if audio_mode == "off":
        print("[Audio] Off; audio observations are silence.")

    # Say how fast it can actually see and act - this is the reaction time.
    print(f"[Perf] Target {args.target_fps:.0f} steps/s at "
          f"{args.img_size}x{args.img_size}px"
          + (f", plus {args.capture_delay * 1000:.0f} ms extra per step"
             if args.capture_delay else "")
          + ".")
    print(f"[Perf] Actions: {len(env.actions)} calibrated button combinations "
          f"(set DIAGNOSTIC = \"actions\" to print them)")

    # Quick capability checks
    if not args.no_tests:
        print("[Test] Capture check...")
        frame = capture_window_printwindow(hwnd)
        if frame is None:
            print("[WARN] PrintWindow returned None. The window may be minimized "
                  "or use an unsupported renderer; frames will train as black.")
        else:
            brightness = float(frame.mean())
            print(f"[OK] Visual capture: {frame.shape}, mean brightness "
                  f"{brightness:.1f}")
            if brightness < 1.0:
                print("[WARN] Captured frame is almost entirely black. Restore "
                      "the window (unfocused is fine) and try again.")

        print("[Test] Audio check (3s)...")
        time.sleep(3)
        audio = env.audio_capture.get_audio()
        print(f"[OK] Audio buffer: {audio.shape}, max={np.abs(audio).max():.4f}")

        print("[Test] Sending a test key (W)...")
        env.input.tap_key('W', 0.1)
        time.sleep(0.5)

    loaded = None
    resume_path = None
    # Resume happens automatically (RESUME_ON_START); TARGET_WINDOW_HWND-style
    # constants are the only way to opt out now.
    want_resume = RESUME_ON_START or bool(args.resume)
    if args.no_resume:
        want_resume = False
        print("[Resume] Skipped (RESUME_ON_START = False); starting fresh.")

    if want_resume:
        if args.resume and args.resume != "auto":
            resume_path = args.resume
        else:
            candidate = os.path.join(args.checkpoint_dir, "last.pt")
            if os.path.exists(candidate):
                resume_path = candidate
            else:
                newest = sorted(
                    (os.path.join(args.checkpoint_dir, f)
                     for f in os.listdir(args.checkpoint_dir)
                     if f.startswith("checkpoint_") and f.endswith(".pt")),
                    key=os.path.getmtime,
                ) if os.path.isdir(args.checkpoint_dir) else []
                resume_path = newest[-1] if newest else None
            if resume_path is None:
                print(f"[Resume] No checkpoint in '{args.checkpoint_dir}' yet; "
                      f"starting from scratch.")

    if resume_path:
        loaded = load_checkpoint(resume_path, device,
                                 expect_actions=env.action_space.n,
                                 expect_frame_size=env.frame_size)
        if loaded is None:
            print("[Resume] Starting fresh instead.")
        elif env.intrinsic.load_state_dict(loaded["checkpoint"].get("intrinsic")):
            print("[Resume] Curiosity nets and reward statistics restored.")

    this_run = resolve_steps(args.steps)
    initial_step = int(loaded["checkpoint"].get("step", 0)) if loaded else 0
    if loaded and initial_step:
        print(f"[Resume] Continuing from step {initial_step}; this run will add "
              f"{describe_steps(this_run)} steps.")
    target_steps = initial_step + this_run
    if target_steps >= FOREVER_STEPS:
        print("[Training] No step limit: training continues until F10 (quit + "
              "checkpoint) or Ctrl+C. Progress is checkpointed automatically, "
              "so restarting picks up where this left off.")

    # Watch for you taking the controls, on a thread of its own so it stays
    # responsive while the training loop is busy inside a PPO update.
    human = HumanWatcher(target_hwnd=hwnd, keymap=keymap,
                         ignored_vks=hotkey_vks(), verbose=True)
    if human.start():
        print(f"[Human] Watching for you: any real key press, mouse button or "
              f"mouse movement hands the controls back to you, and the bot "
              f"takes them again {HUMAN_RELEASE_GRACE_SECONDS:.0f}s after your "
              f"last input.")
    else:
        for err in human.errors:
            print(f"[Human] {err}")
        print("[Human] WARNING: the input hooks could not be installed. The "
              "key-state fallback still catches held keys, but mouse movement "
              "will not hand the controls over. F8 always pauses the bot.")

    print()
    print("=" * 72)
    print("Training. The game window must stay focused for input to land.")
    print(f"Keys: {len(env.actions)} calibrated actions from "
          f"{env.keymap.path or 'the built-in defaults'}.")
    print(f"Rate: target {args.target_fps:.0f} steps/s at "
          f"{args.img_size}x{args.img_size}px - watch the [Perf] lines.")
    print("Touch the keyboard or mouse at any time: the bot lets go instantly, "
          "watches you, and takes over again 3s later.")
    print("=" * 72)

    try:
        train_ppo(
            env,
            total_steps=target_steps,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            keep_checkpoints=args.keep_checkpoints,
            checkpoint_dir=args.checkpoint_dir,
            enable_hotkeys=not args.no_hotkeys,
            final_model_path="background_swiglu_model.pt",
            initial_model=loaded["model"] if loaded else None,
            initial_optimizer=loaded["optimizer"] if loaded else None,
            initial_step=initial_step,
            human=human,
        )
    finally:
        human.stop()
        env.close()
        print("[Done] Environment closed. Hotkeys released.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())