"""
Checkpointing, hotkeys and Ctrl+C, in one place.

Checkpoints are written atomically and pruned to a fixed count, and a rolling
``last.pt`` always holds the freshest state so resuming is just "run it again".
Ordering uses a monotonic sequence number embedded in the filename rather than
the training step, because a resumed run continues its step counter and step
numbers alone would eventually make an older file look newer.
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import signal
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch

from .config import Config

# RegisterHotKey modifiers
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312

_MODIFIERS = {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "alt": MOD_ALT,
              "shift": MOD_SHIFT, "win": MOD_WIN, "super": MOD_WIN}

_NAMED_KEYS = {
    "SPACE": 0x20, "ESC": 0x1B, "TAB": 0x09, "ENTER": 0x0D,
    "INSERT": 0x2D, "DELETE": 0x2E, "HOME": 0x24, "END": 0x23,
    "PAGEUP": 0x21, "PAGEDOWN": 0x22,
    "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27,
}
for _i in range(1, 25):
    _NAMED_KEYS[f"F{_i}"] = 0x6F + _i
for _c in range(0x30, 0x3A):
    _NAMED_KEYS[chr(_c)] = _c
for _c in range(0x41, 0x5B):
    _NAMED_KEYS[chr(_c)] = _c


def parse_hotkey(spec: str):
    """'ctrl+shift+f8' -> (modifier flags, virtual key)."""
    parts = [p.strip().lower() for p in str(spec).split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey")
    key = parts[-1]
    modifiers = 0
    for part in parts[:-1]:
        if part not in _MODIFIERS:
            raise ValueError(f"unknown modifier '{part}' in '{spec}'")
        modifiers |= _MODIFIERS[part]
    vk = _NAMED_KEYS.get(key.upper())
    if vk is None:
        raise ValueError(f"unknown key '{key}' in '{spec}'")
    return modifiers, vk


class HotkeyController:
    """
    Global hotkeys on a thread that owns its own Win32 message loop.

    They fire regardless of which window has focus, which is the point: the
    bot can be paused or checkpointed while the user is working elsewhere.
    """

    def __init__(self, pause: str = "f8", save: str = "f9", quit_: str = "f10"):
        self.specs = {"pause": pause, "save": save, "quit": quit_}
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
        try:
            self._thread = threading.Thread(target=self._loop,
                                            name="Hotkeys", daemon=True)
            self._thread.start()
            self._thread.join(timeout=3.0)
        except Exception as exc:
            self.messages.put(f"Hotkeys unavailable: {exc}")
            return False
        return self.available

    def _loop(self):
        try:
            for action, spec in self.specs.items():
                if not spec:
                    continue
                try:
                    modifiers, vk = parse_hotkey(spec)
                except ValueError as exc:
                    self.messages.put(str(exc))
                    continue
                hotkey_id = self._next_id
                self._next_id += 1
                ok = ctypes.windll.user32.RegisterHotKey(
                    None, hotkey_id, modifiers | MOD_NOREPEAT, vk)
                if ok:
                    self._registered[action] = spec
                    self._id_to_action[hotkey_id] = action
                else:
                    self.messages.put(
                        f"Could not register {spec.upper()} for '{action}' - "
                        f"another program may already use it.")
            if not self._registered:
                return

            class _MSG(ctypes.Structure):
                _fields_ = [("hwnd", ctypes.c_void_p), ("message", ctypes.c_uint),
                            ("wParam", ctypes.c_void_p),
                            ("lParam", ctypes.c_void_p), ("time", ctypes.c_uint),
                            ("pt_x", ctypes.c_long), ("pt_y", ctypes.c_long)]

            msg = _MSG()
            while ctypes.windll.user32.GetMessageW(
                    ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY:
                    action = self._id_to_action.get(int(msg.wParam or 0))
                    if action:
                        self.actions.put(action)
        except Exception as exc:
            self.messages.put(f"Hotkey listener stopped: {exc}")
        finally:
            self._unregister_all()

    def _unregister_all(self):
        for hotkey_id in list(self._id_to_action):
            try:
                ctypes.windll.user32.UnregisterHotKey(None, hotkey_id)
            except Exception:
                pass
        self._id_to_action.clear()
        self._registered.clear()

    def poll(self) -> List[str]:
        out = []
        while True:
            try:
                out.append(self.actions.get_nowait())
            except queue.Empty:
                return out

    def drain_messages(self) -> List[str]:
        out = []
        while True:
            try:
                out.append(self.messages.get_nowait())
            except queue.Empty:
                return out

    def describe(self) -> str:
        if not self._registered:
            return "none"
        return ", ".join(f"{a.upper()}={self._registered[a].upper()}"
                         for a in ("pause", "save", "quit")
                         if a in self._registered)


class CheckpointManager:
    """Atomic, pruned checkpoints plus a rolling resume file."""

    def __init__(self, directory: str, keep: int = 3):
        self.directory = os.path.abspath(directory)
        self.keep = max(1, int(keep))
        os.makedirs(self.directory, exist_ok=True)
        self.last_path = os.path.join(self.directory, "last.pt")
        self.index_path = os.path.join(self.directory, "index.json")
        self._entries: List[dict] = self._load_index()
        self._seq = self._highest_seq()

    def _highest_seq(self) -> int:
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

    def _load_index(self) -> List[dict]:
        try:
            with open(self.index_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return [e for e in data if isinstance(e, dict)]
        except Exception:
            pass
        return []

    def _save_index(self) -> None:
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

    def _discover(self) -> List[dict]:
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
                found.append({"path": path, "seq": seq, "step": seq,
                              "mtime": os.path.getmtime(path),
                              "reason": "discovered"})
        except OSError:
            pass
        return found

    def save(self, payload: dict, step: int, reason: str = "periodic"
             ) -> Optional[str]:
        self._seq += 1
        seq = self._seq
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            self.directory,
            f"checkpoint_{seq:09d}_step{int(step):09d}_{stamp}.pt")
        tmp = path + ".tmp"
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
            try:
                torch.save(payload, self.last_path + ".tmp")
                os.replace(self.last_path + ".tmp", self.last_path)
            except Exception as exc:
                print(f"[Checkpoint] Rolling last.pt copy failed: {exc}")
        except Exception as exc:
            print(f"[Checkpoint] FAILED to save: {exc}")
            for candidate in (tmp, self.last_path + ".tmp"):
                if os.path.exists(candidate):
                    try:
                        os.remove(candidate)
                    except OSError:
                        pass
            return None

        if not self._entries:
            self._entries = self._discover()
        self._entries.append({"path": path, "seq": seq, "step": int(step),
                              "mtime": time.time(), "reason": reason})
        self._prune()
        self._save_index()
        return path

    def _prune(self) -> None:
        known = {e.get("path", "") for e in self._entries}
        for orphan in self._discover():
            if orphan.get("path") not in known:
                self._entries.append(orphan)
                known.add(orphan.get("path"))

        unique: Dict[str, dict] = {}
        for entry in self._entries:
            unique[str(entry.get("path", ""))] = entry
        entries = list(unique.values())
        entries.sort(key=lambda e: (int(e.get("seq") or 0),
                                    float(e.get("mtime", 0.0))))
        doomed = entries[:max(0, len(entries) - self.keep)]
        for entry in doomed:
            path = str(entry.get("path", ""))
            if path and os.path.basename(path) != "last.pt":
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
        self._entries = entries[-self.keep:]

    def list_kept(self) -> List[dict]:
        return sorted((e for e in self._entries
                       if os.path.exists(str(e.get("path", "")))),
                      key=lambda e: int(e.get("seq") or 0))

    def newest(self) -> Optional[str]:
        """The file to resume from: last.pt if present, else the newest."""
        if os.path.exists(self.last_path):
            return self.last_path
        kept = self.list_kept()
        return str(kept[-1]["path"]) if kept else None


class SignalController:
    """
    Everything the user can press to steer a running session.

    The loop polls this once per step; nothing here touches the model, so
    there is no chance of a signal handler writing a checkpoint mid-forward.
    """

    def __init__(self, cfg: Config):
        self.interval = max(0.05, float(cfg.checkpoint_interval_sec))
        self._paused = False
        self._stop = False
        self._manual_save = False
        self._accum = 0.0
        self._last = time.monotonic()
        self._interrupts = 0
        self._pause_notice = False
        self.stop_reason = "running"
        self.started_at = time.monotonic()

        self.hotkeys: Optional[HotkeyController] = None
        if cfg.enable_hotkeys:
            self.hotkeys = HotkeyController(cfg.hotkey_pause, cfg.hotkey_save,
                                            cfg.hotkey_quit)

    def start(self) -> None:
        self._last = time.monotonic()
        if self.hotkeys is not None:
            if self.hotkeys.start():
                for message in self.hotkeys.drain_messages():
                    print(f"[Hotkeys] {message}")
                print(f"[Hotkeys] {self.hotkeys.describe()} "
                      f"(work from any window)")
            else:
                for message in self.hotkeys.drain_messages():
                    print(f"[Hotkeys] {message}")
                print("[Hotkeys] No global hotkeys; Ctrl+C still saves and quits.")
        for name in ("SIGINT", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                pass

    def _on_signal(self, signum, _frame):
        self._interrupts += 1
        if self._interrupts == 1:
            self.stop_reason = "ctrl_c"
            self._stop = True
            print("\n[Signal] Interrupted - saving a checkpoint and stopping...")
            print("[Signal] (Press again to exit without saving.)")
        else:
            print("\n[Signal] Second interrupt - exiting now.")
            raise KeyboardInterrupt

    def service(self) -> List[str]:
        now = time.monotonic()
        self._accum += now - self._last
        self._last = now
        notes: List[str] = []
        if self.hotkeys is not None:
            for message in self.hotkeys.drain_messages():
                print(f"[Hotkeys] {message}")
            for action in self.hotkeys.poll():
                if action == "pause":
                    self._paused = not self._paused
                    self._pause_notice = False
                    notes.append("PAUSED - no input is being sent."
                                 if self._paused else "RESUMED.")
                elif action == "save":
                    self._manual_save = True
                    notes.append("Manual checkpoint requested.")
                elif action == "quit":
                    self.stop_reason = "hotkey"
                    self._stop = True
                    notes.append("Quit hotkey - saving and stopping.")
        return notes

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def stop_requested(self) -> bool:
        return self._stop

    def claim_pause_notice(self) -> bool:
        if self._paused and not self._pause_notice:
            self._pause_notice = True
            return True
        return False

    def checkpoint_due(self) -> Optional[str]:
        if self._manual_save:
            self._manual_save = False
            return "manual"
        if self._accum >= self.interval:
            self._accum -= self.interval
            return "periodic"
        return None


def lower_process_priority(level: str) -> None:
    """
    Ask Windows to schedule this process below the game.

    A bot that competes with the game for CPU makes the game stutter, which
    corrupts the very frames it is learning from.
    """
    if os.name != "nt":
        return
    classes = {"below_normal": 0x00004000,   # BELOW_NORMAL_PRIORITY_CLASS
               "idle": 0x00000040,           # IDLE_PRIORITY_CLASS
               "normal": 0x00000020,         # NORMAL_PRIORITY_CLASS
               "high": 0x00000080}           # HIGH_PRIORITY_CLASS
    flag = classes.get(str(level).lower())
    if flag is None:
        return
    try:
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, flag)
        print(f"[Proc] Process priority set to '{level}'.")
    except Exception as exc:
        print(f"[Proc] Could not set priority: {exc}")
