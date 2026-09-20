"""
The real game, wrapped as an environment.

This is the only file that knows the bot is driving a window, and it knows
nothing about *which* game: it grabs the client area as pixels, applies a
factored action (keys held, mouse turn, one tap or click), and reports what
changed.  The learning layer above it never sees a window handle.

Two deliberate choices:

* **The observation is grey frames plus a difference channel.** No colour, no
  audio.  Colour triples the encoder's input cost for very little control
  information, and an audio branch was measured as the single largest cost in
  the previous design while contributing almost nothing game-agnostic.
* **Held keys are level-triggered, not edge-triggered.** The bot says which
  buttons should be down, and this module presses what is new and releases what
  is gone.  That is what lets it walk continuously instead of tapping a key
  every 33 ms.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .capture import FrameGrabber, FrameStack, Pacer, gray_signature
from .config import Config
from .keys import ActionSpace, InputInjector


class RealGameEnv:
    """
    A live game window as an environment.

    ``step`` takes the dict produced by ``ActionSpace``/``GameSession`` and
    returns ``(observation, truncated, info)``.  ``truncated`` is True only
    when the window has gone away: an in-game reset is the reward module's
    business, not this one's.
    """

    def __init__(self, cfg: Config, hwnd: int, action_space: ActionSpace,
                 dry_run: bool = False):
        self.cfg = cfg
        self.hwnd = int(hwnd)
        self.action_space = action_space
        self.dry_run = bool(dry_run)

        self.grabber = FrameGrabber(self.hwnd)
        self.stack = FrameStack(cfg.frame_size, cfg.frame_stack)
        self.pacer = Pacer(cfg.target_fps, cfg.capture_delay)
        self.input = InputInjector(self.hwnd)

        self.observation_shape = (self.stack.channels, cfg.frame_size,
                                  cfg.frame_size)
        self.gray_channels = cfg.frame_stack

        self._held: set = set()
        self.steps = 0
        self.dead_captures = 0
        self.window_gone = False
        self.paused_by_user = False
        self.last_capture_ms = 0.0
        self.total_capture_seconds = 0.0

    # =====================================================================
    # Observation
    # =====================================================================
    def _grab(self) -> np.ndarray:
        started = time.perf_counter()
        frame = self.grabber.grab()
        self.last_capture_ms = 1000.0 * (time.perf_counter() - started)
        self.total_capture_seconds += time.perf_counter() - started
        if frame is None:
            self.dead_captures += 1
            # Return what the model already believes rather than a black frame:
            # a failed grab is not a game event, and pretending the screen went
            # black would look like a huge, unexplained change.
            return None
        self.dead_captures = 0
        return self.stack.push(frame)

    def reset(self, seed: Optional[int] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
        self.release_all()
        self.stack.clear()
        self.pacer.reset()
        self.steps = 0
        observation = None
        for _ in range(self.cfg.frame_stack):
            observation = self._grab()
            if observation is None:
                break
        if observation is None:
            # Never seen a frame yet: give the model a blank one so the shapes
            # stay right and the loop can decide whether the window is gone.
            observation = np.zeros(self.observation_shape, dtype=np.float32)
        return observation, gray_signature(observation[:self.gray_channels]
                                           .mean(axis=0, dtype=np.float32))

    # =====================================================================
    # Input
    # =====================================================================
    def set_held(self, vks: List[int]) -> None:
        target = {int(v) for v in vks if v}
        if target == self._held:
            return
        if not self.dry_run:
            self.input.begin_action()
            for vk in sorted(self._held - target):
                self.input.release_vk(vk)
            for vk in sorted(target - self._held):
                self.input.press_vk(vk)
        self._held = target

    def release_all(self) -> None:
        if not self.dry_run:
            try:
                self.input.begin_action()
            except Exception:
                pass
            for vk in sorted(self._held):
                self.input.release_vk(vk)
        self._held = set()

    def suspend(self) -> None:
        """Stop injecting entirely (pause). Releases everything first."""
        self.release_all()
        self.input.suspended = True

    def resume(self) -> None:
        self.input.suspended = False

    # =====================================================================
    # Environment API
    # =====================================================================
    def step(self, action: Dict) -> Tuple[np.ndarray, bool, Dict]:
        if self.paused_by_user:
            self.release_all()
        else:
            self.set_held(action.get("held_vks") or [])

            turn = action.get("turn") or (0, 0)
            if not self.dry_run:
                self.input.begin_action()
                if turn[0] or turn[1]:
                    self.input.mouse_move(int(turn[0]), int(turn[1]))
                tap_vk = int(action.get("tap_vk") or 0)
                if tap_vk:
                    self.input.tap_vk(tap_vk, self.cfg.tap_seconds)
                self.input.end_action()

        self.pacer.tick()
        observation = self._grab()
        if observation is None:
            observation = np.zeros(self.observation_shape, dtype=np.float32)

        self.steps += 1
        gone = not self._window_alive()
        if gone:
            self.window_gone = True
        perf = self.pacer.report()
        if perf:
            print(perf, flush=True)
        info = {"window_gone": gone,
                "capture_ms": self.last_capture_ms,
                "steps_per_second": self.pacer.achieved_hz}
        return observation, bool(gone), info

    def _window_alive(self) -> bool:
        try:
            from .capture import win32gui
            return bool(win32gui.IsWindow(self.hwnd))
        except Exception:
            return True

    # =====================================================================
    # Misc
    # =====================================================================
    @property
    def mean_capture_ms(self) -> float:
        return 1000.0 * self.total_capture_seconds / max(1, self.steps)

    def close(self) -> None:
        self.release_all()
        try:
            self.grabber.close()
        except Exception:
            pass
