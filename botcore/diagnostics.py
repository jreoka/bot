"""
Diagnostics: the checks worth running before trusting a training run.

Every mode here answers a question that otherwise gets answered hours later by
watching a log:

* did the keys get released after the last run (and why it matters),
* is the captured image actually live and changing,
* which window is which,
* what action space does this keymap actually produce,
* how long does a decision take on this machine.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np

from .capture import (FrameGrabber, FrameStack, describe_window,
                      enumerate_windows, format_window_table, get_process_name,
                      get_window_pid, looks_like_game,
                      looks_like_non_game_process, require_windows)
from .config import Config
from .keys import ActionSpace, InputInjector, Keymap


def list_windows() -> int:
    """Print every usable window with its handle, for --window."""
    require_windows("Listing windows")
    windows = enumerate_windows()
    if not windows:
        print("No windows found.")
        return 1
    print()
    print("-" * 78)
    print("  WINDOWS (pass one to --window HWND)")
    print("-" * 78)
    print(format_window_table(windows))
    print("-" * 78)
    games = [w for w in windows if looks_like_game(w)]
    if games:
        print(f"  Largest likely game: {games[0]['title']} "
              f"({games[0]['process']}) hwnd={games[0]['hwnd']}")
        print("  That is what the bot picks automatically.")
    else:
        print("  Nothing here looks like a game; the bot will ask you to pick.")
    print("  Marked windows are never auto-selected: the terminal this bot runs")
    print("  in, its own console, and programs that are not games.")
    print()
    return 0


def release_all_keys(cfg: Config, hwnd: int) -> int:
    """
    Lift every key the bot is allowed to press, plus the mouse buttons.

    A run that was killed rather than stopped can leave a game believing a key
    is still down, which shows up as a character walking into a wall forever.
    This is the cure, and it is safe to run at any time.
    """
    require_windows("Releasing keys")
    keymap = Keymap.load(cfg.keymap_path) or Keymap.default()
    injector = InputInjector(hwnd)
    print("[Release] Lifting every key in the whitelist and all mouse buttons.")
    # The key-ups have to land in the game, so ask for its focus once; if that
    # fails they still go out, which is better than leaving a key down.
    injector.acquire_focus(force=True)
    injector.begin_action()
    for name, vk in list(keymap.vks.items()):
        injector.release_vk(int(vk))
    for vk in (0x01, 0x02, 0x04):
        injector.release_vk(vk)
    injector.end_action()
    print("[Release] Done.")
    return 0


def check_capture(cfg: Config, hwnd: int, seconds: float = 5.0) -> int:
    """
    Watch the captured frames and say whether the bot can actually see.
    """
    require_windows("Capture diagnostics")
    grabber = FrameGrabber(hwnd)
    stack = FrameStack(cfg.frame_size, cfg.frame_stack)
    pid = get_window_pid(hwnd)
    print()
    print("-" * 74)
    print("  CAPTURE CHECK")
    print("-" * 74)
    print(f"  Window:  {hwnd}  process {get_process_name(pid) or '?'}")
    print(f"  Observing for {seconds:.0f}s at {cfg.frame_size}px...")

    frames = 0
    deltas: List[float] = []
    previous = None
    black = 0
    failures = 0
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        raw = grabber.grab()
        if raw is None:
            failures += 1
            time.sleep(0.05)
            continue
        observation = stack.push(raw)
        if previous is not None:
            deltas.append(float(np.abs(observation[-1]).mean()))
        previous = observation.copy()
        if float(observation[0].mean()) < 0.01:
            black += 1
        frames += 1
        time.sleep(1.0 / max(1.0, cfg.target_fps))

    elapsed = time.perf_counter() - started
    print(f"  Captured {frames} frame(s) in {elapsed:.1f}s "
          f"({frames / max(1e-6, elapsed):.1f} fps), "
          f"{grabber.mean_ms:.1f} ms per grab")
    if failures:
        print(f"  {failures} grab(s) returned nothing.")
    if frames < 3:
        print("  [FAIL] The window is not giving up any frames. Is it "
              "minimized, or is the game still loading?")
        grabber.close()
        return 1
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    change = 100.0 * sum(1 for d in deltas if d > 1e-3) / max(1, len(deltas))
    if black > frames * 0.9:
        print("  [FAIL] Almost every frame is black. A hardware-accelerated "
              "game sometimes refuses PrintWindow; try borderless windowed "
              "mode, or run the game windowed.")
        grabber.close()
        return 1
    if mean_delta < 1e-4:
        print("  [WARN] The image is not changing at all. The bot will learn "
              "nothing from a still screen: the game is probably paused while "
              "unfocused, or the window is showing a loading screen.")
    else:
        print(f"  [OK]   Image is live: mean change {mean_delta:.4f}, "
              f"changing on {change:.0f}% of frames.")
    print("-" * 74)
    print()
    grabber.close()
    return 0


def show_actions(cfg: Config, keymap: Keymap) -> int:
    """Print the action space this keymap produces."""
    keymap.set_mouse_turn(cfg.mouse_turn_pixels)
    space = ActionSpace.build(keymap, max_held=cfg.max_held_keys,
                              speed_levels=cfg.speed_levels)
    base = max(1, int(keymap.default_mouse_turn))
    print()
    print("-" * 74)
    print("  ACTION SPACE")
    print("-" * 74)
    print(f"  Keymap: {keymap.path or '(built-in defaults)'}")
    print(f"  {keymap.describe()}")
    print(f"  {space.describe()}")
    print()
    print(f"  Held keys, each decided independently "
          f"(at most {space.max_held} at once):")
    for index, name in enumerate(space.hold_names):
        print(f"    [{index}] {name}")
    print()
    print("  Turning (two choices per step: which way, and how fast).")
    print(f"  The base step is the bot's own: it starts at {base}px and the")
    print(f"  bot moves it inside [{cfg.mouse_turn_min}, {cfg.mouse_turn_max}]"
          f" pixels as it learns this game.")
    print("    directions:")
    for index, (name, delta) in enumerate(space.directions):
        rendered = ("do not turn" if delta == (0, 0)
                    else f"move mouse by {delta[0]:+d}, {delta[1]:+d} per 1x")
        print(f"      [{index}] {name:<9} {rendered}")
    print("    speeds (multipliers on the base step):")
    for index, level in enumerate(space.speed_levels):
        pixels = max(1, int(round(base * float(level))))
        print(f"      [{index}] x{float(level):<5g} -> {pixels:+d}px from the "
              f"current {base}px base")
    print()
    print("  Tap / click (one choice per step):")
    for index, entry in enumerate(space.taps):
        print(f"    [{index}] {entry.get('label', '?')}")
    print("-" * 74)
    print()
    return 0


def benchmark(cfg: Config, keymap: Keymap, decisions: int = 120) -> int:
    """
    Measure the real per-decision cost of this configuration on this machine.

    This is the number that decides whether the bot can keep up with the game,
    so it is measured rather than assumed.
    """
    import torch

    from .game import RealGameEnv

    hwnd = cfg.window_hwnd
    if not hwnd:
        print("[Bench] Needs a window: pass --window HWND (see --list-windows).")
        return 1
    require_windows("Benchmarking")

    keymap.set_mouse_turn(cfg.mouse_turn_pixels)
    space = ActionSpace.build(keymap, max_held=cfg.max_held_keys,
                              speed_levels=cfg.speed_levels)
    try:
        from .session import GameSession
    except Exception as exc:
        print(f"[Bench] Could not build a session: {exc}")
        return 1

    # Apply the configured thread count before measuring anything, or the
    # benchmark would be reporting a number for settings the run will not use.
    torch.set_num_threads(max(1, int(cfg.torch_threads)))

    env = RealGameEnv(cfg, int(hwnd), space, dry_run=True)
    session = GameSession(cfg, env, space, keymap=keymap)
    session.reset(seed=0)

    print()
    print("-" * 74)
    print("  BENCHMARK (dry run: nothing is injected)")
    print("-" * 74)
    print(f"  {cfg.describe()}")
    print(f"  torch threads: {torch.get_num_threads()}")
    for _ in range(15):
        action = session.decide()
        session.step(action)

    model_times, env_times = [], []
    for _ in range(int(decisions)):
        started = time.perf_counter()
        action = session.decide()
        model_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        session.step(action)
        env_times.append(time.perf_counter() - started)

    model_ms = 1000.0 * float(np.median(model_times))
    env_ms = 1000.0 * float(np.median(env_times))
    total_ms = model_ms + env_ms
    print(f"  decision + model : {model_ms:6.2f} ms")
    print(f"  env + capture    : {env_ms:6.2f} ms")
    print(f"  total per step   : {total_ms:6.2f} ms  "
          f"({1000.0 / max(1e-6, total_ms):.1f} steps/s)")
    budget_ms = 1000.0 / max(1.0, cfg.target_fps)
    print(f"  target           : {budget_ms:6.2f} ms  "
          f"({cfg.target_fps:.0f} steps/s)")
    if total_ms > budget_ms:
        print(f"  [WARN] Too slow for the target rate ({total_ms:.0f} ms of "
              f"{budget_ms:.0f} ms). The bot will still run; it will react late.")
        # Say which half is the problem, because the two have different fixes
        # and guessing between them is how an afternoon disappears.
        if env_ms > model_ms:
            print(f"         The time is in capture ({env_ms:.0f} ms), not the "
                  f"model ({model_ms:.0f} ms). A big window costs more to grab "
                  f"than a small one; run the game windowed, or lower "
                  f"target_fps.")
            print("         Cross-check with: python bot1.py --check-capture")
        else:
            print(f"         The time is in the model ({model_ms:.0f} ms). "
                  f"Lower embed_dim (now {cfg.embed_dim}), mem_tokens (now "
                  f"{cfg.mem_tokens}), transformer_layers (now "
                  f"{cfg.transformer_layers}) or frame_size (now "
                  f"{cfg.frame_size}), and keep torch_threads at 1.")
    else:
        print("  [OK]   Fast enough for the target rate.")
    print("-" * 74)
    print()
    session.close()
    return 0


def action_space_advice(cfg: Config, keymap: Keymap,
                        space: ActionSpace) -> List[str]:
    """
    Say out loud when the whitelist cannot produce movement.

    This is the other half of "the bot just stands there".  A keymap recorded
    from a short calibration can easily end up containing no movement key at
    all - a player who spent the minute in a menu, or in a game where the
    forward key is W and the recorder only caught the strafe key.  The action
    space is then perfectly well-formed and completely unable to walk: the
    policy presses what it has, the screen barely changes, and the run looks
    like a broken learner when it is really a broken whitelist.

    A cursor-locked game has the same failure by a different route.  Such a
    title hides the real look speed from the recorder, so the measured mouse
    step comes out at a pixel or two, and the bot's idea of "turn the view" is
    a twitch nothing can see.  Both are checked here because both are fixed by
    a command-line flag rather than by waiting for the bot to learn better.
    """
    lines: List[str] = []
    holds = {str(name).upper() for name in space.hold_names}
    movement = {"W", "A", "S", "D"}
    if not (holds & movement):
        lines.append(
            f"No movement key in the whitelist: hold=[{', '.join(sorted(holds)) or '-'}]. "
            f"A bot with no way to walk cannot make progress on any game - it "
            f"will press what it has and look like it is shaking on the spot.")
        lines.append(
            f"Fix: re-run --calibrate and spend the recording actually walking "
            f"around, or edit '{cfg.keymap_path}' to add W/A/S/D (or whatever "
            f"this game calls forward).")
    if keymap is not None and int(keymap.default_mouse_turn) < 4:
        lines.append(
            f"The measured mouse turn step is {int(keymap.default_mouse_turn)} "
            f"px, which is the signature of a game that locks the cursor: the "
            f"recorder cannot see the real look speed, so the measurement is a "
            f"twitch. The bot now sets its own speed - it starts from this "
            f"number, scales it up when a turn does not move the view, and "
            f"chooses the multiplier per decision - so this is not fatal. If "
            f"the view still never turns, pass --mouse-turn 20 (or more) to "
            f"start it higher.")
    if len(space.taps) > 6:
        lines.append(
            f"{len(space.taps)} tap/click choices are enabled, so a large share "
            f"of every rollout is spent on clicking things. In a game with an "
            f"inventory or an attack button this is mostly wasted input.")
    return lines


def preflight(cfg: Config, keymap: Optional[Keymap],
              target: Optional[dict] = None) -> List[str]:
    """
    Return a list of warnings worth printing before a long run starts.

    ``target`` is the window that was just selected, and half of these warnings
    are about it: driving the wrong window is the one mistake that produces a
    completely healthy-looking run with nothing happening in the game, so it is
    worth more than one line of checking.
    """
    warnings: List[str] = preflight_warnings(cfg, keymap, target)
    space = None
    if keymap is not None:
        try:
            space = ActionSpace.build(keymap, max_held=cfg.max_held_keys,
                                      speed_levels=cfg.speed_levels)
        except Exception:
            space = None
    if space is not None:
        warnings.extend(action_space_advice(cfg, keymap, space))
    return warnings


def preflight_warnings(cfg: Config, keymap: Optional[Keymap],
                       target: Optional[dict] = None) -> List[str]:
    """The window/liveness warnings, before the action-space advice."""
    warnings: List[str] = []
    import torch
    if os.name != "nt":
        warnings.append("Not running on Windows: capture and input injection "
                        "will not work.")
    if torch.get_num_threads() > 2:
        warnings.append(
            f"torch is using {torch.get_num_threads()} threads; on a small "
            "model this is usually slower than 1. Set BOT_TORCH_THREADS=1 to "
            "compare.")
    if keymap is None:
        warnings.append(
            "No keymap file: using built-in defaults. Run --calibrate once so "
            "the bot may only press the keys you actually use.")
    elif not keymap.holds:
        warnings.append("The keymap has no hold keys, so the bot cannot walk "
                        "or hold anything down.")
    if target is not None:
        process = str(target.get("process") or "")
        if looks_like_non_game_process(process):
            warnings.append(
                f"The selected window is {describe_window(target)}, which is "
                f"not a game. Every key the bot presses will go to that "
                f"program instead of the game. Check --list-windows, then "
                f"restart with --window HWND.")
        if keymap is not None and keymap.game_hwnd:
            if int(target.get("hwnd") or 0) != int(keymap.game_hwnd):
                warnings.append(
                    f"--calibrate recorded the game as '{keymap.game}' "
                    f"(hwnd {keymap.game_hwnd}), but this run selected "
                    f"{describe_window(target)}. If the recorded handle is "
                    f"still the game, restart with --window {keymap.game_hwnd}.")
    if keymap is not None:
        measured = float((keymap.mouse_sensitivity or {}).get(
            "median_pixels_per_step") or 0.0)
        if keymap.default_mouse_turn < 4:
            warnings.append(
                f"The mouse turn step is {int(keymap.default_mouse_turn)} "
                f"pixel(s) (measured median {measured:.1f} px at calibration). "
                f"A game that locks the cursor hides your real look speed from "
                f"the recorder. The bot starts from this number and scales it "
                f"up on its own when turns do not move the view, but if you "
                f"already know the right step, --mouse-turn sets it.")
    return warnings
