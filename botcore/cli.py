"""
The command surface: one command, plus a few optional modes.

    python bot1.py                     learn to play the game in the game window
    python bot1.py --calibrate         show the bot which keys you use (once)
    python bot1.py --selftest          prove the algorithm learns, on a game it
                                       has never seen, with no game knowledge
    python bot1.py --list-windows      find the handle for --window
    python bot1.py --check-capture     is the bot actually seeing the game?
    python bot1.py --actions           what can this bot press?
    python bot1.py --benchmark         is this machine fast enough?
    python bot1.py --release           lift every key the last run left down

Everything is plain flags rather than subcommands, because the common case is
"run it" and the rest are one-shot checks.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

# ---- imports are done defensively so a missing optional package prints a
# ---- single useful line instead of a traceback from deep inside torch.
def _need(module: str, pip_name: str = ""):
    try:
        return __import__(module)
    except ImportError as exc:
        print(f"[Setup] Missing Python package '{pip_name or module}' ({exc}).")
        print("[Setup] Install the training dependencies with:")
        print("[Setup]   python -m pip install torch numpy opencv-python pywin32")
        sys.exit(2)


_need("numpy")
_need("torch")
if os.name == "nt":
    _need("win32gui", "pywin32")
    _need("win32api", "pywin32")
    _need("win32ui", "pywin32")
    _need("win32process", "pywin32")

import torch

from . import __version__
from .calibrate import KeyRecorder, find_toggle_vk
from .capture import require_windows, resolve_target_window
from .config import ARCH_NAME, REWARD_VERSION, Config
from .diagnostics import (benchmark, check_capture, list_windows, preflight,
                          release_all_keys, show_actions)
from .keys import ActionSpace, Keymap
from .runtime import CheckpointManager, SignalController, lower_process_priority
from .session import GameSession, atomic_save, load_checkpoint


# =============================================================================
# Argument parsing
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    cfg = Config()
    parser = argparse.ArgumentParser(
        prog="bot1.py",
        description="A small, CPU-friendly, game-agnostic learning bot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version",
                        version=f"bot1 {__version__} ({ARCH_NAME}, reward v{REWARD_VERSION})")

    # ---- what to do ----
    parser.add_argument("--calibrate", action="store_true",
                        help="watch you play and record which keys to allow")
    parser.add_argument("--selftest", action="store_true",
                        help="train on a synthetic game and report whether the "
                             "algorithm learned it (no game needed)")
    parser.add_argument("--list-windows", action="store_true",
                        help="list windows and their handles")
    parser.add_argument("--check-capture", action="store_true",
                        help="verify the bot can see live frames")
    parser.add_argument("--actions", action="store_true",
                        help="show the action space this keymap produces")
    parser.add_argument("--benchmark", action="store_true",
                        help="measure this machine's per-decision cost")
    parser.add_argument("--release", action="store_true",
                        help="lift every key the whitelist allows")

    # ---- selection ----
    parser.add_argument("--window", type=int, default=None,
                        help="window handle to play (see --list-windows)")
    parser.add_argument("--pick", action="store_true",
                        help="always show the window picker")
    parser.add_argument("--keymap", default=cfg.keymap_path,
                        help=f"keymap file (default {cfg.keymap_path})")

    # ---- observation and control ----
    parser.add_argument("--frame-size", type=int, default=cfg.frame_size,
                        help=f"observation resolution (default {cfg.frame_size})")
    parser.add_argument("--frame-stack", type=int, default=cfg.frame_stack,
                        help=f"grey frames stacked (default {cfg.frame_stack})")
    parser.add_argument("--action-repeat", type=int, default=cfg.action_repeat,
                        help=f"control steps per decision (default {cfg.action_repeat})")
    parser.add_argument("--fps", type=float, default=cfg.target_fps,
                        help=f"target decisions per second (default {cfg.target_fps:.0f})")
    parser.add_argument("--max-held-keys", type=int, default=cfg.max_held_keys,
                        help=f"keys the bot may hold at once (default {cfg.max_held_keys})")

    # ---- training ----
    parser.add_argument("--rollout", type=int, default=cfg.rollout_steps,
                        help=f"steps per PPO update (default {cfg.rollout_steps})")
    parser.add_argument("--minibatch", type=int, default=cfg.minibatch_size,
                        help=f"steps per minibatch (default {cfg.minibatch_size})")
    parser.add_argument("--epochs", type=int, default=cfg.epochs_per_update,
                        help=f"passes per rollout (default {cfg.epochs_per_update})")
    parser.add_argument("--lr", type=float, default=cfg.adam_lr,
                        help=f"learning rate (default {cfg.adam_lr})")
    parser.add_argument("--entropy", type=float, default=cfg.entropy_coef,
                        help=f"entropy bonus (default {cfg.entropy_coef})")
    parser.add_argument("--steps", type=int, default=0,
                        help="stop after this many decisions (0 = forever)")
    parser.add_argument("--update-budget", type=float,
                        default=cfg.update_seconds_budget,
                        help=f"seconds a PPO update may take "
                             f"(default {cfg.update_seconds_budget:.1f})")
    parser.add_argument("--torch-threads", type=int, default=cfg.torch_threads,
                        help=f"torch threads (default {cfg.torch_threads})")
    parser.add_argument("--seed", type=int, default=cfg.seed)

    # ---- run control ----
    parser.add_argument("--checkpoint-dir", default=cfg.checkpoint_dir)
    parser.add_argument("--checkpoint-every", type=float,
                        default=cfg.checkpoint_interval_sec,
                        help="seconds between automatic checkpoints")
    parser.add_argument("--keep", type=int, default=cfg.checkpoint_keep,
                        help="checkpoints to keep")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any existing checkpoint and start over")
    parser.add_argument("--no-hotkeys", action="store_true",
                        help="disable the global pause/save/quit keys")
    parser.add_argument("--priority", default=cfg.priority,
                        choices=["below_normal", "idle", "normal", "high"])
    parser.add_argument("--log", default=None,
                        help="append per-update metrics as JSON lines to this file")
    parser.add_argument("--preview", action="store_true",
                        help="show a small live view of what the bot sees")
    parser.add_argument("--dry-run", action="store_true",
                        help="run everything but inject no input")

    # ---- self-test tuning ----
    parser.add_argument("--selftest-steps", type=int, default=24000,
                        help="decisions for --selftest (default 24000)")
    parser.add_argument("--selftest-corridor", type=int, default=60,
                        help="length of the synthetic corridor")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config(
        window_hwnd=args.window,
        prefer_game_window=not args.pick,
        frame_size=args.frame_size,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        target_fps=args.fps,
        max_held_keys=args.max_held_keys,
        rollout_steps=args.rollout,
        minibatch_size=args.minibatch,
        epochs_per_update=args.epochs,
        adam_lr=args.lr,
        entropy_coef=args.entropy,
        update_seconds_budget=args.update_budget,
        torch_threads=args.torch_threads,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval_sec=args.checkpoint_every,
        checkpoint_keep=args.keep,
        resume=not args.fresh,
        enable_hotkeys=not args.no_hotkeys,
        priority=args.priority,
        log_json=args.log,
        preview=args.preview,
        keymap_path=args.keymap,
    )
    return cfg.env_overrides().validate()


# =============================================================================
# Modes
# =============================================================================

def run_calibration(cfg: Config, args) -> int:
    """
    Record which keys the user actually plays with.

    Calibration must never press anything, and that is guaranteed structurally
    rather than by a flag: no ``RealGameEnv`` is ever constructed here, so the
    only object that can inject input does not exist for the duration.
    """
    require_windows("Calibration")
    target = resolve_target_window(cfg.window_hwnd, prefer_game_window=False,
                                   allow_prompt=True)
    if target is None:
        print("[Calibrate] No window chosen; nothing to do.")
        return 1

    toggle_vk = find_toggle_vk(cfg.calibrate_key)
    recorder = KeyRecorder(toggle_vk=toggle_vk,
                           target_hwnd=int(target["hwnd"]),
                           only_when_focused=True)
    print()
    print("=" * 74)
    print("  CALIBRATION")
    print("=" * 74)
    print(f"  Target: '{target['title']}' ({target.get('process') or '?'})")
    print()
    print("  Play the game normally for a minute or so.")
    print(f"  Press {cfg.calibrate_key.upper()} to START recording, play, then")
    print(f"  press {cfg.calibrate_key.upper()} again to STOP.")
    print()
    print("  The bot records which keys you use, and how long you hold each")
    print("  one - that is how it tells movement keys from menu keys. It also")
    print("  measures how far you move the mouse, which becomes its turn")
    print("  step. It cannot press anything during this: injection is off.")
    print("=" * 74)
    print()

    if not recorder.start():
        print("[Calibrate] Could not install the input hooks "
              "(are you running as a normal user, not a service?).")
        for error in recorder.errors:
            print(f"           {error}")
        recorder.stop()
        return 1

    started = time.perf_counter()
    last_status = started
    try:
        while True:
            recorder.pump()
            time.sleep(0.01)
            now = time.perf_counter()
            if now - last_status > 5.0:
                last_status = now
                state = "RECORDING" if recorder.recording else "waiting"
                print(f"  [{now - started:4.0f}s] {state} - "
                      f"{recorder.events} event(s)", flush=True)
            if not recorder.recording and recorder.toggles >= 2:
                break
            if recorder.toggles >= 1 and not recorder.recording:
                break
    except KeyboardInterrupt:
        print("\n[Calibrate] Stopped by Ctrl+C.")
    finally:
        recorder.stop()

    print()
    print("-" * 74)
    print("  WHAT WAS RECORDED")
    print("-" * 74)
    print(recorder.report())
    print("-" * 74)

    keymap = recorder.build_keymap(min_hold=cfg.calibrate_min_hold)
    if not keymap.holds and not keymap.vks:
        print("[Calibrate] Nothing usable was recorded, so the existing keymap "
              "is left untouched.")
        return 1
    path = keymap.save(cfg.keymap_path)
    print(f"[Calibrate] Wrote {path}")
    print(f"            {keymap.describe()}")
    print()
    print("  Next: python bot1.py   (it will pick these keys up automatically)")
    print()
    show_actions(cfg, keymap)
    return 0


def run_selftest(args) -> int:
    from .synth import run_learning_check
    results = run_learning_check(
        verbose=True, seed=args.seed, steps=args.selftest_steps,
        frame_size=min(48, args.frame_size),
        rollout_steps=min(1024, args.rollout),
        corridor_length=args.selftest_corridor,
    )
    return 0 if results.get("ok") else 1


def run_training(cfg: Config, args) -> int:
    require_windows("Training")
    lower_process_priority(cfg.priority)
    torch.set_num_threads(max(1, int(cfg.torch_threads)))

    keymap = Keymap.load(cfg.keymap_path)
    if keymap is None:
        keymap = Keymap.default()
        print(f"[Keymap] No '{cfg.keymap_path}': using built-in defaults.")
        print("[Keymap] Run --calibrate once so the bot may only press the "
              "keys you actually use.")
    else:
        print(f"[Keymap] {keymap.describe()}")

    target = resolve_target_window(cfg.window_hwnd,
                                   prefer_game_window=cfg.prefer_game_window,
                                   allow_prompt=True)
    if target is None:
        print("[Window] No window selected; nothing to do.")
        return 1
    cfg.window_hwnd = int(target["hwnd"])

    for warning in preflight(cfg, keymap):
        print(f"[Check] {warning}")

    space = ActionSpace.build(keymap, max_held=cfg.max_held_keys,
                              turn_levels=cfg.turn_levels)
    print(f"[Actions] {space.describe()}")

    from .game import RealGameEnv
    env = RealGameEnv(cfg, cfg.window_hwnd, space, dry_run=args.dry_run)
    session = GameSession(cfg, env, space, keymap=keymap)

    checkpoint_manager = CheckpointManager(cfg.checkpoint_dir, cfg.checkpoint_keep)
    resumed = False
    if cfg.resume:
        path = checkpoint_manager.newest()
        if path:
            payload = load_checkpoint(path, session.device)
            if payload is not None:
                resumed = session.load_state_dict(payload)
                if resumed:
                    print(f"[Resume] Continuing from '{path}' at step "
                          f"{session.total_steps:,}.")
        if not resumed and path:
            print("[Resume] Starting a fresh run (the checkpoint above was "
                  "not compatible with this configuration).")

    print()
    print("-" * 74)
    print(f"  bot1 {__version__} - {ARCH_NAME}")
    print(f"  {cfg.describe()}")
    print(f"  {space.describe()}")
    parameters = sum(p.numel() for p in session.policy.parameters())
    print(f"  {parameters:,} parameters, torch threads "
          f"{torch.get_num_threads()}")
    print("-" * 74)
    print("  F8 pause | F9 save | F10 quit | Ctrl+C save and quit")
    print(f"  Checkpoint every {cfg.checkpoint_interval_sec / 60:.1f} min, "
          f"keeping {cfg.checkpoint_keep} in '{cfg.checkpoint_dir}'")
    print("  Stop at any time; running it again continues from where it was.")
    print("-" * 74)
    print()

    control = SignalController(cfg)
    control.start()

    def save(reason: str) -> None:
        step = session.total_steps
        payload = session.state_dict()
        path = checkpoint_manager.save(payload, step=step, reason=reason)
        if path:
            print(f"[Checkpoint] '{reason}' at step {step:,} -> {path}")
        if reason in ("final",):
            bad = session.trainer.nonfinite_parameters()
            if bad:
                print(f"[Checkpoint] WARNING: non-finite weights in {bad[:3]}; "
                      f"NOT writing {cfg.final_model_path}.")
            elif atomic_save({"model": session.policy.state_dict(),
                              "config": session.state_dict()["model_config"],
                              "action_space": space.to_dict(),
                              "step": step}, cfg.final_model_path):
                print(f"[Checkpoint] Final weights -> {cfg.final_model_path}")

    max_decisions = int(args.steps) if args.steps and args.steps > 0 else None
    try:
        session.run(control=control, max_decisions=max_decisions, save_fn=save)
    except KeyboardInterrupt:
        print("\n[Run] Interrupted.")
    finally:
        print("[Run] Saving a final checkpoint...")
        save("final")
        session.close()

    print()
    print(f"[Done] {session.total_steps:,} decisions this run, "
          f"{session.episodes:,} episode(s).")
    for entry in checkpoint_manager.list_kept():
        print(f"       step {entry['step']:>9,}  ({entry['reason']})  "
              f"{entry['path']}")
    print("[Done] Resume with: python bot1.py")
    print()
    return 0


# =============================================================================
# Entry point
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest(args)

    if args.list_windows:
        return list_windows()

    cfg = config_from_args(args)

    if args.actions:
        keymap = Keymap.load(cfg.keymap_path) or Keymap.default()
        return show_actions(cfg, keymap)

    if args.release:
        target = resolve_target_window(cfg.window_hwnd,
                                       prefer_game_window=True,
                                       allow_prompt=True)
        if target is None:
            print("[Release] No window chosen.")
            return 1
        return release_all_keys(cfg, int(target["hwnd"]))

    if args.calibrate:
        return run_calibration(cfg, args)

    if args.check_capture or args.benchmark:
        target = resolve_target_window(cfg.window_hwnd,
                                       prefer_game_window=True,
                                       allow_prompt=True)
        if target is None:
            print("[Check] No window chosen.")
            return 1
        cfg.window_hwnd = int(target["hwnd"])
        if args.check_capture:
            return check_capture(cfg, cfg.window_hwnd)
        keymap = Keymap.load(cfg.keymap_path) or Keymap.default()
        return benchmark(cfg, keymap)

    torch.set_num_threads(max(1, int(cfg.torch_threads)))
    try:
        return run_training(cfg, args)
    except RuntimeError as exc:
        print(f"[Error] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
