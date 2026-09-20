"""
Every tunable in one place.

The defaults are chosen for a 6-core laptop CPU with no GPU (they were measured
on a Ryzen 5 7530U).  Changing them is not required to run the bot: the shipped
values keep a control step in the low milliseconds and a PPO update under a
couple of seconds, which leaves the machine free to run the game.

Two rules were applied throughout, and they are worth preserving:

* Anything that grows without bound gets a cap here.
* Anything that loops gets a time or iteration budget here.

That is what makes "runs for hours without stalling" a property of the
configuration rather than a hope.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple


def _env(name: str, default):
    """Read an override from the environment, coercing to the default's type."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Config:
    # =====================================================================
    # Window / capture
    # =====================================================================
    # None = auto-detect a likely game window (or prompt). An int = that hwnd.
    window_hwnd: Optional[int] = None
    # True = silently take the largest likely game window; False = ask.
    prefer_game_window: bool = True

    # =====================================================================
    # Observation
    #
    # Frames are converted to grayscale once, resized once, and then the model
    # sees the last `frame_stack` of them plus their first difference. Stacking
    # grey frames is how the bot perceives motion, and it costs nothing extra:
    # the CNN always runs on the same fixed number of channels.
    # =====================================================================
    frame_size: int = 96
    frame_stack: int = 4
    # How many control steps one decision is repeated for. 1 = decide every
    # step. Raising it multiplies the effective throughput of everything
    # downstream (fewer decisions per second of game) at the cost of reaction
    # time, and is the cheapest speed lever available.
    action_repeat: int = 2

    # =====================================================================
    # Control rate
    # =====================================================================
    # Wall-clock ceiling on decisions per second. The bot cannot act faster
    # than it can see, and it cannot see faster than this.
    #
    # 20 rather than 30 because capture, not the model, sets the ceiling:
    # PrintWindow on a 1536x816 window measured 20-35 ms on the reference
    # machine, so a 30 fps budget leaves no headroom for the game. Run
    # `--benchmark` to see this machine's real capture cost and whether this
    # target fits; raise it if the benchmark says there is room.
    target_fps: float = 20.0
    # Extra sleep per step. 0 is right unless the game stutters under input.
    capture_delay: float = 0.0
    # How long a tap (a key press+release inside one step) is held down.
    tap_seconds: float = 0.05

    # =====================================================================
    # Model
    #
    # Deliberately small. The CNN turns one frame into a vector; a GRU carries
    # memory across steps. That is enough for pixel control and costs a
    # fraction of a transformer over a stack of frames, with no sequence-length
    # squared term anywhere.
    # =====================================================================
    embed_dim: int = 128
    hidden_dim: int = 192
    cnn_width: int = 32
    # Recurrent truncated-backprop length for PPO updates. Longer sees further
    # back but costs more per update; 16 is a good CPU compromise.
    seq_len: int = 16

    # =====================================================================
    # PPO
    # =====================================================================
    # Transitions collected per update. This is the single most important
    # number for whether the bot learns at all: an update computed from one
    # transition is pure noise (and normalising a single advantage to zero
    # makes the gradient exactly zero - a silent, permanent stall).
    rollout_steps: int = 1024
    minibatch_size: int = 256
    epochs_per_update: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    adam_lr: float = 3e-4
    adam_eps: float = 1e-5
    grad_clip: float = 0.5
    # Reward scaling: advantages are normalised, so this only sets the units
    # the value function has to track.
    reward_scale: float = 1.0

    # =====================================================================
    # Learning signal
    #
    # Four terms, all computed from pixels and the buttons pressed. No game
    # knowledge, no score reading, no manual feedback.
    # =====================================================================
    # Episodic novelty: reaching a state never reached since the last reset.
    # This is the term that rewards *progress*; everything else is support.
    # It is also the loudest term on purpose. For a policy to move, the reward
    # of one decision has to be distinguishable from the reward of another, and
    # a bonus that fires only on genuinely new ground is the sharpest such
    # signal available without knowing anything about the game.
    w_episodic: float = 3.0
    # A floor paid every step, scaled by how deep into the episode the bot has
    # got. This is what makes "press on" better than "stand still" rather than
    # merely different: without it, novelty fires once per new state and then
    # standing at the start is as good as standing at the frontier.
    # See novelty.IntrinsicReward.compute.
    w_depth: float = 1.0
    # Paid for the *increase* in depth on this step, and charged for a
    # decrease. This is dense shaping for the credit assignment the episodic
    # term is too sparse to provide on its own. Kept smaller than the episodic
    # weight: a large potential-style difference can cancel the sparse bonus
    # and leave the bot with no preference between exploring and re-treading.
    w_progress: float = 0.5
    # What a repeat visit to an already-seen state is worth, as a fraction of a
    # first visit. Well below 1 so exploring beats re-treading, and above 0 so
    # the reward does not vanish in a long episode.
    novelty_decay: float = 0.5
    # Random network distillation: prediction error of a fixed random network,
    # scaled by its own running spread. Broad, cheap curiosity that keeps the
    # bot moving in the very first steps, before it has discovered anything,
    # and fades on its own as states become familiar. Kept quieter than the
    # episodic term on purpose: raw RND error is largest where the screen is
    # most chaotic, so a loud version buys a bot that stares at whatever
    # flickers hardest.
    w_rnd: float = 1.0
    # A small charge for pressing nothing, growing while nothing is pressed, so
    # that inaction is never the free option. Without this the safest action is
    # noop and a policy can settle there permanently.
    w_idle: float = 0.02
    idle_ramp_steps: float = 25.0
    idle_ramp_max: float = 4.0
    # How many new states make a full-strength bonus. Bigger means the bonus
    # decays more slowly over a long episode, which is what keeps a reward
    # visible in a game where a single life lasts thousands of steps.
    novelty_bins: float = 32.0

    # =====================================================================
    # Reward normalisation
    #
    # Intrinsic rewards drift as the bot learns. Rather than hand-tune weights
    # against a moving signal, each term is divided by a running estimate of
    # its own scale, and the total is clipped. This is what keeps the policy
    # gradient meaningful on hour ten, not just hour one.
    # =====================================================================
    scale_floor: float = 1e-3
    reward_clip: float = 10.0

    # =====================================================================
    # Episode / reset detection
    #
    # There is no "game over" signal available, so resets are inferred. Two
    # observations do it:
    #   * a sudden large visual change that no action could plausibly explain
    #     (a death screen, a respawn, a loading screen), and
    #   * a return to the state seen right after the last reset.
    # Getting this roughly right matters: episodic novelty that never resets
    # turns into a one-off exploration bonus and then teaches nothing.
    # =====================================================================
    reset_jump: float = 0.35
    reset_revisit: float = 0.02
    # A reset candidate is ignored if it happens again within this many steps,
    # otherwise a flickering screen could reset every frame and pay forever.
    reset_cooldown: int = 8

    # =====================================================================
    # Liveness / stall guard
    #
    # This is the fix for "it stalls after a while". Three independent
    # detectors, each of which prints a specific diagnosis rather than leaving
    # the user watching a frozen log:
    #   * frozen screen  - the game stopped rendering or simulating
    #   * flat novelty   - the bot has saturated what it can reach
    #   * slow update    - the training side is eating the wall clock
    # =====================================================================
    frozen_warn_seconds: float = 120.0
    # Per-update wall-clock ceiling. Collection resumes even if the update has
    # minibatches left, so the loop always advances.
    update_seconds_budget: float = 6.0
    # Print a full status block this often (in environment steps).
    status_every: int = 2000
    # Warn when a PPO update's median duration exceeds this.
    slow_update_seconds: float = 3.0

    # =====================================================================
    # Novelty memory caps (all fixed-size; nothing here grows forever)
    # =====================================================================
    # Episodic memory: states seen since the last reset.
    episodic_capacity: int = 20000
    # Global visit counts across the whole run, capped LRU.
    global_capacity: int = 200000

    # =====================================================================
    # RND / forward model
    # =====================================================================
    rnd_dim: int = 128
    rnd_hidden: int = 256
    # Plain SGD on purpose: a two-layer MLP does not need an adaptive
    # optimiser, and this keeps its per-step cost and memory flat.
    rnd_lr: float = 0.02
    fwd_lr: float = 0.02

    # =====================================================================
    # Input budget
    #
    # A bot that can hold eleven keys at once will spend its life in menus. The
    # action space is therefore factored - which keys are held, how far to turn,
    # what to tap - and the number of simultaneously held keys is capped.
    # =====================================================================
    max_held_keys: int = 4
    mouse_turn_pixels: int = 20
    turn_levels: Tuple[Tuple[str, float], ...] = (
        ("fine", 0.4), ("normal", 1.0), ("fast", 2.5))
    # Virtual keys the recorder refuses to learn and the bot refuses to press.
    # F1/F3 are hardware/debug overlays in many games; unbinding them keeps the
    # bot from reaching into the game's own settings or its dev overlays. 0x70 =
    # F1, 0x72 = F3. Extend this if your game has other "do not touch" keys.
    blocked_vks: Tuple[int, ...] = (0x70, 0x72, 0x5B, 0x5C, 0x5D, 0x12, 0x09)

    # ---- keymap / calibration ---------------------------------------------------
    # The JSON whitelist of keys the bot may press. Written by --calibrate, read
    # by every other run. Delete it to fall back to the built-in defaults.
    keymap_path: str = "keymap.json"
    # The start/stop key --calibrate listens for.
    calibrate_key: str = "f8"
    # Presses shorter than this during --calibrate are treated as taps rather
    # than as a movement key being held.
    calibrate_min_hold: float = 0.12

    # =====================================================================
    # Checkpointing
    # =====================================================================
    checkpoint_dir: str = "checkpoints_v2"
    checkpoint_interval_sec: float = 300.0
    checkpoint_keep: int = 3
    resume: bool = True
    final_model_path: str = "gamebot.pt"

    # =====================================================================
    # Control / housekeeping
    # =====================================================================
    hotkey_pause: str = "f8"
    hotkey_save: str = "f9"
    hotkey_quit: str = "f10"
    enable_hotkeys: bool = True
    # Threads torch may use. Deliberately 1: this model is small enough that
    # the per-op thread hand-off costs more than the parallelism saves. It was
    # measured at 4.6 ms/step on one thread against 6.1 ms on four and 10.1 ms
    # on eight for a 48px observation on a 12-thread CPU - the more threads it
    # was given, the slower it got. Raise it only if the model is made much
    # bigger, and measure rather than assume.
    torch_threads: int = 1
    priority: str = "below_normal"   # below_normal | normal | high
    preview: bool = False
    show_model: bool = False
    log_json: Optional[str] = None
    seed: int = 0

    # ---- derived / validation ----
    def validate(self) -> "Config":
        """Fail loudly here rather than three hours into a run."""
        if self.frame_size < 40:
            raise ValueError(
                f"frame_size {self.frame_size} is too small; the encoder needs "
                f"at least 40 (96 is the default)")
        if self.frame_stack < 1:
            raise ValueError("frame_stack must be at least 1")
        if self.action_repeat < 1:
            raise ValueError("action_repeat must be at least 1")
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if self.rollout_steps < self.seq_len:
            raise ValueError(
                f"rollout_steps {self.rollout_steps} must be at least seq_len "
                f"{self.seq_len}")
        if self.minibatch_size < self.seq_len:
            self.minibatch_size = self.seq_len
        if self.max_held_keys < 1:
            raise ValueError("max_held_keys must be at least 1")
        if self.embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4")
        if self.hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4")
        return self

    @property
    def channels(self) -> int:
        """Input channels the CNN sees: the grey stack plus one difference."""
        return int(self.frame_stack) + 1

    def update_minibatches(self) -> int:
        """Minibatches per epoch, given the rollout and minibatch size."""
        n = max(1, int(self.rollout_steps))
        mb = max(self.seq_len, int(self.minibatch_size))
        return max(1, n // mb)

    def env_overrides(self) -> "Config":
        """Apply BOT_* environment overrides, for quick experiments."""
        for f in fields(self):
            name = f"BOT_{f.name.upper()}"
            if name in os.environ:
                setattr(self, f.name, _env(name, getattr(self, f.name)))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        return (
            f"obs {self.frame_stack}x{self.frame_size}x{self.frame_size} grey "
            f"(+diff), repeat {self.action_repeat}, "
            f"model {self.embed_dim}d/{self.hidden_dim}h, "
            f"rollout {self.rollout_steps} x {self.epochs_per_update} epochs "
            f"in {self.minibatch_size}-step minibatches"
        )


# The shape of the checkpoint format. Bumped whenever a saved run stops being
# loadable, so an incompatible file is refused instead of half-loaded.
CHECKPOINT_VERSION = 2
# Bumped whenever the reward function changes meaning. A checkpoint trained
# against a different objective is refused, because resuming it would silently
# continue a policy that was optimised for something else - which is exactly
# how a run ends up babysitting a no-op policy.
REWARD_VERSION = 2
# Bumped when the model architecture changes in a way that invalidates weights.
ARCH_NAME = "cnn-gru-v2"
