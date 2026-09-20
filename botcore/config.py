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
    # grey frames is how the bot perceives motion without the model having to
    # infer it from a single image, and it costs nothing extra: the patch
    # embedding always runs on the same fixed number of channels.
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
    # One model: a causal SwiGLU + RoPE transformer over a rolling window of
    # frame representations.  Deliberately small, because it runs on a CPU
    # beside the game:
    #
    #   embed_dim        width of every token and of the memory window
    #   mem_tokens       how many past frames the window holds. This is the
    #                    transformer's entire memory, and it is the number that
    #                    trades context for per-step cost: attention is over
    #                    (mem_tokens + patch tokens), so doubling it roughly
    #                    triples the attention work (the SwiGLU blocks dominate
    #                    at these sizes, which is why a bigger window is
    #                    affordable at all).
    #   patch_size       pixels per patch token; patch_width is the width of the
    #                    single convolution that turns patches into tokens.
    #                    The transformer does the sequence work - there is no
    #                    CNN stack in front of it.
    #   ffn_hidden       SwiGLU hidden width (2 * embed_dim is the usual ratio)
    # =====================================================================
    embed_dim: int = 96
    mem_tokens: int = 16
    patch_size: int = 16
    patch_width: int = 64
    transformer_layers: int = 2
    ffn_hidden: int = 192
    attention_heads: int = 4
    attention_dropout: float = 0.0
    # Sequence length used to truncate the PPO update. The memory window is
    # rebuilt inside the update from stored summaries, and gradients are cut at
    # the sequence boundary; 16 is the usual CPU compromise between seeing
    # further back and costing more per update.
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
    # Four groups of terms, all computed from pixels and the buttons pressed.
    # No game knowledge, no score reading, no manual feedback.
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
    w_depth_progress: float = 0.5
    # What a repeat visit to an already-seen state is worth, as a fraction of a
    # first visit. Well below 1 so exploring beats re-treading, and above 0 so
    # the reward does not vanish in a long episode.
    novelty_decay: float = 0.5
    # How many steps a fresh episode must last before the per-state novelty
    # bonus is paid in full. An inferred reset clears the episodic memory, so
    # without this a policy that dies every other step is paid for "new" states
    # every other step - the reward farms resets, and the bot learns to die.
    # Ramping it in makes a two-step life worth almost nothing.
    novelty_survival_steps: float = 12.0
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
    # The transformer's own curiosity head
    #
    # The model predicts the next frame's representation; its error is the
    # novelty signal the reward is built from, so no second network exists.
    # `transition_hidden` is the width of the small MLP that does the
    # predicting, and `transition_lr` is its own learning rate (it is trained by
    # its own optimiser so a curiosity gradient never moves the policy's trunk).
    # =====================================================================
    transition_hidden: int = 192
    transition_lr: float = 3e-4
    # Curiosity is paid for prediction error in units of how much that error
    # usually varies, so the weight means the same thing on hour ten as on hour
    # one - and so a bot sitting in front of a noisy screen is not paid more
    # than one making progress. `w_novelty_cap` stops a single chaotic frame
    # from dominating a rollout.
    #
    # There is no separate "learning progress" weight: a term that pays for the
    # prediction error falling was measured to invert the reward on the
    # synthetic corridor (standing still scored 61% of walking forward), so the
    # count-based episodic term is what carries progress instead.
    w_novelty: float = 1.0
    w_novelty_cap: float = 2.0
    # How fast the running estimate of "ordinary prediction error" follows the
    # signal. It must track the model's own improvement - as the predictor gets
    # better everywhere, what counts as surprising falls with it - without
    # becoming so fast that a single chaotic frame redefines "ordinary".
    error_decay: float = 0.99

    # =====================================================================
    # Input budget
    #
    # A bot that can hold eleven keys at once will spend its life in menus. The
    # action space is therefore factored - which keys are held, which way to
    # turn, how fast to turn, what to tap - and the number of simultaneously
    # held keys is capped.
    # =====================================================================
    max_held_keys: int = 4

    # ---- mouse speed: the bot sets its own ----
    #
    # The view-depth decision is two things a player decides separately: which
    # way to swing, and how far. "How far" is a *speed*, and the bot owns it:
    #
    #   * the policy has a speed head, so the multiplier on any turn is a
    #     decision it makes and PPO trains like any other;
    #   * `mouse_turn_start` is the base step the multiplier applies to, in
    #     pixels. `mouse_turn_pixels` pins it when the user knows better
    #     (--mouse-turn, or a --calibrate recording), otherwise --calibrate's
    #     measurement is used, and the bot is free to move the base within
    #     [mouse_turn_min, mouse_turn_max] as it learns what this game's view
    #     actually needs.
    #
    # That range is what makes a cursor-locked game survivable: such a title
    # hides the real look speed from the recorder, so the measured step comes
    # out at a pixel or two, and a bot that could only ever use the measurement
    # could never turn its view at all.
    mouse_turn_pixels: int = 0
    mouse_turn_start: int = 20
    mouse_turn_min: int = 4
    mouse_turn_max: int = 120
    # The multipliers the speed head chooses between, applied to the base step.
    # The top entry has to be large enough to be a real spin on the spot and the
    # bottom one small enough to be an aim correction; what sits between them is
    # what the bot uses to match a game it has never seen.
    speed_levels: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    # How hard the bot is allowed to correct its own base step when the view
    # turns by far less (or far more) than the pixels it sent. 0 disables the
    # correction and leaves the base where it started.
    mouse_adapt_rate: float = 0.08

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
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval_sec: float = 300.0
    checkpoint_keep: int = 3
    resume: bool = True
    final_model_path: str = "gamebot.pt"

    # =====================================================================
    # Control / housekeeping
    # =====================================================================
    # F8 starts the run and then pauses/resumes it. Starting on a key rather
    # than at launch is deliberate: the bot must not send a single keystroke
    # until the user has clicked the game, or the first keys land in whatever
    # window had focus when the script was launched - usually the terminal.
    hotkey_pause: str = "f8"
    hotkey_save: str = "f9"
    hotkey_quit: str = "f10"
    enable_hotkeys: bool = True
    # True: wait for the start key (F8) before touching the game. False:
    # start injecting immediately (--start-now).
    wait_for_start: bool = True
    # How hard the bot may fight for the foreground window:
    #   "once"   - bring the game forward when a run starts or resumes, then
    #              leave the desktop alone. If the game loses focus, input
    #              stops until it comes back, so keys never land elsewhere.
    #   "always" - re-assert focus on every action. This is what makes the
    #              terminal and the game wrestle for the keyboard; kept only
    #              for games that will not accept input any other way.
    #   "never"  - never touch focus; the user brings the game forward.
    focus_policy: str = "once"
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
                f"frame_size {self.frame_size} is too small; the patch "
                f"embedding needs at least 40 (96 is the default)")
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
        if self.embed_dim < 8:
            raise ValueError("embed_dim must be at least 8")
        if self.embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4")
        if self.mem_tokens < 1:
            raise ValueError("mem_tokens must be at least 1")
        if self.patch_size < 2:
            raise ValueError("patch_size must be at least 2")
        if self.transformer_layers < 1:
            raise ValueError("transformer_layers must be at least 1")
        if self.ffn_hidden < self.embed_dim:
            raise ValueError("ffn_hidden must be at least embed_dim")
        if self.attention_heads < 1:
            raise ValueError("attention_heads must be at least 1")
        if self.embed_dim % self.attention_heads != 0:
            raise ValueError(
                f"embed_dim {self.embed_dim} must be divisible by "
                f"attention_heads {self.attention_heads}")
        if (self.embed_dim // self.attention_heads) % 2 != 0:
            raise ValueError(
                f"embed_dim / attention_heads must be even for rotary "
                f"position embedding (got "
                f"{self.embed_dim // self.attention_heads})")
        if not self.speed_levels:
            raise ValueError("speed_levels must list at least one multiplier")
        if min(float(s) for s in self.speed_levels) <= 0.0:
            raise ValueError("speed_levels must all be positive")
        if self.mouse_turn_max < self.mouse_turn_min:
            raise ValueError("mouse_turn_max must be >= mouse_turn_min")
        if not (self.mouse_turn_min <= self.mouse_turn_start
                <= self.mouse_turn_max):
            raise ValueError(
                f"mouse_turn_start {self.mouse_turn_start} must lie between "
                f"mouse_turn_min {self.mouse_turn_min} and mouse_turn_max "
                f"{self.mouse_turn_max}")
        if self.focus_policy not in ("once", "always", "never"):
            raise ValueError(
                f"focus_policy '{self.focus_policy}' is not one of "
                f"'once', 'always', 'never'")
        return self

    @property
    def channels(self) -> int:
        """Input channels the model sees: the grey stack plus one difference."""
        return int(self.frame_stack) + 1

    def frame_tokens(self) -> int:
        """Patch tokens one frame becomes, given frame_size and patch_size."""
        step = max(2, int(self.patch_size))
        grid = max(1, (int(self.frame_size) + step - 1) // step)
        return grid * grid

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
        speeds = "/".join(f"{float(s):g}" for s in self.speed_levels)
        return (
            f"obs {self.frame_stack}x{self.frame_size}x{self.frame_size} grey "
            f"(+diff), repeat {self.action_repeat}, "
            f"swiglu+rope transformer {self.embed_dim}d x "
            f"{self.transformer_layers}L x {self.attention_heads}H "
            f"(ffn {self.ffn_hidden}, window {self.mem_tokens} frames, "
            f"{self.frame_tokens()} patch token(s)/frame), "
            f"turn speed x[{speeds}] from a {self.mouse_turn_start}px base, "
            f"rollout {self.rollout_steps} x {self.epochs_per_update} epochs "
            f"in {self.minibatch_size}-step minibatches"
        )


# The shape of the checkpoint format. Bumped whenever a saved run stops being
# loadable, so an incompatible file is refused instead of half-loaded.
CHECKPOINT_VERSION = 3
# Bumped whenever the reward function changes meaning. A checkpoint trained
# against a different objective is refused, because resuming it would silently
# continue a policy that was optimised for something else - which is exactly
# how a run ends up babysitting a no-op policy.
REWARD_VERSION = 3
# Bumped when the model architecture changes in a way that invalidates weights.
ARCH_NAME = "swiglu-rope-transformer-v1"
