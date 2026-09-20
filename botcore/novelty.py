"""
The learning signal, built entirely from pixels and buttons.

There is no game knowledge anywhere in here. Everything below is computed from
what the bot can see and what it pressed, which is what makes the same reward
work on any game.

What it pays for, and why each term exists
------------------------------------------

**Episodic novelty** - reaching a state that has not been reached since the
last reset. This is the term that means *progress*. "Go somewhere you have
never been since this attempt started" is the generic description of doing well
in a game, and unlike a global novelty bonus it does not decay to nothing once
the whole map has been seen: it goes back to zero at every reset and starts
paying again. Resets are inferred, not observed (see `ResetDetector`). The
per-state bonus is ramped in over the first few steps of an episode, because a
reset clears the memory: without that ramp a policy which dies every other step
is paid for "new" states every other step, which is a reward for dying.

**Prediction novelty** - the transformer's own next-frame prediction error, in
units of how much that error usually varies for this game. What the bot cannot
yet predict is, by definition, what it has not understood; the term is broad and
cheap, it keeps the bot moving in the very first steps, and it fades on its own
as the predictor is taught the states the policy visits. It is capped, because
uncapped prediction error is largest where the screen is most chaotic and a loud
version of it buys a bot that stares at whatever flickers hardest.

**Anti-degeneracy** - a small charge for pressing nothing, growing the longer
nothing is pressed, plus a charge for repeating the same screen over and over
within a short window. Without the first, "noop" is the cheapest action and a
policy can settle there permanently. Without the second, the bot can find one
loud, chaotic spot and farm it forever while calling it novelty.

What is deliberately *not* here
-------------------------------
A **learning-progress** term - paying for the prediction error *falling* - has
been tried and removed. It sounds right and it inverts the reward: against fixed
policies on the synthetic corridor, "never press anything" next to one static
screen produced the largest sustained drop in error and earned 61% of what
walking forward earned. Curiosity that only pays while it is being reduced
rewards finding somewhere quiet to sit. The count-based episodic term, which
pays for *being somewhere new since the last reset*, is what carries progress.

Every term is divided by a running estimate of its own scale, so the weights in
the config stay meaningful on hour ten and not just hour one. The total is
clipped, and every component is reported, so a run that has stopped learning
shows it in the log rather than only in the game.

Where the prediction comes from
-------------------------------
The signal used to be two side networks - a random-target distillation pair and
a separate forward model. Both are gone. The transformer already has a head that
predicts the next frame's representation (see `ActorCritic.predict_next`), and
its error is exactly the quantity those networks were approximating, so the
reward reads that instead. Nothing here owns a network any more: the only model
in the process is the policy, and `Trainer.train_transition` is what teaches it.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Set, Tuple

import numpy as np
import torch

from .config import Config
from .model import RunningScale

# Cells in the coarse fingerprint used for novelty and reset detection.
# 256 cells at 16 levels is deliberately the resolution at which "the same
# place, a slightly different view" is still the same key: too coarse and the
# episodic bonus cannot tell two rooms apart, too fine and every frame of
# camera noise looks like a brand new situation.
GRID = 256
LEVELS = 16


def _signature_key(signature: np.ndarray) -> int:
    """
    Hash a frame fingerprint to an integer.

    The resolution here sets what "a new situation" means, which is the single
    most consequential choice in the whole reward. 256 cells at 16 levels is
    fine enough that walking from one end of a corridor to the other is a
    hundred new states rather than one, and coarse enough that a flickering
    texture is not.
    """
    grid = signature.reshape(GRID, signature.size // GRID).mean(axis=1)
    quantised = np.clip((grid * LEVELS).astype(np.int64), 0, LEVELS - 1)
    bits = np.stack([(quantised >> shift) & 1 for shift in (3, 2, 1, 0)],
                    axis=1).reshape(-1)
    return hash(np.packbits(bits).tobytes()) & 0xFFFFFFFFFFFFFFFF


class ResetDetector:
    """
    Infers "the game just started over" from pixels alone.

    No game exposes this in a way a generic bot can read, so it is inferred
    from two independent observations:

    * **A jump** - the frame changed far more than the ambient frame-to-frame
      motion right after an action that did not touch anything. A death screen,
      a respawn, a level transition. The ambient estimate is what makes this
      robust: it compares the change against how much this game normally moves,
      so a fast-moving game does not look like it is resetting constantly.
    * **A return** - the current view matches the view the bot had right after
      the last reset, closely. Respawns tend to put you back where you began.

    A cooldown stops a flickering screen from resetting on every frame, which
    would pay the episodic bonus forever.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ambient = 0.0
        self.ambient_samples = 0
        self.anchor: Optional[np.ndarray] = None
        self.last_reset_step = -10 ** 9
        self.resets = 0
        self.last_reason = ""
        # Typical next-frame prediction error, kept here so the jump test can
        # ask "was this change surprising?" without reaching into the model.
        self.forward_model_reference = 1.0

    def note_anchor(self, signature: np.ndarray) -> None:
        """Record what the world looks like at the start of an episode."""
        self.anchor = np.array(signature, dtype=np.float32, copy=True)

    def update_ambient(self, delta: float, engaged: bool) -> None:
        """
        Learn how much this game moves on its own.

        Only steps where the bot pressed nothing feed this estimate, because
        those are the only steps where the change is known not to be caused by
        the bot.
        """
        if engaged:
            return
        self.ambient_samples += 1
        # A slow exponential average: the point is a rough baseline of normal
        # motion, not a precise statistic.
        self.ambient += 0.02 * (delta - self.ambient)

    def check(self, step: int, signature: np.ndarray, delta: float,
              engaged: bool, novel_fraction: float,
              surprise: float = 0.0) -> Tuple[bool, str]:
        """
        Return (is_reset, reason).

        Three independent tests, because no single one is reliable in a game
        nobody has described:

        * **surprise** - the model's prediction of the next frame was far worse
          than it usually is, and something big changed. This is the most
          general signal: a death, a respawn, a level load or a menu all look
          like "the world did something I cannot account for".
        * **jump** - a large change the bot did not cause, judged against how
          much this game normally moves when nothing is pressed.
        * **revisit** - the bot is back where this episode started and has
          found nothing new, which is what a respawn looks like.

        Getting this right is not cosmetic. If a death is not recognised as a
        reset, the novelty memory is never cleared, and dying becomes just
        another source of new states - so the bot learns to die, which is
        exactly the failure this detector exists to prevent.
        """
        if step - self.last_reset_step < int(self.cfg.reset_cooldown):
            return False, ""

        reference = max(1e-6, self.forward_model_reference)

        # A big, unexplained change in the world.
        startled = (delta > self.cfg.reset_jump
                    and surprise > 2.5 * reference)
        if startled:
            self._fire(step)
            return True, "surprise" if engaged else "jump"

        # A large change the bot did not cause, versus how much this game
        # normally moves on its own.
        if not engaged and delta > self.cfg.reset_jump \
                and delta > 3.0 * max(self.ambient, 1e-3):
            self._fire(step)
            return True, "jump"

        # Back to where this episode started, and nowhere new since.
        if self.anchor is not None and novel_fraction < 0.02:
            distance = float(np.abs(signature - self.anchor).mean())
            if distance < self.cfg.reset_revisit:
                self._fire(step)
                return True, "revisit"

        return False, ""

    def _fire(self, step: int) -> None:
        self.last_reset_step = int(step)
        self.resets += 1


class IntrinsicReward:
    """
    Computes the reward for one transition, plus the component breakdown the
    status line reports.

    It owns no network. The prediction the curiosity terms are built from comes
    from the policy's own next-frame head, handed in through
    `Trainer.act`/`RolloutBuffer`; the only state kept here is statistics about
    how large that error usually is.
    """

    def __init__(self, cfg: Config, device: torch.device,
                 action_vector_size: int):
        self.cfg = cfg
        self.device = device
        # How many new states per step counts as a full-strength novelty term.
        # A bigger number means the bonus takes longer to fall off, which is
        # what keeps a reward visible in a long episode.
        self.novelty_horizon = float(max(4.0, cfg.novelty_bins * 3.0))
        # How long the bot has to survive before the per-state novelty bonus is
        # paid in full. See `compute` - this is what stops a policy from farming
        # resets by dying.
        self.survival_steps = float(max(1.0, cfg.novelty_survival_steps))
        # `action_vector_size` is kept in the signature: the action description
        # is still what the rollout carries, and the transition head is free to
        # start consuming it again without every caller changing.
        self.action_vector_size = int(action_vector_size)

        # Streaming estimate of "ordinary" next-frame prediction error for this
        # game. Both curiosity terms are measured against it, which is what
        # makes their weights mean the same thing at hour ten as at hour one.
        self.error_running = RunningScale()

        self.detector = ResetDetector(cfg)

        # Count-based memories, both strictly bounded.
        self.episodic: Dict[int, int] = {}
        self.global_visits: Set[int] = set()
        self._global_order: Deque[int] = deque()

        # Anti-degeneracy bookkeeping.
        self.idle_streak = 0
        self.recent: Deque[int] = deque(maxlen=40)

        self.prev_signature: Optional[np.ndarray] = None
        # The previous step's frame representation and the model's prediction of
        # it, kept so this step can score how wrong that prediction was.
        self.prev_summary: Optional[np.ndarray] = None
        self.prev_prediction: Optional[np.ndarray] = None
        self.episode_steps = 0
        self.episode_novel = 0
        # Smoothed, capped depth. Kept separate from the raw count so the
        # reward cannot grow without bound through a long episode.
        self.depth_ema = 0.0
        self.prev_depth: Optional[float] = None
        # How fast the depth estimate follows the count of new states. Higher
        # means a single step that reaches new ground stands out more.
        self.depth_rate = 0.25

        self.last: Dict[str, float] = {}
        self.totals: Dict[str, float] = {}
        self.steps = 0

    def reset_episode(self, signature: Optional[np.ndarray] = None) -> None:
        """
        Start a new episode: forget what has been visited since the last one,
        and re-anchor the reset detector on the new starting view.

        The global visit counts deliberately survive. They describe the whole
        run, and re-paying for the same room every episode would teach the bot
        to go in circles.
        """
        self.episodic.clear()
        self.episode_steps = 0
        self.episode_novel = 0
        self.depth_ema = 0.0
        self.prev_depth = None
        self.idle_streak = 0
        self.recent.clear()
        self.prev_summary = None
        self.prev_prediction = None
        if signature is not None:
            self.detector.note_anchor(signature)

    def note_human(self, signature: Optional[np.ndarray] = None) -> None:
        """A step the bot did not take (paused). Do not charge it to the bot."""
        self.idle_streak = 0
        self.prev_signature = None if signature is None else np.array(
            signature, dtype=np.float32, copy=True)

    # ---- main entry ----
    def compute(self, summary: Optional[np.ndarray], signature: np.ndarray,
                action_vector: np.ndarray, engaged: bool, step: int,
                prediction: Optional[np.ndarray] = None,
                has_prediction: bool = True) -> Tuple[float, bool, str]:
        """
        Reward one transition.

        ``summary`` is this frame's representation and ``prediction`` is what the
        transformer expected it to be (both produced by the single forward pass
        the policy already ran, so nothing here re-encodes the frame).
        ``has_prediction`` is False when no predecessor made one - the first
        decision of a run, or the first after the window was reset - because a
        prediction of zero is not a prediction of nothing.  Returns
        (reward, is_reset, reason).
        """
        cfg = self.cfg
        self.steps += 1
        self.episode_steps += 1

        delta = 0.0
        if self.prev_signature is not None:
            delta = float(np.abs(signature - self.prev_signature).mean())
        self.detector.update_ambient(delta, engaged)

        key = _signature_key(signature)
        self.recent.append(key)

        # ---- episodic novelty: the term that means progress ----
        #
        # This is the load-bearing part of the whole reward, so it is split
        # into the two questions a policy actually asks.
        #
        # "is being here good?" - a floor that grows with how deep into the
        # episode the bot has got. It is paid every step, so the state the bot
        # reaches after discovering twenty new situations is *continuously*
        # worth more than the one it started in. Without it, the signal fires
        # once per new state and then a policy that has seen everything nearby
        # has no reason to prefer being anywhere, which is the shape that makes
        # "stand still" a fixed point.
        #
        # "is this new?" - a bonus that decays with every repeat visit this
        # episode, so exploring pays and re-treading old ground does not.
        visits = self.episodic.get(key, 0)
        self.episodic[key] = visits + 1
        if visits == 0:
            self.episode_novel += 1

        # Depth is capped at one "horizon" worth of new states. It is a running
        # estimate rather than an exact count, which is what keeps it bounded:
        # an uncapped counter grows without limit over a long episode, and a
        # reward that keeps growing is one no value function can ever fit.
        # The smoothing is deliberately quick, so that a step which reaches new
        # ground is visibly different from one that does not. A slow average
        # spreads that difference over many steps, and a signal smeared over
        # many steps gives the policy nothing to attribute to a single choice.
        self.depth_ema += self.depth_rate * (min(self.episode_novel,
                                                 self.novelty_horizon)
                                             - self.depth_ema)
        depth = self.depth_ema / self.novelty_horizon
        depth_reward = cfg.w_episodic * cfg.w_depth * depth
        # The per-state bonus is gated on having survived a little while. This
        # is the one piece of the reward that a bot can farm by dying: an
        # inferred reset clears the episodic memory, so a policy that dies every
        # other step sees "new" states every other step and is paid for them,
        # which is exactly how a learner ends up killing itself on purpose.
        # Ramping the bonus in over the first few steps of an episode makes a
        # two-step life worth almost nothing while leaving a sustained
        # exploration of the same ground worth full price - measured on the
        # synthetic corridor, this is what keeps "walk backwards and die" from
        # scoring 42% of "walk forwards".
        survival = min(1.0, self.episode_steps / self.survival_steps)
        novelty_reward = (cfg.w_episodic * survival
                          * (cfg.novelty_decay ** min(visits, 64)))

        # Progress towards more depth, paid only for the *increase*.
        #
        # This is what makes a single decision attributable. Novelty alone is a
        # function of the whole frame history, so in a fast-moving game nearly
        # every step counts as novel and every action looks equally good - the
        # advantage differences that reach the policy are then indistinguishable
        # from noise, and the policy never moves. Paying for the change in a
        # potential fixes that: an action that increases depth earns, and an
        # action that decreases it costs, on the spot.
        progress_reward = 0.0
        if self.prev_depth is not None:
            delta_depth = (self.depth_ema - self.prev_depth) / self.novelty_horizon
            progress_reward = cfg.w_episodic * cfg.w_depth_progress * delta_depth
        self.prev_depth = self.depth_ema
        episodic = depth_reward + novelty_reward + progress_reward
        novel_fraction = (self.episode_novel / float(self.episode_steps)
                          if self.episode_steps > 0 else 0.0)

        # ---- global counts (diagnostics + a slow, broad bonus) ----
        if key not in self.global_visits:
            self.global_visits.add(key)
            self._global_order.append(key)
            if len(self._global_order) > int(cfg.global_capacity):
                self._evict_global()

        if len(self.episodic) > int(cfg.episodic_capacity):
            # Bounded memory: keep the most recent half. Cheap, and it only
            # happens after a very long single episode.
            for old in list(self.episodic.keys())[:len(self.episodic) // 2]:
                self.episodic.pop(old, None)

        # ---- the model's own prediction of this frame ----
        #
        # `prediction` is what the transformer expected this frame's
        # representation to be, made one step earlier. The error against what it
        # actually is, in units of how much that error usually varies, is the
        # whole curiosity signal - one number, from the one model, with no side
        # network to train or store.
        #
        # On the very first step of an episode there is no prediction yet (the
        # prediction is stored on the step that makes it), so the term is simply
        # not paid rather than paid on a stale pair.
        error = 0.0
        if has_prediction and prediction is not None and summary is not None:
            predicted = torch.as_tensor(prediction, dtype=torch.float32,
                                        device=self.device).reshape(-1)
            actual = torch.as_tensor(summary, dtype=torch.float32,
                                     device=self.device).reshape(-1)
            if predicted.numel() == actual.numel():
                error = float(torch.mean((predicted - actual) ** 2).item())
        if not np.isfinite(error):
            error = self.error_running.mean
        # Surprise is measured on the *spread* of the error, not its level, and
        # the estimate is updated before the reward reads it. That matters: a
        # running mean of a nearly constant error sits almost on top of it, so
        # "worse than usual" and "better than usual" both come out as rounding
        # noise, and the bot is paid for the level of chaos on its screen
        # instead of for the surprise of it. `RunningScale` gives a spread with
        # a floor that cannot collapse, so a novelty bonus starts near 1.0 and
        # stays comparable for the life of the run.
        self.error_running.update(error)
        spread = max(1e-6, float(self.error_running.std))

        # ---- reset detection ----
        #
        # `surprise` is the prediction error on the scale the reward uses it on,
        # so "far worse than this game usually is" is a fixed multiple of the
        # running spread rather than a number tuned per game.
        surprise = error / spread
        self.detector.forward_model_reference = float(self.error_running.mean)
        is_reset, reason = self.detector.check(
            step, signature, delta, engaged, novel_fraction,
            surprise=float(surprise))

        # ---- anti-degeneracy ----
        if engaged:
            self.idle_streak = 0
        else:
            self.idle_streak += 1
        ramp = min(1.0 + self.idle_streak / max(cfg.idle_ramp_steps, 1.0),
                   cfg.idle_ramp_max)
        idle_cost = cfg.w_idle * ramp if not engaged else 0.0

        # Repeating one screen for a long stretch is not novelty, it is a loop.
        repeats = sum(1 for k in self.recent if k == key)
        repeat_ratio = repeats / max(1, len(self.recent))
        repetition_cost = 0.0
        if repeat_ratio > 0.75 and len(self.recent) >= self.recent.maxlen:
            repetition_cost = cfg.w_idle * 2.0 * (repeat_ratio - 0.75) / 0.25

        # ---- prediction novelty ----
        #
        # The transformer's own next-frame error, in units of how much that
        # error usually varies. Around 1.0 means "as surprising as this game
        # normally is"; more than that means the bot is somewhere it does not
        # yet understand, and the term fades on its own because the predictor
        # keeps being taught the states the policy visits. It is capped, because
        # uncapped prediction error is largest where the screen is most chaotic
        # and a loud version of it buys a bot that stares at whatever flickers
        # hardest.
        #
        # There is deliberately no "learning progress" term on top of this any
        # more. The version that paid for the error *falling* was measured
        # against fixed policies on the synthetic corridor and it inverted the
        # reward: standing still next to one static screen produced the largest
        # sustained drop, so "never press anything" earned 61% of what walking
        # forward earned. Curiosity that only pays while it is being reduced
        # rewards finding somewhere quiet to sit, and the count-based episodic
        # term is what actually distinguishes progress.
        novelty_term = min(cfg.w_novelty * (error / spread), cfg.w_novelty_cap)

        reward = (episodic + novelty_term - idle_cost - repetition_cost)

        self.last = {
            "episodic": float(episodic),
            "novelty": float(novelty_term),
            "error": float(error),
            "idle": -float(idle_cost),
            "repeat": -float(repetition_cost),
            "total": float(reward),
        }
        for name, value in self.last.items():
            self.totals[name] = self.totals.get(name, 0.0) + value

        # ---- advance state ----
        self.prev_summary = (None if summary is None
                             else np.array(summary, dtype=np.float32,
                                           copy=True))
        self.prev_prediction = (None if prediction is None
                                else np.array(prediction, dtype=np.float32,
                                              copy=True))
        self.prev_signature = np.array(signature, dtype=np.float32, copy=True)

        if is_reset:
            self.reset_episode(signature)
            self.detector.last_reason = reason

        return float(np.clip(reward, -cfg.reward_clip, cfg.reward_clip)), \
            is_reset, reason

    def _evict_global(self) -> None:
        """Drop the oldest quarter of the global visit set, in one pass."""
        drop = max(1, len(self._global_order) // 4)
        for _ in range(drop):
            if not self._global_order:
                break
            self.global_visits.discard(self._global_order.popleft())

    # ---- reporting / persistence ----
    @property
    def mean_reward(self) -> float:
        return self.totals.get("total", 0.0) / max(1, self.steps)

    def summary(self) -> str:
        n = max(1, self.steps)
        return (f"rwd/step {self.totals.get('total', 0.0) / n:+.4f} "
                f"(epi {self.totals.get('episodic', 0.0) / n:+.4f}, "
                f"nov {self.totals.get('novelty', 0.0) / n:+.4f}, "
                f"idle {self.totals.get('idle', 0.0) / n:+.4f})")

    def state_dict(self) -> dict:
        return {
            "error_stats": self.error_running.state(),
            "ambient": self.detector.ambient,
            "steps": self.steps,
        }

    def load_state_dict(self, state: dict, strict: bool = False) -> bool:
        if not state:
            return False
        try:
            if "error_stats" in state:
                self.error_running.load(state["error_stats"])
            self.detector.ambient = float(state.get("ambient", 0.0))
            self.steps = int(state.get("steps", 0))
            return True
        except Exception as exc:
            if strict:
                raise
            print(f"[Reward] Intrinsic state not restored ({exc}); "
                  f"curiosity restarts from scratch.")
            return False
