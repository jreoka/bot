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
paying again. Resets are inferred, not observed (see `ResetDetector`).

**RND novelty** - the prediction error of a fixed random network, scaled by its
own running spread. Broad, cheap curiosity that keeps the bot moving before it
has learned anything at all. It fades on its own as states become familiar.

**Learning progress** - how much a forward model's prediction of the next frame
just *improved*, not how wrong it is. This distinction is the whole point: raw
prediction error is highest where the screen is most chaotic, so paying for it
teaches the bot to find noise and stay there. Paying for improvement teaches it
to seek out the parts of the game it can actually learn to control, and the
reward stops once a region is understood.

**Anti-degeneracy** - a small charge for pressing nothing, growing the longer
nothing is pressed, plus a charge for repeating the same screen over and over
within a short window. Without the first, "noop" is the cheapest action and a
policy can settle there permanently. Without the second, the bot can find one
loud, chaotic spot and farm it forever while calling it novelty.

Every term is divided by a running estimate of its own scale, so the weights in
the config stay meaningful on hour ten and not just hour one. The total is
clipped, and every component is reported, so a run that has stopped learning
shows it in the log rather than only in the game.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional, Set, Tuple

import numpy as np
import torch

from .config import Config
from .model import ForwardModel, RandomNetworkDistillation

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
        # Typical forward-model error, kept here so the jump test can ask
        # "was this change surprising?" without reaching into the model.
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

        * **surprise** - the forward model's prediction of the next frame was
          far worse than it usually is, and something big changed. This is the
          most general signal: a death, a respawn, a level load or a menu all
          look like "the world did something I cannot account for".
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
    """

    def __init__(self, cfg: Config, device: torch.device,
                 action_vector_size: int):
        self.cfg = cfg
        self.device = device
        # How many new states per step counts as a full-strength novelty term.
        # A bigger number means the bonus takes longer to fall off, which is
        # what keeps a reward visible in a long episode.
        self.novelty_horizon = float(max(4.0, cfg.novelty_bins * 3.0))

        rnd_in = int(cfg.embed_dim) + int(action_vector_size)
        self.rnd = RandomNetworkDistillation(
            rnd_in, int(cfg.rnd_hidden), int(cfg.rnd_dim), float(cfg.rnd_lr)
        ).to(device)
        self.forward_model = ForwardModel(
            int(cfg.embed_dim), int(action_vector_size),
            int(cfg.rnd_hidden), float(cfg.fwd_lr)
        ).to(device)

        self.detector = ResetDetector(cfg)

        # Count-based memories, both strictly bounded.
        self.episodic: Dict[int, int] = {}
        self.global_visits: Set[int] = set()
        self._global_order: Deque[int] = deque()

        # Anti-degeneracy bookkeeping.
        self.idle_streak = 0
        self.recent: Deque[int] = deque(maxlen=40)

        self.prev_embed: Optional[torch.Tensor] = None
        self.prev_signature: Optional[np.ndarray] = None
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
        self.prev_embed = None
        if signature is not None:
            self.detector.note_anchor(signature)

    def note_human(self, signature: Optional[np.ndarray] = None) -> None:
        """A step the bot did not take (paused). Do not charge it to the bot."""
        self.idle_streak = 0
        self.prev_signature = None if signature is None else np.array(
            signature, dtype=np.float32, copy=True)

    # ---- main entry ----
    def compute(self, embed: torch.Tensor, signature: np.ndarray,
                action_vector: np.ndarray, engaged: bool,
                step: int) -> Tuple[float, bool, str]:
        """
        Reward one transition.

        ``embed`` is the frame embedding the policy already computed, so
        nothing here re-encodes the frame.  Returns (reward, is_reset, reason).
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
        novelty_reward = cfg.w_episodic * (cfg.novelty_decay ** min(visits, 64))

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
            progress_reward = cfg.w_episodic * cfg.w_progress * delta_depth
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

        # ---- reset detection ----
        self.detector.forward_model_reference = self.forward_model.scale.mean
        is_reset, reason = self.detector.check(
            step, signature, delta, engaged, novel_fraction,
            surprise=float(self.forward_model.last_error))

        # ---- novelty nets ----
        #
        # These run as forward passes only. Training them here would mean a
        # backward pass on every control step - which measured as about 40% of
        # the step cost - so the gradient work is batched into the PPO update
        # instead (see `train_nets`). The reward the policy sees is unchanged:
        # RND's error and the forward model's error do not depend on their own
        # current weights, so a forward pass now and a batch step later give
        # the same numbers.
        #
        # Everything is 1-D per sample, then batched to (1, D): one decision has
        # no batch to speak of, and keeping the shapes explicit avoids the
        # silent broadcasting bugs that hide in this kind of code.
        flat_embed = embed.reshape(-1)
        action_tensor = torch.as_tensor(action_vector, dtype=torch.float32,
                                        device=self.device)
        rnd_input = torch.cat([flat_embed, action_tensor]).unsqueeze(0)
        rnd_novelty = self.rnd.novelty(rnd_input, train=False)

        if self.prev_embed is not None:
            self.forward_model.error_only(
                self.prev_embed.unsqueeze(0), action_tensor.unsqueeze(0),
                flat_embed.unsqueeze(0))

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

        # Normalised novelty terms. The RND term is the secondary one: it keeps
        # the bot moving through the very early steps, before it has discovered
        # anything at all, and fades on its own as states become familiar. It
        # is deliberately quieter than the episodic term, because raw RND error
        # is largest where the screen is most chaotic, and a loud version of it
        # buys a bot that stares at whatever flickers hardest.
        rnd_term = min(cfg.w_rnd * rnd_novelty, 2.0)

        reward = (episodic + rnd_term - idle_cost - repetition_cost)

        self.last = {
            "episodic": float(episodic),
            "rnd": float(rnd_term),
            "idle": -float(idle_cost),
            "repeat": -float(repetition_cost),
            "total": float(reward),
        }
        for name, value in self.last.items():
            self.totals[name] = self.totals.get(name, 0.0) + value

        # ---- advance state ----
        self.prev_embed = flat_embed.detach()
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

    # ---- batched training, run during the PPO update ----
    def train_nets(self, embed: torch.Tensor, action: torch.Tensor,
                   next_embed: torch.Tensor, previous_embed: torch.Tensor,
                   minibatch: int = 256, steps: int = 1,
                   budget_seconds: float = 0.0) -> Dict[str, float]:
        """
        Train the RND predictor and the forward model on batches of the
        rollout just collected.

        Doing this here rather than on every control step is worth about 40% of
        the step cost, because it turns thousands of single-sample backward
        passes into a handful of batched ones.  The reward is unaffected: the
        error of these nets does not depend on their current weights, so a
        forward pass during collection and a batch step afterwards produce the
        same numbers.
        """
        if embed.shape[0] == 0:
            return {}
        started = time.time()
        n = int(embed.shape[0])
        rnd_in = torch.cat([embed, action], dim=-1)
        out: Dict[str, float] = {}
        order = np.arange(n)
        rng = np.random.default_rng(0)

        done = 0
        for _ in range(max(1, int(steps))):
            rng.shuffle(order)
            batch = max(1, int(minibatch))
            for start in range(0, n, batch):
                idx = torch.as_tensor(order[start:start + batch],
                                      dtype=torch.long, device=embed.device)
                out["rnd_loss"] = self.rnd.train_batch(rnd_in[idx])
                pairs = idx[(idx > 0)]
                if pairs.numel() > 0:
                    out["forward_loss"] = self.forward_model.train_batch(
                        previous_embed[pairs], action[pairs],
                        embed[pairs])
                done += 1
                if budget_seconds > 0 and time.time() - started >= budget_seconds:
                    out["batches"] = float(done)
                    return out
        out["batches"] = float(done)
        return out

    # ---- reporting / persistence ----
    @property
    def mean_reward(self) -> float:
        return self.totals.get("total", 0.0) / max(1, self.steps)

    def summary(self) -> str:
        n = max(1, self.steps)
        parts = (f"rwd/step {self.totals.get('total', 0.0) / n:+.4f} "
                 f"(epi {self.totals.get('episodic', 0.0) / n:+.4f}, "
                 f"rnd {self.totals.get('rnd', 0.0) / n:+.4f}, "
                 f"idle {self.totals.get('idle', 0.0) / n:+.4f})")
        return parts

    def state_dict(self) -> dict:
        return {
            "rnd_predictor": self.rnd.predictor.state_dict(),
            "rnd_target": self.rnd.target.state_dict(),
            "rnd_stats": self.rnd.scale.state(),
            "forward_net": self.forward_model.net.state_dict(),
            "forward_stats": self.forward_model.scale.state(),
            "ambient": self.detector.ambient,
            "steps": self.steps,
        }

    def load_state_dict(self, state: dict, strict: bool = False) -> bool:
        if not state:
            return False
        try:
            self.rnd.predictor.load_state_dict(state["rnd_predictor"])
            self.rnd.target.load_state_dict(state["rnd_target"])
            self.rnd.scale.load(state["rnd_stats"])
            self.forward_model.net.load_state_dict(state["forward_net"])
            self.forward_model.scale.load(state["forward_stats"])
            self.detector.ambient = float(state.get("ambient", 0.0))
            self.steps = int(state.get("steps", 0))
            return True
        except Exception as exc:
            if strict:
                raise
            print(f"[Reward] Intrinsic state not restored ({exc}); "
                  f"curiosity restarts from scratch.")
            return False
