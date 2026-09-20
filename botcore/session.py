"""
The session: one game, one policy, and the loop that runs them.

This is the piece that decides whether the bot keeps making progress.  It owns
the outer loop and, critically, every health check in it. A run that has
stopped learning should say so in the log, with a specific reason, rather than
leaving the user to notice that the screen has been the same for an hour.

Health checks, all of which report a diagnosis rather than just a number:

* **frozen window** - the game stopped rendering or simulating (unfocused
  pause, a crashed renderer, a still loading screen). Detected from the frame
  difference and reported once.
* **flat novelty** - the bot is no longer reaching states it has not reached
  before. Either it has saturated what it can do, or it has collapsed onto a
  loop. Both are diagnosed explicitly because they need different responses.
* **no-op collapse** - the policy stopped pressing anything. This is the
  classic silent failure of intrinsic rewards, and it is visible in the action
  histogram before it is visible in the game.
* **slow update** - the training half is eating the wall clock, which is what
  makes a run degrade over hours as the machine's state changes.

A session whose `step()` is driven from outside (see the self-test) gets the
same machinery, so the thing that is verified in `--selftest` is the thing that
runs against a real game.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from .capture import gray_signature
from .config import ARCH_NAME, CHECKPOINT_VERSION, REWARD_VERSION, Config
from .keys import ActionSpace, Keymap
from .model import ActorCritic
from .novelty import IntrinsicReward
from .replay import RolloutBuffer
from .trainer import Trainer


class HealthMonitor:
    """Tracks the slow-moving quantities that reveal a run going wrong."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.window = 4000
        self.novel_steps = 0
        self.novelty_series: Deque[float] = deque(maxlen=8)
        self.engaged_series: Deque[float] = deque(maxlen=8)
        self.update_series: Deque[float] = deque(maxlen=8)
        self.warnings: List[str] = []
        self.frozen_since: Optional[float] = None
        self.frozen_reported = False
        self.last_delta = 0.0
        self.noop_streak = 0
        self.longest_noop_streak = 0

    def note_step(self, delta: float, engaged: bool,
                  episodic_bonus: float) -> Optional[str]:
        """Feed one step. Returns a message the first time a problem appears."""
        self.last_delta = float(delta)
        if episodic_bonus > 0.0:
            self.novel_steps += 1
        if engaged:
            self.noop_streak = 0
        else:
            self.noop_streak += 1
            self.longest_noop_streak = max(self.longest_noop_streak,
                                           self.noop_streak)

        message = None
        if self.cfg.frozen_warn_seconds > 0:
            if delta < 1e-4:
                if self.frozen_since is None:
                    self.frozen_since = time.monotonic()
                elif (not self.frozen_reported and
                      time.monotonic() - self.frozen_since
                      >= self.cfg.frozen_warn_seconds):
                    self.frozen_reported = True
                    message = (
                        f"[Health] The screen has not changed for "
                        f"{self.cfg.frozen_warn_seconds:.0f}s. The game has "
                        f"almost certainly stopped rendering or simulating "
                        f"while unfocused, so there is nothing to learn from. "
                        f"Fix it in the game (disable 'pause when unfocused', "
                        f"enable background running, use borderless windowed) "
                        f"or the window is minimized.")
            else:
                self.frozen_since = None
                self.frozen_reported = False
        return message

    def end_window(self, steps: int, engaged_steps: int,
                   update_seconds: float) -> List[str]:
        """Called when the status block is printed; returns diagnoses."""
        messages: List[str] = []
        if steps <= 0:
            return messages
        novelty_rate = 1000.0 * self.novel_steps / steps
        engaged_rate = 100.0 * engaged_steps / steps
        self.novelty_series.append(novelty_rate)
        self.engaged_series.append(engaged_rate)
        self.update_series.append(update_seconds)

        if len(self.novelty_series) >= 4:
            recent = list(self.novelty_series)[-2:]
            earlier = list(self.novelty_series)[:-2]
            if max(recent) < 1.0 and np.mean(earlier) > 5.0:
                messages.append(
                    "[Health] New-state rate has flattened: the bot is no "
                    "longer reaching situations it has not seen. Either it has "
                    "exhausted what this action set can reach (add keys with "
                    "--calibrate), or it has settled into a loop. Check the "
                    "action histogram below: one action dominating means a "
                    "loop, many actions with no novelty means saturation.")

        if engaged_rate < 15.0:
            messages.append(
                f"[Health] The policy pressed nothing on "
                f"{100.0 - engaged_rate:.0f}% of steps. Intrinsic reward can "
                f"settle on noop because it is never punished by dying. Raise "
                f"w_idle, or raise entropy_coef to force exploration back.")
        if self.longest_noop_streak > self.cfg.status_every:
            messages.append(
                f"[Health] Longest run of idle decisions: "
                f"{self.longest_noop_streak}. The policy is stuck on noop.")
        return messages

    def start_window(self) -> None:
        self.novel_steps = 0
        self.longest_noop_streak = 0


class GameSession:
    """
    Drives one game with one policy.

    `env` only has to provide reset()/step(spec)/close() and observations, so
    the same session runs against a real window or against the synthetic game
    used by --selftest.
    """

    def __init__(self, cfg: Config, env, action_space: ActionSpace,
                 keymap: Optional[Keymap] = None,
                 device: Optional[torch.device] = None):
        self.cfg = cfg.validate()
        self.env = env
        self.action_space = action_space
        self.keymap = keymap
        self.device = device or torch.device("cpu")
        self.observation_shape = tuple(env.observation_shape)

        self.policy = ActorCritic(
            self.cfg, action_space.head_sizes, observation_shape_channels(
                self.observation_shape)).to(self.device)
        self.reward = IntrinsicReward(
            self.cfg, self.device, action_space.action_vector_size)
        self.trainer = Trainer(self.cfg, self.policy, self.device)
        self.buffer = RolloutBuffer(
            self.cfg.rollout_steps, self.cfg.embed_dim, self.cfg.mem_tokens,
            len(action_space.hold_vks), self.device)
        self.health = HealthMonitor(self.cfg)

        # The view depth the bot starts from. --calibrate measures it, and
        # --mouse-turn pins it; either way the bot is free to move it inside the
        # configured bounds afterwards, because the only thing that knows how
        # far this game's view really turns is the game.
        if keymap is not None:
            self.policy.set_mouse_step(float(
                min(max(int(keymap.default_mouse_turn),
                        int(self.cfg.mouse_turn_min)),
                    int(self.cfg.mouse_turn_max))))
        if self.cfg.mouse_turn_pixels:
            self.policy.set_mouse_step(float(self.cfg.mouse_turn_pixels))

        # Running performance split, which is what the [Perf] line reports.
        self.env_seconds = 0.0
        self.update_seconds = 0.0
        self.episodes = 0
        self.decisions = 0
        self.total_steps = 0
        self.hidden = self.policy.initial_hidden(1, self.device)
        self.observation: Optional[np.ndarray] = None
        self.observation_signature: Optional[np.ndarray] = None
        self.last_value = 0.0
        self._started = time.monotonic()
        self._log_file = None
        self._status_window_steps = 0
        self._status_engaged = 0
        self._status_novel = 0
        self._held_vks: List[int] = []
        self._action_counts: Dict[str, int] = defaultdict(int)
        self._alerted_no_delivery = False
        self._alerted_non_game = False
        self._pending_summary = np.zeros(self.cfg.embed_dim, dtype=np.float32)
        self._pending_prediction = np.zeros(self.cfg.embed_dim,
                                            dtype=np.float32)
        self._pending_context = np.zeros(self.cfg.embed_dim, dtype=np.float32)
        self._pending_window = self.hidden.squeeze(0).cpu().numpy().copy()
        # A prediction is only meaningful once a *previous* decision has made
        # one; see `decide`.
        self._pending_has_prediction = False
        # How many turns the bot has taken, how many of them it could actually
        # see change the picture, and the recent history the correction in
        # `_note_turn` is gated on.
        self._turn_decisions = 0
        self._turn_observations = 0
        self._turn_window: Deque[int] = deque(maxlen=20)

        if self.cfg.log_json:
            os.makedirs(os.path.dirname(os.path.abspath(self.cfg.log_json))
                        or ".", exist_ok=True)
            self._log_file = open(self.cfg.log_json, "a", encoding="utf-8")

    # =====================================================================
    # Environment interaction
    # =====================================================================
    def reset(self, seed: Optional[int] = None,
              keep_rollout: bool = False) -> np.ndarray:
        """
        Start an episode.

        ``keep_rollout`` is what separates "this is a new episode" from "this is
        a new run". A game that ends every few hundred steps must not throw
        away the rollout being collected, or - if the episodes are shorter than
        the rollout - the buffer would never fill and the bot would collect
        frames forever without ever updating. That failure is silent and looks
        exactly like a hang.
        """
        observation, _signature = self.env.reset(seed=seed)
        self.observation = observation
        self.observation_signature = self._signature_of(observation)
        self.hidden = self.policy.initial_hidden(1, self.device)
        self.reward.reset_episode(self.observation_signature)
        self.episodes += 1
        if not keep_rollout:
            self.buffer.reset()
            self.last_value = 0.0
        self._pending_summary = np.zeros(self.cfg.embed_dim, dtype=np.float32)
        self._pending_prediction = np.zeros(self.cfg.embed_dim,
                                            dtype=np.float32)
        self._pending_has_prediction = False
        self.decide()
        return observation

    def decide(self) -> Tuple[np.ndarray, int, int, int]:
        """
        Choose an action for the current observation.

        Everything the later stages need is stashed here: the log-probability
        and value of the decision, the memory window that was in force *before*
        it (which is what the transformer PPO update needs), the pooled context
        and this frame's summary (which the curiosity head works from), and the
        model's prediction of the *next* frame, so the reward at the next step
        can score it without another forward pass.
        """
        action, log_prob, value, new_hidden, context, summary, previous = \
            self.trainer.act(self.observation, self.hidden)
        self._pending_log_prob = log_prob
        self._pending_value = value
        self._pending_window = previous
        self._pending_summary = summary
        self._pending_context = context
        self.hidden = new_hidden
        # What this frame says the next frame should look like. The reward
        # compares it against the real thing one step later; see
        # IntrinsicReward.compute.
        context_tensor = torch.as_tensor(context, dtype=torch.float32,
                                         device=self.device).unsqueeze(0)
        with torch.no_grad():
            self._pending_prediction = self.policy.predict_next(
                context_tensor).squeeze(0).cpu().numpy()
        # Only now is there a prediction to score against the next frame: the
        # first decision of a run (and the first after a reset window change)
        # has no predecessor to have predicted it.
        self._pending_has_prediction = True
        return action

    @property
    def hold_vks(self) -> List[int]:
        return list(self.action_space.hold_vks)

    def hold_keys(self) -> List[int]:
        """Virtual keys currently held by the bot (for release on shutdown)."""
        return list(self._held_vks)

    def action_for_env(self, action: Tuple[np.ndarray, int, int, int]) -> Dict:
        """
        Translate a factored decision into something an environment can apply.

        A dict rather than parallel arguments, because the two consumers want
        different things from it: the real game wants the keys and the pixel
        delta, the synthetic game wants a flat index.

        The pixel delta is computed here, from the bot's *current* mouse step,
        which is the part of the speed decision the bot owns (see
        `_note_turn`).  The policy chose a direction and a multiplier; this
        turns that into pixels.
        """
        held = np.asarray(action[0]).reshape(-1)
        held_vks: List[int] = []
        for i, bit in enumerate(held):
            if bit > 0.5 and i < len(self.action_space.hold_vks):
                held_vks.append(int(self.action_space.hold_vks[i]))
        direction_index = int(action[1])
        speed_index = int(action[2])
        tap_index = int(action[3])
        step = self.policy.mouse_step_value()
        turn = self.action_space.turn_delta(direction_index, speed_index, step)
        tap_vk = 0
        if 0 < tap_index < len(self.action_space.taps):
            tap_vk = int(self.action_space.taps[tap_index].get("vk", 0))
        mask = 0
        for i, bit in enumerate(held[:16]):
            if bit > 0.5:
                mask |= 1 << i
        index = ((mask * self.action_space.n_turn + direction_index)
                 * self.action_space.n_speed + speed_index)
        index = index * self.action_space.n_tap + tap_index
        return {"held_vks": held_vks, "turn": tuple(turn), "tap_vk": tap_vk,
                "index": int(index), "held_mask": int(mask),
                "turn_index": direction_index, "speed_index": speed_index,
                "tap_index": tap_index, "mouse_step": int(step)}

    def step(self, action: Tuple[np.ndarray, int, int, int]
             ) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        """
        Apply a decision and record one transition.

        The same decision is held for `action_repeat` frames: one model
        evaluation per repeat frames is a straight throughput win, and holding
        a button for several frames is how the input reads to the game anyway.

        Returns (observation, reward, done, info), where `done` means the
        environment genuinely ended - a closed window, or a synthetic episode
        hitting its limit. An inferred in-game reset is *not* a done: it is
        handled inside the reward, because the game keeps running.
        """
        cfg = self.cfg
        if self.buffer.full:
            # Refuse loudly rather than dropping the transition. Silently
            # discarding rollout data would make the update non-on-policy,
            # which corrupts PPO in a way nothing else in the log would show.
            raise RuntimeError(
                "the rollout buffer is full; call learn() before stepping "
                "again (GameSession.run does this automatically)")
        env_action = self.action_for_env(action)
        started = time.perf_counter()
        observation, truncated, _info = self.env.step(env_action)
        for _ in range(cfg.action_repeat - 1):
            if truncated:
                break
            observation, extra, _info = self.env.step(env_action)
            truncated = truncated or extra
        self.env_seconds += time.perf_counter() - started

        engaged = bool(action[0].sum() or action[1] or action[2] or action[3])
        signature = self._signature_of(observation)

        # Reward, novelty and reset detection all reuse the representation and
        # the prediction the policy already produced. One transformer pass per
        # step, for every consumer.
        action_vector = self.action_space.describe_action(
            action[0], action[1], action[2], action[3])
        reward, reset, reason = self.reward.compute(
            self._pending_summary, signature, action_vector, engaged=engaged,
            step=self.total_steps, prediction=self._pending_prediction,
            has_prediction=self._pending_has_prediction)

        self.buffer.add(
            context=self._pending_context,
            summary=self._pending_summary,
            window=self._pending_window,
            held=action[0],
            direction=int(action[1]),
            speed=int(action[2]),
            tap=int(action[3]),
            log_prob=self._pending_log_prob,
            value=self._pending_value,
            reward=reward,
            terminal=bool(truncated),
            reset=bool(reset),
            action_vector=action_vector,
        )
        self._held_vks = list(env_action["held_vks"])
        self._action_counts[self._action_label(action)] += 1
        self._note_turn(action, env_action, signature)

        delta = 0.0
        if self.observation_signature is not None:
            delta = float(np.abs(signature - self.observation_signature).mean())
        message = self.health.note_step(
            delta, engaged=engaged,
            episodic_bonus=self.reward.last.get("episodic", 0.0))
        if message:
            self._print(message)

        self.observation = observation
        self.observation_signature = signature
        self.decisions += 1
        self.total_steps += 1
        self._status_window_steps += 1
        if engaged:
            self._status_engaged += 1
        if self.reward.last.get("episodic", 0.0) > 0:
            self._status_novel += 1
        if reset:
            self.episodes += 1
            # The next transition is on the far side of a reset, so there is
            # nothing for the curiosity head to learn from across it; the
            # buffer marks the step and the reward module has already dropped
            # its previous summary for the same reason.
            self._pending_summary = np.zeros(self.cfg.embed_dim,
                                             dtype=np.float32)
            self._pending_prediction = np.zeros(self.cfg.embed_dim,
                                                dtype=np.float32)

        info = {"reward": reward, "reset": reset, "reset_reason": reason,
                "reward_parts": dict(self.reward.last)}
        self._check_input_delivery()
        return observation, reward, bool(truncated), info

    # ---- health: is the bot driving anything? ----
    def _input_stats(self) -> Optional[Dict[str, object]]:
        """Input/delivery counters from the environment, when it reports them."""
        getter = getattr(self.env, "input_stats", None)
        if not callable(getter):
            return None
        try:
            stats = getter()
        except Exception:
            return None
        return stats if isinstance(stats, dict) else None

    def _check_input_delivery(self) -> None:
        """
        Say so, once and out loud, when the loop is running but nothing is
        reaching a window.

        This is the failure behind "it says 20 steps a second but nothing
        happens": every frame arrives, the policy decides on schedule, the
        rollout fills, and every keystroke is delivered to some other window -
        or to none at all. Nothing else in the log distinguishes that from a
        run that is working, so it is checked directly rather than inferred
        from the reward going flat.
        """
        stats = self._input_stats()
        if not stats:
            return
        if not self._alerted_non_game and stats.get("non_game"):
            self._alerted_non_game = True
            self._print(
                f"[Health] This run is driving '{stats.get('title')}' "
                f"({stats.get('process')}), which looks like a terminal, "
                f"browser or editor rather than a game. Input sent there does "
                f"not reach a game at all. Check the line above that says "
                f"which window was selected, then restart with --pick or "
                f"--window HWND (see --list-windows).")
        if self._alerted_no_delivery:
            return
        # Give a fresh run long enough to have pressed something at all: the
        # first few steps are legitimately idle while the frame stack fills.
        if self.total_steps < max(50, self.cfg.status_every // 4):
            return
        if int(stats.get("delivered") or 0) > 0:
            return
        self._alerted_no_delivery = True
        if stats.get("focused"):
            detail = ("The window it selected is in front, so the window is "
                      "not the problem: either the policy has not pressed "
                      "anything yet, or the game was told to ignore input "
                      "from an unfocused window (--focus always is the "
                      "workaround for a few games).")
        else:
            detail = ("The game window is not focused, so nothing can be "
                      "sent. Click the game - input resumes on its own.")
        self._print(
            f"[Health] {self.total_steps} decisions and not one key or mouse "
            f"event has been delivered. The bot is deciding at full speed "
            f"about a window it is not driving. {detail}")

    # ---- helpers ----
    def _action_label(self, action: Tuple[np.ndarray, int, int, int]) -> str:
        """
        Readable name for a decision, coarse enough to be a useful histogram.

        Individual held keys are collapsed to their count, because what the
        status line needs to answer is "is it pressing things, and is one thing
        dominating" - not which exact combination of four keys it chose.
        """
        held = np.asarray(action[0]).reshape(-1)
        count = int((held > 0.5).sum())
        parts = [f"{count} key(s)"] if count else ["no keys"]
        _name, delta = self.action_space.direction(int(action[1]))
        if delta != (0, 0):
            parts.append("turn")
        if action[3]:
            parts.append("tap")
        return "+".join(parts)

    def _note_turn(self, action: Tuple[np.ndarray, int, int, int],
                   env_action: Dict, signature: np.ndarray) -> None:
        """
        Watch whether turns actually move the view, and adjust the bot's own
        mouse step when they plainly do not.

        This is the half of "the bot sets its own mouse speed" that no policy
        gradient can reach.  The policy learns *which* speed multiplier is worth
        choosing - that decision has a log-probability and PPO trains it like any
        other.  What it cannot learn is what one pixel of mouse movement is worth
        in this game, because that is not a decision: a game that locks the
        cursor reads a raw delta of 20 as a twitch, and no amount of reward
        shaping tells the policy that the number itself was wrong.

        So it is measured instead, and deliberately slowly: a single frame is
        evidence of very little, so the ratio of "turns that moved the view" is
        accumulated over a window and the base step only moves when almost all
        of them, or almost none of them, did.  Both directions are bounded by
        `mouse_turn_min`/`mouse_turn_max` and reported in the status block, so
        this cannot quietly run away.
        """
        cfg = self.cfg
        _name, delta = self.action_space.direction(int(action[1]))
        if delta == (0, 0):
            return
        self._turn_decisions += 1

        moved = float(np.abs(signature - self.observation_signature).mean()) \
            if self.observation_signature is not None else 0.0
        if moved > 0.001:
            self._turn_observations += 1

        rate = float(cfg.mouse_adapt_rate)
        if rate <= 0.0:
            return
        self._turn_window.append(1 if moved > 0.001 else 0)
        if len(self._turn_window) < self._turn_window.maxlen:
            return
        moved_share = sum(self._turn_window) / len(self._turn_window)
        if moved_share <= 0.2:
            # Repeatedly swinging the view and seeing the same picture: the step
            # is too small for whatever this game reads from the mouse.
            self.policy.nudge_mouse_step(1.0 + rate)
        elif moved_share >= 0.98:
            self.policy.nudge_mouse_step(1.0 / (1.0 + rate))

    # =====================================================================
    # The bot's own mouse speed
    # =====================================================================
    def mouse_speed_line(self) -> str:
        """What the bot is currently using as its turn step, and its range."""
        base = self.policy.mouse_step_value()
        levels = [float(s) for s in self.cfg.speed_levels] or [1.0]
        rendered = "/".join(
            str(max(1, int(round(base * level)))) for level in levels)
        seen = (f"{self._turn_observations} of {self._turn_decisions} turn(s) "
                f"moved the view")
        return (f"mouse {base}px per 1x step, x[{rendered}] across "
                f"{len(levels)} speed(s) - {seen}")

    def _signature_of(self, observation: np.ndarray) -> np.ndarray:
        """
        Coarse fingerprint of an observation.

        Taken from the grey channels the model already sees - the position
        channels averaged out - so this is one numpy pass and no colour
        conversion. Which channels those are depends on where the observation
        came from, so `gray_channels` is read from the environment rather than
        assumed; getting it wrong yields a constant fingerprint, which silently
        removes the entire episodic reward without raising anything.
        """
        count = int(getattr(self.env, "gray_channels", self.cfg.frame_stack))
        frames = observation[:max(1, count)]
        return gray_signature(frames.mean(axis=0, dtype=np.float32))

    def _print(self, text: str) -> None:
        print(text, flush=True)

    # =====================================================================
    # Learning
    # =====================================================================
    def learn(self) -> Dict[str, float]:
        """
        Run one PPO update over the collected rollout, then teach the
        transformer's own curiosity head on the same data.

        The curiosity head is trained here rather than inside the control loop
        so its gradient work is batched, and through its own optimiser so a
        curiosity gradient never moves the policy's trunk (see
        `Trainer.train_transition`).
        """
        if len(self.buffer) < 2:
            return {}
        started = time.perf_counter()
        metrics = self.trainer.update(self.buffer, self.last_value)

        # Share whatever is left of the update budget with the curiosity head,
        # so a slow machine still collects frames rather than silently spending
        # its whole life training curiosity.
        spent = time.perf_counter() - started
        remaining = max(0.0, self.cfg.update_seconds_budget * 0.5 - spent)
        tensors = self.buffer.tensors()
        if "action_vector" in tensors and remaining > 0.0:
            metrics.update(self.trainer.train_transition(
                tensors, minibatch=self.cfg.minibatch_size, steps=2,
                budget_seconds=remaining))

        self.update_seconds += time.perf_counter() - started
        self.buffer.reset()
        if self._log_file is not None and metrics:
            record = {"step": self.total_steps, "time": time.time(), **metrics}
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()
        return metrics

    def bootstrapped_value(self) -> float:
        """Value of the current observation, for the GAE tail."""
        obs = torch.as_tensor(self.observation, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        self.policy.eval()
        with torch.no_grad():
            value, _hidden = self.policy.value_only(obs, self.hidden)
        self.last_value = float(value.item())
        return self.last_value

    # =====================================================================
    # Reporting
    # =====================================================================
    def status_block(self) -> None:
        wall = self.env_seconds + self.update_seconds
        env_share = 100.0 * self.env_seconds / wall if wall > 0 else 100.0
        steps = max(1, self._status_window_steps)
        elapsed = time.monotonic() - self._started
        rate = self.total_steps / max(1e-6, elapsed)
        self._print("")
        self._print(f"[Step {self.total_steps}] {rate:.1f} decisions/s, "
                    f"{env_share:.0f}% of wall clock in the game "
                    f"({100.0 - env_share:.0f}% training), "
                    f"{self.episodes} episode(s)")
        self._print(f"          {self.trainer.status_line()}")
        self._print(f"          {self.reward.summary()}  "
                    f"| resets {self.reward.detector.resets} "
                    f"(last: {self.reward.detector.last_reason or '-'}), "
                    f"{self._status_novel * 1000 // steps} new states/1000 "
                    f"steps, {self._status_engaged * 100 // steps}% of steps "
                    f"pressed something")
        self._print(f"          {self.mouse_speed_line()}")
        stats = self._input_stats()
        if stats:
            self._print(
                f"          input: "
                f"{'focused' if stats.get('focused') else 'NOT FOCUSED'} - "
                f"{int(stats.get('delivered') or 0)} event(s) delivered to "
                f"{stats.get('window')}, "
                f"{int(stats.get('skipped_unfocused') or 0)} decision(s) "
                f"skipped while unfocused")

        total = sum(self._action_counts.values())
        if total:
            top = sorted(self._action_counts.items(), key=lambda kv: -kv[1])[:4]
            share = 100.0 * top[0][1] / total
            rendered = ", ".join(f"{name} {100.0 * n / total:.0f}%"
                                 for name, n in top)
            self._print(f"          decisions: {rendered}")
            if share > 90.0:
                self._print("          [Health] One decision type owns almost "
                            "every step. Raise entropy_coef to push the policy "
                            "apart, or lower w_idle if it is 'no keys'.")
        # Back to a *defaultdict*: the next step does
        # `self._action_counts[label] += 1`, and a plain dict here raises
        # KeyError on the first action label that was not in the previous
        # window - which killed the run on the first status block.
        self._action_counts = defaultdict(int)

        for message in self.health.end_window(
                steps, self._status_engaged, self.trainer.last_update_seconds):
            self._print("          " + message)
        if self.trainer.last_update_seconds > self.cfg.slow_update_seconds:
            self._print(
                f"          [Health] The last update took "
                f"{self.trainer.last_update_seconds:.1f}s "
                f"({100.0 - env_share:.0f}% of wall clock is training). "
                f"Lower rollout_steps, minibatch_size or seq_len, or raise "
                f"update_seconds_budget if the machine can spare it.")
        self.health.start_window()
        self._status_window_steps = 0
        self._status_engaged = 0
        self._status_novel = 0

    # =====================================================================
    # The loop
    # =====================================================================
    def run(self, control=None, max_decisions: Optional[int] = None,
            save_fn=None) -> None:
        """
        Collect, learn, report, forever (or until `max_decisions`).

        `control` is anything exposing the SignalController-style API
        (service/paused/stop_requested/checkpoint_due/claim_pause_notice); when
        it is None the loop simply runs.  `save_fn(reason)` is called when a
        checkpoint is due.
        """
        if control is not None and not self.wait_for_start(control):
            return
        # One attempt at the foreground window, now that the user has said
        # "go". Per the focus policy this never happens again on its own.
        self.focus_env()
        self.reset(seed=self.cfg.seed)
        last_status = self.total_steps

        while max_decisions is None or self.total_steps < max_decisions:
            if control is not None:
                for message in control.service():
                    self._print(f"[Control] {message}")
                if control.paused:
                    if control.claim_pause_notice():
                        self._print("[Control] PAUSED: no input is being sent. "
                                    "Press the pause key again to resume.")
                    # Suspending the environment is what actually stops the
                    # input, not merely releasing what is held: without it the
                    # next injected key would land while the user believes they
                    # are paused.
                    self.suspend_env()
                    while control.paused and not control.stop_requested:
                        for message in control.service():
                            self._print(f"[Control] {message}")
                        self.release_held()
                        time.sleep(0.05)
                    self.resume_env()
                    self.focus_env()
                    # The world moved on while paused, so the transformer's
                    # memory window and the pending observation are no longer a
                    # chain.
                    self.observation = self._current_observation()
                    self.observation_signature = self._signature_of(
                        self.observation)
                    self.hidden = self.policy.initial_hidden(1, self.device)
                    self.reward.reset_episode(self.observation_signature)
                    self.decide()
                due = control.checkpoint_due()
                if due and save_fn is not None:
                    save_fn(due)
                if control.stop_requested:
                    break

            while len(self.buffer) < self.cfg.rollout_steps:
                action = self.decide()
                _obs, _reward, done, _info = self.step(action)
                if done:
                    # reset() re-decides for the fresh observation, so the
                    # pending action always matches the pending observation.
                    # The rollout survives: an episode boundary is not the
                    # end of the data-collection run.
                    self.reset(seed=None, keep_rollout=True)
                if control is None:
                    continue
                # Checked inside the collection loop as well: a rollout can take
                # a minute at a low control rate, and a checkpoint that waits for
                # a whole rollout is a checkpoint that is missing when the
                # machine loses power.
                for message in control.service():
                    self._print(f"[Control] {message}")
                due = control.checkpoint_due()
                if due and save_fn is not None:
                    save_fn(due)
                if control.paused or control.stop_requested:
                    break
            if control is not None and control.stop_requested:
                break
            # A pause caught mid-rollout drops the partial batch, because the
            # transitions either side of it are not a continuous chain.
            if control is not None and control.paused:
                self.buffer.reset()
                continue

            self.bootstrapped_value()
            self.learn()

            # A NaN anywhere makes every later checkpoint worthless, so it is
            # caught here rather than hours later.
            bad = self.trainer.nonfinite_parameters()
            if bad:
                self._print(f"[Health] Non-finite weights in {bad[:3]}"
                            f"{'...' if len(bad) > 3 else ''}. "
                            f"Stopping so the previous checkpoint stays good.")
                break

            if self.total_steps - last_status >= self.cfg.status_every:
                last_status = self.total_steps
                self.status_block()

    # ---- environment control ----
    def wait_for_start(self, control) -> bool:
        """
        Block until the user presses the start key. Returns False if they quit
        instead.

        Nothing at all is sent before this, which is what makes the sequence
        "start the script, click the game, press the start key" safe: the bot
        cannot type into whichever window happened to have focus at launch.
        A control object without the gate (or with it already satisfied) starts
        immediately, so this is a no-op for the self-test and for --start-now.
        """
        gate = getattr(control, "awaiting_start", False)
        awaiting = bool(gate() if callable(gate) else gate)
        if not awaiting:
            return True
        key = str(getattr(self.cfg, "hotkey_pause", "f8")).upper()
        self._print("")
        self._print(f"[Control] Ready. Click the game window so it has focus, "
                    f"then press {key} to start.")
        self._print("[Control] Nothing is being sent until then, so it is safe "
                    "to use the terminal.")
        while control.awaiting_start and not control.stop_requested:
            for message in control.service():
                self._print(f"[Control] {message}")
            time.sleep(0.05)
        if control.stop_requested:
            self._print("[Control] Quit before starting; nothing was sent.")
            return False
        return True

    def focus_env(self) -> None:
        """
        Let the environment bring its window forward, once.

        Deliberately not per step: an injector that grabs the foreground on
        every action is an injector that fights the user for the keyboard.
        """
        acquire = getattr(self.env, "acquire_focus", None)
        if callable(acquire):
            try:
                focused = acquire()
            except Exception as exc:
                self._print(f"[Control] Could not focus the game window: {exc}")
                return
            if focused is False:
                self._print("[Control] The game window could not be brought "
                            "forward (Windows allows that only from the "
                            "foreground app). Click the game - input starts on "
                            "its own once the game has focus.")

    def _current_observation(self) -> np.ndarray:
        """Ask the environment for a fresh observation, if it can give one."""
        getter = getattr(self.env, "_get_obs", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                pass
        if self.observation is None:
            return np.zeros(self.observation_shape, dtype=np.float32)
        return self.observation

    def suspend_env(self) -> None:
        """Stop the environment injecting input (pause)."""
        suspend = getattr(self.env, "suspend", None)
        if callable(suspend):
            try:
                suspend()
            except Exception as exc:
                self._print(f"[Control] Could not suspend input: {exc}")

    def resume_env(self) -> None:
        resume = getattr(self.env, "resume", None)
        if callable(resume):
            try:
                resume()
            except Exception as exc:
                self._print(f"[Control] Could not resume input: {exc}")

    def release_held(self) -> None:
        """Let go of everything the bot is holding (pause, shutdown)."""
        release = getattr(self.env, "release_all", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass
        self._held_vks = []

    def close(self) -> None:
        self.release_held()
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        close = getattr(self.env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    # =====================================================================
    # Checkpointing
    # =====================================================================
    def state_dict(self) -> dict:
        return {
            "version": CHECKPOINT_VERSION,
            "arch": ARCH_NAME,
            "reward_version": REWARD_VERSION,
            "model_config": {
                "embed_dim": self.cfg.embed_dim,
                "mem_tokens": self.cfg.mem_tokens,
                "patch_size": self.cfg.patch_size,
                "patch_width": self.cfg.patch_width,
                "transformer_layers": self.cfg.transformer_layers,
                "ffn_hidden": self.cfg.ffn_hidden,
                "attention_heads": self.cfg.attention_heads,
                "frame_size": self.cfg.frame_size,
                "frame_stack": self.cfg.frame_stack,
                "channels": observation_shape_channels(self.observation_shape),
                "head_sizes": list(self.action_space.head_sizes),
                "observation_shape": list(self.observation_shape),
                "mouse_step": self.policy.mouse_step_value(),
            },
            "action_space": self.action_space.to_dict(),
            "policy": self.policy.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "intrinsic": self.reward.state_dict(),
            "step": self.total_steps,
            "episodes": self.episodes,
            "seconds": time.monotonic() - self._started,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
            },
        }

    def load_state_dict(self, state: dict, strict: bool = False) -> bool:
        """Restore a run. Returns False (changing nothing) if incompatible."""
        if not state:
            return False
        try:
            if state.get("reward_version") != REWARD_VERSION:
                raise ValueError(
                    f"checkpoint was trained against reward version "
                    f"{state.get('reward_version')}, this build uses "
                    f"{REWARD_VERSION}")
            if state.get("arch") != ARCH_NAME:
                raise ValueError(
                    f"checkpoint architecture '{state.get('arch')}' does not "
                    f"match this build's '{ARCH_NAME}'")
            shapes = state.get("model_config", {}).get("observation_shape")
            if shapes is not None and list(shapes) != list(self.observation_shape):
                raise ValueError(
                    f"checkpoint observation {shapes} does not match the "
                    f"current {list(self.observation_shape)}")
            heads = state.get("model_config", {}).get("head_sizes")
            if heads is not None and list(heads) != list(
                    self.action_space.head_sizes):
                raise ValueError(
                    f"checkpoint action space {heads} does not match the "
                    f"current {list(self.action_space.head_sizes)}")
            self.policy.load_state_dict(state["policy"])
            self.trainer.optimizer.load_state_dict(state["optimizer"])
            self.reward.load_state_dict(state.get("intrinsic") or {})
            # The bot's own mouse step is part of what it has learned, so it
            # comes back with the weights rather than being reset to the
            # calibration default on every resume.
            mouse_step = state.get("model_config", {}).get("mouse_step")
            if mouse_step:
                self.policy.set_mouse_step(float(mouse_step))
            self.total_steps = int(state.get("step", 0))
            self.episodes = int(state.get("episodes", 0))
            rng = state.get("rng") or {}
            if "torch" in rng:
                torch.set_rng_state(rng["torch"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            return True
        except Exception as exc:
            if strict:
                raise
            self._print(f"[Resume] Checkpoint not usable ({exc}); "
                        f"starting a fresh run.")
            return False


def observation_shape_channels(shape) -> int:
    """(C, H, W) -> C."""
    if len(shape) != 3:
        raise ValueError(f"observation shape must be (C, H, W), got {shape}")
    return int(shape[0])


def load_checkpoint(path: str, device: Optional[torch.device] = None
                    ) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location=device or "cpu", weights_only=False)
    except Exception as exc:
        print(f"[Resume] Could not read '{path}': {exc}")
        return None


def atomic_save(payload: dict, path: str) -> bool:
    """Write via a temporary file so an interrupted save cannot corrupt a good one."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return True
    except Exception as exc:
        print(f"[Checkpoint] Save to '{path}' failed: {exc}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False
