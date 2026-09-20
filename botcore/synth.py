"""
A synthetic game, whose progress is known exactly.

This exists so that "the bot learns to play" is a statement that can be checked
rather than hoped for.  `--selftest` runs the identical session, model, reward
and trainer against this environment and reports whether the bot moved further
into it over time.

The game is deliberately the smallest thing that still has the property that
matters: it is a long corridor, the bot can only walk forward or back, going
backwards kills it, and dying sends it back to the start.  So the only way to
see more of the game is to walk forward, and the only way to be rewarded for it
is to reach cells it has not reached since the last attempt.  That is exactly
the structure the intrinsic reward claims to exploit, and it is the structure
of a great many real games.

The bot is given no advantage here beyond what it gets in a real game: it sees
a rendered first-person-ish view of the corridor (never its own coordinates),
it presses the same keys (one forward key, one backwards key, and turning),
and its reward comes from the same reward module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .capture import gray_signature
from .config import Config


class SyntheticEnv:
    """
    A corridor the bot must learn to walk down.

    Observation: (frame_stack + 1, size, size) float32 in [0, 1], the same
    shape and meaning as the real environment's, so the session treats both
    identically.

    Action dict: ``held_mask`` bit 0 is the forward key, bit 1 is the backward
    key, ``turn_index`` 0 is no turn and 1/2 are left/right.
    """

    FORWARD_BIT = 0
    BACKWARD_BIT = 1

    def __init__(self, cfg: Config, length: int = 100,
                 max_episode_steps: int = 300, seed: int = 0):
        self.cfg = cfg
        self.length = int(length)
        self.max_episode_steps = int(max_episode_steps)
        self.rng = np.random.default_rng(seed)
        self.observation_shape = (cfg.channels, cfg.frame_size, cfg.frame_size)
        self.max_episode_steps = int(max_episode_steps)
        # Which channels of the observation are grey frames (the rest are the
        # difference channel). The session reads this so it fingerprints the
        # picture rather than a derived channel.
        self.gray_channels = int(cfg.frame_stack)

        self.cell = 0
        self.facing = 1
        self.steps = 0
        self.deaths = 0
        self.max_cell_seen = 0
        self.last_cell = 0
        self.total_forward_steps = 0
        # Steps still to spend on the death screen. A death that never ends
        # would make dying a place to sit and farm novelty, which is what an
        # absorbing failure state does to any intrinsic reward. Real games
        # respawn, so this one does too.
        self.dead_for = 0

        self._frames: List[np.ndarray] = []
        self._observation = np.zeros(self.observation_shape, dtype=np.float32)

    # =====================================================================
    # Rendering
    # =====================================================================
    def _gray_view(self) -> np.ndarray:
        """
        Render what the bot sees: a corridor ahead, brighter the closer the
        far wall is, plus side walls.  There are no coordinates in the image -
        the bot has to work out where it is from the view, like in a real game.
        """
        size = self.cfg.frame_size
        remaining = self.length - self.cell
        frame = np.zeros((size, size), dtype=np.float32)

        # Distant fog dims with distance; the far wall is brighter up close.
        horizon = int(size * 0.45)
        band = max(2, int(size * 0.18))
        brightness = 0.25 + 0.6 / (1.0 + remaining / 3.0)
        frame[horizon:horizon + band, :] = brightness

        # Side walls, converging toward the centre with distance.
        centre = size // 2
        spread = max(2, int(size * 0.42 * min(1.0, remaining / 6.0 + 0.15)))
        frame[:, max(0, centre - spread - 6):max(0, centre - spread)] = 0.35
        frame[:, centre + spread:centre + spread + 6] = 0.35

        # A facing marker: a small bright block offset by the heading, so
        # turning is visible without revealing the position.
        marker_x = centre + self.facing * int(size * 0.18)
        marker_y = horizon + band + 2
        frame[marker_y:marker_y + 4,
              max(0, marker_x - 2):marker_x + 2] = 1.0

        # Walking further in genuinely changes the picture even at a distance.
        frame[horizon:horizon + band, :] *= (1.0 + 0.03 * self.cell)
        return np.clip(frame, 0.0, 1.0)

    def _death_view(self) -> np.ndarray:
        """A distinct screen the reset detector can see as 'something changed'."""
        frame = np.full((self.cfg.frame_size, self.cfg.frame_size), 0.05,
                        dtype=np.float32)
        band = max(2, self.cfg.frame_size // 8)
        frame[::band, :] = 0.9
        return frame

    def _build_observation(self) -> np.ndarray:
        gray = self._death_view() if self.cell < 0 else self._gray_view()
        self._frames.append(gray)
        if len(self._frames) > self.cfg.frame_stack:
            self._frames.pop(0)
        while len(self._frames) < self.cfg.frame_stack:
            self._frames.insert(0, self._frames[0])
        stack = np.stack(self._frames, axis=0)
        self._observation[:self.cfg.frame_stack] = stack
        self._observation[self.cfg.frame_stack] = np.abs(
            stack[-1] - stack[0])
        return self._observation

    # =====================================================================
    # Environment API
    # =====================================================================
    def reset(self, seed: Optional[int] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.cell = 0
        self.facing = 1
        self.steps = 0
        self.last_cell = 0
        self.dead_for = 0
        self._frames = []
        observation = self._build_observation()
        return observation, gray_signature(observation[0])

    def step(self, action: Dict) -> Tuple[np.ndarray, bool, Dict]:
        mask = int(action.get("held_mask", 0))
        turn = int(action.get("turn_index", 0))

        if turn == 1:
            self.facing = -1
        elif turn == 2:
            self.facing = 1

        if self.cell < 0:
            # On the death screen. Count it down and respawn, which is the one
            # moment a reset detector should be certain about.
            self.dead_for -= 1
            if self.dead_for <= 0:
                self.cell = 0
                self.facing = 1
                self._frames = []
        else:
            forward = bool(mask & (1 << self.FORWARD_BIT))
            backward = bool(mask & (1 << self.BACKWARD_BIT))
            if forward and not backward:
                self.cell += 1
                self.total_forward_steps += 1
            elif backward and not forward:
                self.cell -= 1
            if self.cell >= self.length:
                self.cell = self.length - 1
            if self.cell < 0:
                self.deaths += 1
                self.dead_for = 3

        observation = self._build_observation()
        self.steps += 1
        self.max_cell_seen = max(self.max_cell_seen, self.cell)

        died = self.cell < 0
        truncated = (not died) and self.steps >= self.max_episode_steps
        return observation, bool(truncated), {"cell": self.cell,
                                              "died": died}

    # ---- evaluation ----
    def progress(self) -> int:
        """Current goal progress: how far into the corridor the bot is."""
        return max(0, self.cell)

    def close(self) -> None:
        pass

    def release_all(self) -> None:
        pass


class SyntheticGridEnv(SyntheticEnv):
    """
    A second, harder synthetic game: a 2-D grid with a goal.

    Same interface and same reward; the difference is that progress needs two
    independent behaviours (go north, and go east) rather than one, so it
    catches a reward function that only happens to work for a single corridor.
    """

    FORWARD_BIT = 0
    BACKWARD_BIT = 1
    STRAFE_BIT = 2

    def __init__(self, cfg: Config, extent: int = 6,
                 max_episode_steps: int = 150, seed: int = 0):
        super().__init__(cfg, length=extent, max_episode_steps=max_episode_steps,
                         seed=seed)
        self.extent = int(extent)
        self.x = 0
        self.y = 0

    def reset(self, seed: Optional[int] = None):
        observation, signature = super().reset(seed)
        self.x = 0
        self.y = 0
        return observation, signature

    def progress(self) -> int:
        return max(0, self.x) + max(0, self.y)

    def step(self, action: Dict) -> Tuple[np.ndarray, bool, Dict]:
        mask = int(action.get("held_mask", 0))
        turn = int(action.get("turn_index", 0))
        if turn == 1:
            self.facing = -1
        elif turn == 2:
            self.facing = 1

        if self.cell >= 0:
            forward = bool(mask & (1 << self.FORWARD_BIT))
            backward = bool(mask & (1 << self.BACKWARD_BIT))
            strafe = bool(mask & (1 << self.STRAFE_BIT))
            if forward and not backward:
                self.cell += 1
            elif backward and not forward:
                self.cell -= 1
            if strafe:
                self.x += 1
            if self.cell >= self.extent:
                self.cell = self.extent - 1
            if self.x >= self.extent:
                self.x = self.extent - 1

        observation = self._build_observation()
        self.steps += 1
        self.max_cell_seen = max(self.max_cell_seen, self.cell)
        died = self.cell < 0
        truncated = (not died) and self.steps >= self.max_episode_steps
        return observation, bool(truncated), {"cell": self.cell,
                                              "x": self.x, "died": died}

    def _gray_view(self) -> np.ndarray:
        """Corridor brightness blended with the side-progress marker."""
        base = super()._gray_view()
        size = self.cfg.frame_size
        marker = int(size * 0.5 * min(1.0, max(0, self.x) / max(1, self.extent)))
        base[-8:, :marker] = 0.85
        return base


# =============================================================================
# The learning check
# =============================================================================

def _peak_distance(cells: List[int]) -> int:
    """How far into the world these states got, measured from where it started."""
    if not cells:
        return 0
    return max(0, max(cells) - cells[0])


def greedy_action(policy, observation: np.ndarray, hidden, device,
                  rng: Optional[np.random.Generator] = None):
    """
    A decision that follows the policy's own preference without sampling noise.

    The held-key head is a *set* of binary bits, so there is no single argmax:
    thresholding every bit at 0.5 would have an undecided policy press every key
    at once, which looks like a broken agent rather than an undecided one.
    Instead the bits are ranked by their log-odds and the top-k are taken, where
    k is the count the distribution favours. That is the mode of the policy in
    the only sense that is well defined for a set-valued action.
    """
    import torch

    policy.eval()
    with torch.no_grad():
        obs = torch.as_tensor(observation, dtype=torch.float32,
                              device=device).unsqueeze(0)
        embed = policy.encode(obs)
        hidden = policy.gru(embed, hidden)
        hold_logits, turn_logits, tap_logits, _value = policy.heads(hidden)

    logits = hold_logits.squeeze(0)
    probabilities = torch.sigmoid(logits)
    key_count = logits.numel()
    cap = min(policy.max_held, key_count)
    # k = the number of keys the distribution expects to be held.
    expected = float(probabilities.sum().item())
    k = int(round(expected))
    k = max(0, min(cap, k))
    mask = 0
    if k > 0:
        order = torch.argsort(logits, descending=True)[:k]
        for index in order.tolist():
            mask |= 1 << int(index)
    action = {"held_mask": int(mask),
              "turn_index": int(torch.argmax(turn_logits, dim=-1).item()),
              "tap_index": int(torch.argmax(tap_logits, dim=-1).item()),
              "held_vks": [], "turn": (0, 0), "tap_vk": 0}
    return action, hidden


def evaluate(policy, env, device, episodes: int = 5, max_steps: int = 200
             ) -> Dict[str, float]:
    """Run the policy without sampling noise and measure how far it gets."""
    distances: List[int] = []
    steps_taken: List[int] = []
    for _episode in range(int(episodes)):
        observation, _signature = env.reset(seed=None)
        hidden = policy.initial_hidden(1, device)
        cells: List[int] = []
        for _step in range(int(max_steps)):
            action, hidden = greedy_action(policy, observation, hidden, device)
            observation, done, info = env.step(action)
            cells.append(int(info.get("cell", 0)))
            if done or info.get("died"):
                break
        distances.append(_peak_distance(cells))
        steps_taken.append(len(cells))
    return {
        "mean_distance": float(np.mean(distances)) if distances else 0.0,
        "best_distance": float(max(distances)) if distances else 0.0,
        "mean_steps": float(np.mean(steps_taken)) if steps_taken else 0.0,
    }


def _synthetic_train_config(cfg: Config, frame_size: int,
                            rollout_steps: int, seed: int) -> Config:
    """A config for the synthetic runs: same algorithm, no wall-clock pacing."""
    return Config(
        frame_size=int(frame_size),
        frame_stack=4,
        action_repeat=2,
        rollout_steps=int(rollout_steps),
        minibatch_size=256,
        epochs_per_update=4,
        target_fps=0.0,
        enable_hotkeys=False,
        torch_threads=int(cfg.torch_threads),
        seed=int(seed),
        checkpoint_dir="",
        log_json=None,
        max_held_keys=3,
        novelty_bins=48.0,
    ).validate()


def _synthetic_action_space(cfg: Config):
    """
    A keymap whose hold order puts the two movement keys in the first two bit
    positions, so the synthetic world can read them. The bot is never told
    which key means what - it has to discover that pressing one helps.
    """
    from .keys import ActionSpace, Keymap

    keymap = Keymap.default()
    keymap.holds = ["W", "S", "D"]
    keymap.taps = []
    keymap.mouse_buttons = []
    keymap.vks = {"W": 0x57, "S": 0x53, "D": 0x44}
    return ActionSpace.build(
        keymap, max_held=cfg.max_held_keys,
        turn_levels=(("normal", 1.0), ("fast", 2.0)),
        include_mouse_taps=False)


def random_baseline(env, max_held: int, episodes: int = 5,
                    max_steps: int = 300, seed: int = 12345
                    ) -> Dict[str, float]:
    """
    How far an agent that presses keys at random gets.

    This is the control that makes the self-test mean something. In a small
    maze a random walk scores well by accident, so "the trained policy got
    further" only says something if it also beat random by a clear margin.
    """
    rng = np.random.default_rng(seed)
    distances: List[int] = []
    for _episode in range(int(episodes)):
        env.reset(seed=None)
        cells: List[int] = []
        for _step in range(int(max_steps)):
            mask = 0
            for index in range(max_held):
                if rng.random() < 0.25:
                    mask |= 1 << index
            turn = int(rng.integers(0, 3))
            observation, done, info = env.step(
                {"held_mask": mask, "turn_index": turn, "tap_index": 0,
                 "held_vks": [], "turn": (0, 0), "tap_vk": 0})
            cells.append(int(info.get("cell", 0)))
            if done or info.get("died"):
                break
        distances.append(_peak_distance(cells))
    return {"mean_distance": float(np.mean(distances)) if distances else 0.0,
            "best_distance": float(max(distances)) if distances else 0.0}


def run_learning_check(verbose: bool = True, seed: int = 0,
                       steps: int = 24000, frame_size: int = 48,
                       rollout_steps: int = 1024,
                       corridor_length: int = 60) -> Dict[str, object]:
    """
    Train the real algorithm on a game whose progress is known, and report
    whether it got better.

    Nothing here is special-cased for the synthetic environment: it builds the
    same Config, the same GameSession, the same reward module and the same PPO
    update that run against a real window.  That is the point - this is a
    measurement of the bot, not of a toy.
    """
    import time as _time

    from .session import GameSession

    results: Dict[str, object] = {"checks": []}
    cfg_for_build = Config()

    def report(name: str, before: Dict[str, float], after: Dict[str, float],
               session, elapsed: float, control: Optional[Dict[str, float]] = None,
               target_distance: float = 0.0):
        improvement = after["mean_distance"] - before["mean_distance"]
        baseline = control["mean_distance"] if control else 0.0
        reached_target = after["mean_distance"] >= 0.75 * max(1.0, target_distance)
        # "Learned" means it ends up playing the game, or it clearly improved on
        # where it started. Grading purely on improvement would fail a lucky
        # initialisation that already plays well, which says nothing about the
        # algorithm; grading purely on the final score would pass a policy that
        # was handed the answer by its initial weights.
        learned = reached_target or improvement > 0.25 * max(1.0, target_distance)
        passed = learned and after["mean_distance"] >= 5.0
        if control is not None:
            # Beating a random walk is the part that cannot happen by accident.
            passed = passed and after["mean_distance"] > 1.5 * baseline
        check = {
            "name": name,
            "before": before["mean_distance"],
            "after": after["mean_distance"],
            "best": after["best_distance"],
            "target_distance": target_distance,
            "random_baseline": baseline,
            "improvement": improvement,
            "steps": session.total_steps,
            "solved_at": solved_at,
            "seconds": elapsed,
            "steps_per_second": session.total_steps / max(1e-9, elapsed),
            "passed": bool(passed),
        }
        results["checks"].append(check)
        if verbose:
            baseline_text = (f"   random {baseline:5.1f}" if control is not None
                             else "")
            print(f"  {name:<24} distance {before['mean_distance']:5.1f} -> "
                  f"{after['mean_distance']:5.1f}{baseline_text}"
                  f"   {session.total_steps} steps in {elapsed:5.1f}s"
                  f"   {'LEARNED' if passed else 'DID NOT LEARN'}")
        return check

    if verbose:
        print()
        print("-" * 78)
        print("  SELF-TEST: can the identical algorithm learn a game it has")
        print("  never seen, with no game-specific code and no score reading?")
        print("-" * 78)

    # ---- 1. corridor: the cleanest statement of 'make progress' ----
    cfg = _synthetic_train_config(cfg_for_build, frame_size, rollout_steps, seed)
    env = SyntheticEnv(cfg, length=corridor_length, max_episode_steps=300,
                       seed=seed)
    session = GameSession(cfg, env, _synthetic_action_space(cfg))
    control = random_baseline(env, cfg.max_held_keys, episodes=5,
                              max_steps=300)
    before = evaluate(session.policy, env, session.device, episodes=5,
                      max_steps=300)
    started = _time.perf_counter()
    session.reset(seed=seed)
    solved_at = 0
    updates_done = 0
    while session.total_steps < int(steps):
        action = session.decide()
        _obs, _reward, done, _info = session.step(action)
        if done:
            session.reset(seed=None, keep_rollout=True)
        if len(session.buffer) >= cfg.rollout_steps:
            session.bootstrapped_value()
            session.learn()
            updates_done += 1
            # Stop as soon as the bot is demonstrably playing the game, rather
            # than spending the whole budget proving it twice. This also means
            # the step budget can be generous without the test becoming slow.
            if updates_done % 4 == 0:
                probe = evaluate(session.policy, env, session.device,
                                 episodes=3, max_steps=300)
                if probe["mean_distance"] > 0.5 * corridor_length:
                    solved_at = session.total_steps
                    break
    elapsed = _time.perf_counter() - started
    after = evaluate(session.policy, env, session.device, episodes=5,
                     max_steps=300)
    results["corridor"] = report("corridor (go forward)", before, after,
                                 session, elapsed, control,
                                 target_distance=corridor_length)
    results["passed"] = bool(results["corridor"]["passed"])

    # ---- 2. the reward must prefer progress over going backwards ----
    #
    # An absolute threshold would be meaningless here, because it would have to
    # be retuned every time a reward weight changes. What must stay true is the
    # *ordering*: a policy that walks forward has to earn clearly more per step
    # than one that walks backwards, or the reward is not measuring progress.
    if verbose:
        print()
        print("  Control: the reward must prefer going forward to going")
        print("  backwards, or it is not a progress signal.")

    def fixed_policy_reward(mask: int, trials: int = 400) -> float:
        env_ = SyntheticEnv(cfg, length=corridor_length,
                            max_episode_steps=300, seed=seed)
        session = GameSession(cfg, env_, _synthetic_action_space(cfg))
        session.reset(seed=seed)
        # One bit at a time, so "forward" really means forward.
        action = (np.zeros(3, dtype=np.float32), 0, 0)
        if mask:
            action[0][int(np.log2(mask))] = 1.0
        total = 0.0
        for _ in range(trials):
            _obs, reward, done, _info = session.step(action)
            total += reward
            if done:
                session.reset(seed=None)
        return total / trials

    forward_reward = fixed_policy_reward(1 << SyntheticEnv.FORWARD_BIT)
    control_reward = fixed_policy_reward(1 << SyntheticEnv.BACKWARD_BIT)
    ratio = control_reward / max(1e-6, forward_reward)
    control_passed = ratio < 0.35
    results["forward_reward_per_step"] = forward_reward
    results["backwards_reward_per_step"] = control_reward
    results["backwards_ratio"] = ratio
    results["control_passed"] = bool(control_passed)
    if verbose:
        print(f"  {'always forward':<24} reward/step {forward_reward:+.4f}")
        print(f"  {'always backwards':<24} reward/step {control_reward:+.4f}"
              f"   {100.0 * ratio:.0f}% of forward"
              f"   {'correctly lower' if control_passed else 'TOO HIGH: the reward can be farmed by dying'}")

    # ---- 3. the reward must not be farmable by standing still ----
    idle_reward = fixed_policy_reward(0)
    idle_ratio = idle_reward / max(1e-6, forward_reward)
    idle_passed = idle_ratio < 0.35
    results["idle_reward_per_step"] = idle_reward
    results["idle_ratio"] = idle_ratio
    results["idle_passed"] = bool(idle_passed)
    if verbose:
        print(f"  {'never press anything':<24} reward/step {idle_reward:+.4f}"
              f"   {100.0 * idle_ratio:.0f}% of forward"
              f"   {'correctly lower' if idle_passed else 'TOO HIGH: standing still pays'}")

    results["ok"] = bool(results["passed"] and control_passed and idle_passed)
    if verbose:
        print("-" * 78)
        print(f"  RESULT: {'PASS' if results['ok'] else 'FAIL'}")
        print("-" * 78)
        print()
    return results
