"""
Storage for one rollout, and the sequence minibatch sampler.

Two details here matter more than they look:

**The memory window is stored, not recomputed.** A transformer policy in PPO
needs the window that was actually in force when each action was taken.  The
obvious shortcut - re-run the encoder over the rollout to rebuild the windows -
makes the update score old actions under new inputs, which quietly corrupts the
importance ratio.  Storing one window per step costs a few hundred kilobytes per
rollout and removes the problem entirely.

**Memory is a fixed allocation.**  The arrays are allocated once at the rollout
size and overwritten.  Nothing about this buffer grows over a run, which is one
of the reasons the bot does not degrade after a few hours.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-capacity storage for one on-policy rollout."""

    def __init__(self, capacity: int, embed_dim: int, mem_tokens: int,
                 n_keys: int, device: torch.device):
        self.capacity = int(capacity)
        self.embed_dim = int(embed_dim)
        self.mem_tokens = max(1, int(mem_tokens))
        self.n_keys = int(n_keys)
        self.device = device
        self.reset()

        # Preallocated, never reallocated. `window` is the transformer's memory
        # window *before* each step; `summary` is the representation of the
        # frame that step saw, which is what the transformer's next-frame
        # prediction is scored against one step later.
        self._window = np.zeros((capacity, self.mem_tokens, embed_dim),
                                dtype=np.float32)
        self._summary = np.zeros((capacity, embed_dim), dtype=np.float32)
        self._context = np.zeros((capacity, embed_dim), dtype=np.float32)
        self._action_vec_dim = 0
        self._action_vec: Optional[np.ndarray] = None
        self._held = np.zeros((capacity, n_keys), dtype=np.float32)
        self._turn = np.zeros(capacity, dtype=np.int64)
        self._speed = np.zeros(capacity, dtype=np.int64)
        self._tap = np.zeros(capacity, dtype=np.int64)
        self._logp = np.zeros(capacity, dtype=np.float32)
        self._value = np.zeros(capacity, dtype=np.float32)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._done = np.zeros(capacity, dtype=np.float32)
        self._reset = np.zeros(capacity, dtype=np.float32)

    def reset(self) -> None:
        self.size = 0
        self.episodes = 0

    def __len__(self) -> int:
        return self.size

    @property
    def full(self) -> bool:
        return self.size >= self.capacity

    def add(self, context: np.ndarray, summary: np.ndarray,
            window: np.ndarray, held: np.ndarray, direction: int, speed: int,
            tap: int, log_prob: float, value: float, reward: float,
            terminal: bool, reset: bool = False,
            action_vector: Optional[np.ndarray] = None) -> None:
        """
        Add one transition.

        ``terminal`` means the environment really ended (window closed, or a
        synthetic episode reached its limit).  An *inferred* game reset is not
        a terminal step: the game continues, so the value bootstrap must too.
        Conflating the two teaches the critic that the world is about to
        vanish, which shows up as a value function that never learns anything.

        ``reset`` does mark an inferred reset, because it invalidates the
        transition the curiosity head would otherwise learn from: across a
        respawn there is no "next frame" that follows from the last one.

        ``action_vector`` exists so the curiosity net can be trained on batches
        after the rollout, instead of doing a backward pass inside the control
        loop.
        """
        if self.size >= self.capacity:
            raise RuntimeError("RolloutBuffer overflow: add() past capacity")
        i = self.size
        self._context[i] = context
        self._summary[i] = summary
        self._window[i] = window
        self._held[i] = held
        self._turn[i] = int(direction)
        self._speed[i] = int(speed)
        self._tap[i] = int(tap)
        self._logp[i] = float(log_prob)
        self._value[i] = float(value)
        self._reward[i] = float(reward)
        self._done[i] = 1.0 if terminal else 0.0
        self._reset[i] = 1.0 if reset else 0.0
        if action_vector is not None:
            if self._action_vec is None:
                self._action_vec_dim = int(action_vector.size)
                self._action_vec = np.zeros((self.capacity,
                                             self._action_vec_dim),
                                            dtype=np.float32)
            self._action_vec[i] = action_vector
        self.size += 1
        if terminal:
            self.episodes += 1

    # ---- tensors ----
    def _tensor(self, array: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(array[:self.size], dtype=dtype, device=self.device)

    def tensors(self) -> Dict[str, torch.Tensor]:
        out = {
            "context": self._tensor(self._context),
            "summary": self._tensor(self._summary),
            "hidden": self._tensor(self._window),
            "held": self._tensor(self._held),
            "turn": self._tensor(self._turn, torch.long),
            "speed": self._tensor(self._speed, torch.long),
            "tap": self._tensor(self._tap, torch.long),
            "logp": self._tensor(self._logp),
            "value": self._tensor(self._value),
            "reward": self._tensor(self._reward),
            "done": self._tensor(self._done),
            "reset": self._tensor(self._reset),
        }
        if self._action_vec is not None:
            out["action_vector"] = self._tensor(self._action_vec)
        return out

    def rewards(self) -> np.ndarray:
        return self._reward[:self.size]

    def values(self) -> np.ndarray:
        return self._value[:self.size]

    def dones(self) -> np.ndarray:
        return self._done[:self.size]

    # ---- sequences ----
    def sequence_batches(self, seq_len: int, minibatch: int,
                         rng: np.random.Generator,
                         max_sequences: Optional[int] = None
                         ) -> Iterator[Tuple[List[int], torch.Tensor]]:
        """
        Yield (start_indices, observation_indices) for one minibatch.

        A minibatch is a set of whole sequences of consecutive steps.  The
        memory window stored at each start index seeds the transformer, so the
        forward pass sees exactly the history the policy had at the time, and
        gradients are cut at the sequence boundary.  That is the standard
        truncated-BPTT arrangement, and it is what makes a transformer PPO
        update both correct and cheap.
        """
        seq_len = max(2, int(seq_len))
        n_sequences = self.size // seq_len
        if n_sequences <= 0:
            return
        if max_sequences is not None:
            n_sequences = min(n_sequences, max(1, int(max_sequences)))

        starts = np.arange(n_sequences, dtype=np.int64) * seq_len
        rng.shuffle(starts)
        per_batch = max(1, int(minibatch) // seq_len)
        for begin in range(0, n_sequences, per_batch):
            chunk = starts[begin:begin + per_batch]
            if chunk.size == 0:
                continue
            # (T, B) index grid.
            grid = (chunk[None, :] + np.arange(seq_len, dtype=np.int64)[:, None])
            yield list(chunk), torch.as_tensor(grid, dtype=torch.long,
                                               device=self.device)


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                last_value: float, gamma: float, lam: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalised advantage estimation, in numpy and backwards, which is all it
    needs to be: this is a few thousand floats, not a tensor job.

    ``dones`` masks the value bootstrap on a genuine terminal step only.  A
    time-limit truncation is bootstrapped like any other step, because the game
    did not actually end - treating a segment boundary as a real end is how a
    value function learns that the world is about to disappear.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in range(n - 1, -1, -1):
        next_value = last_value if t == n - 1 else float(values[t + 1])
        not_done = 1.0 - float(dones[t])
        delta = float(rewards[t]) + gamma * next_value * not_done - float(values[t])
        last = delta + gamma * lam * not_done * last
        advantages[t] = last
    returns = advantages + values
    return advantages, returns


def normalise_advantages(advantages: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance advantages.

    The guard against a degenerate rollout is the point: with a single sample,
    or with every advantage identical, the standard deviation is zero and
    dividing by it produces either NaN or - if the code special-cases it - a
    gradient of exactly zero.  That silent zero is the classic way a run
    "trains" for hours without changing at all, so the epsilon here is
    deliberate and the caller is told to collect a bigger rollout instead.
    """
    if advantages.size == 0:
        return advantages
    std = float(advantages.std())
    if std < 1e-6:
        return advantages - float(advantages.mean())
    return (advantages - float(advantages.mean())) / (std + 1e-8)
