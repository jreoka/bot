"""
The model: how the bot sees, remembers, and decides.

Shape of the whole thing:

    frame  (C, S, S)  ->  CNN  ->  embed (E,)  ->  GRU  ->  heads
                                              |
                                              +-> novelty / forward model

Why this and not a transformer over stacked frames: a transformer's cost grows
with the square of the number of frames in its window, and every control step
re-runs the whole window.  A CNN plus a recurrent state costs the same for one
frame as for a hundred, because the state carries the history.  On a CPU that
is the difference between a few milliseconds and a few hundred per step, and it
is what makes this bot usable while the game is running.

Everything here is deliberately small: around 400k parameters in total, which
trains in well under a second per update on a laptop core.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ARCH_NAME, Config


# =============================================================================
# Encoder
# =============================================================================

class FrameEncoder(nn.Module):
    """
    (B, C, S, S) -> (B, embed_dim).

    Strided convolutions and a fixed adaptive pool, so the output width does
    not depend on the input resolution: changing frame_size does not change the
    shape of anything downstream and does not invalidate the rest of the model.
    """

    def __init__(self, in_channels: int, width: int, embed_dim: int):
        super().__init__()
        w = int(width)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, w, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w * 2, w * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.proj = nn.Linear(w * 2 * 4 * 4, int(embed_dim))
        self.norm = nn.LayerNorm(int(embed_dim))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.proj.weight, gain=1.0)
        nn.init.zeros_(self.proj.bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(self.conv(frames)))


# =============================================================================
# Policy: GRU + factored heads
# =============================================================================

class ActorCritic(nn.Module):
    """
    Recurrent policy with three independent decision heads.

    The held-key head is a set of independent Bernoulli bits with a top-k
    constraint, not a softmax over combinations.  Two reasons: the number of
    combinations of eleven keys is 2048 and the bot would never explore it, and
    a Bernoulli factorisation lets the useful sub-decisions (walk, and also
    sprint) be learned separately and then combined at sample time - which is
    exactly what a player does.

    ``sample`` and ``sequence_log_prob`` must agree exactly, because PPO
    compares the two.  Both apply the same top-k rule, so they do.
    """

    def __init__(self, cfg: Config, head_sizes: Tuple[int, int, int],
                 observation_channels: int):
        super().__init__()
        self.cfg = cfg
        self.arch = ARCH_NAME
        self.head_sizes = tuple(int(h) for h in head_sizes)
        self.n_hold, self.n_turn, self.n_tap = self.head_sizes
        self.n_keys = max(1, self.n_hold - 1)
        self.max_held = max(1, min(int(cfg.max_held_keys), self.n_keys))

        self.encoder = FrameEncoder(observation_channels, cfg.cnn_width,
                                    cfg.embed_dim)
        self.gru = nn.GRUCell(int(cfg.embed_dim), int(cfg.hidden_dim))

        h = int(cfg.hidden_dim)
        self.hold_head = nn.Linear(h, self.n_keys)
        self.turn_head = nn.Linear(h, self.n_turn)
        self.tap_head = nn.Linear(h, self.n_tap)
        self.value_head = nn.Linear(h, 1)

        # A near-uniform start for the decisions: a policy that begins already
        # certain of one action never explores its way out of it.
        for head in (self.hold_head, self.turn_head, self.tap_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        # The held-key head starts biased *against* pressing, so the keys a
        # policy does press have to be earned rather than being the default.
        # Without this, every key drifts up together under the top-k rule
        # (each key is only in the running when its logit is high, so pushing
        # all of them up is a stable fixed point) and the bot ends up holding
        # every key it owns, which moves it nowhere.
        nn.init.constant_(self.hold_head.bias, -1.5)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    # ---- the held-key decision ----
    #
    # Each key is an independent Bernoulli bit. Two reasons this beats a softmax
    # over key combinations, and one reason it beats a constrained top-k rule
    # (which this file used first, and which is subtly and silently wrong):
    #
    #   * There are 2**11 combinations. A softmax over them can never be
    #     explored; independent bits let "walk" and "also sprint" be learned
    #     separately and then combined, which is what a player does.
    #   * The gradient of an independent Bernoulli log-probability with respect
    #     to *its own* logit has the right sign for every key, taken or not.
    #     The top-k rule does not: its normaliser couples every key to every
    #     other, and an untaken key ends up with the same gradient sign as a
    #     taken one. A policy trained that way drifts toward holding every key
    #     at once, which moves it nowhere, and the symptom is indistinguishable
    #     from "this bot cannot learn".
    #
    # The cap on simultaneous keys is applied when sampling, by keeping the most
    # likely keys. That is a truncation rather than an exact constrained
    # distribution, and it is deliberate - the exact version is the coupling
    # described above.
    def sample_held(self, logits: torch.Tensor):
        """(..., n_keys) -> bits, and the log-probability of those bits."""
        probabilities = torch.sigmoid(logits)
        bits = torch.bernoulli(probabilities)
        return bits, self.held_log_prob(logits, bits)

    def truncate_held(self, bits: torch.Tensor, logits: torch.Tensor
                      ) -> torch.Tensor:
        """Keep at most max_held keys: the ones the policy is most sure of."""
        if bits.shape[-1] <= self.max_held:
            return bits
        order = torch.argsort(logits, dim=-1, descending=True)
        keep = torch.zeros_like(bits)
        keep.scatter_(-1, order[..., :self.max_held], 1.0)
        return bits * keep

    def held_log_prob(self, logits: torch.Tensor,
                      held: torch.Tensor) -> torch.Tensor:
        """Log-probability of a set of bits, under the independent bits."""
        return (-F.binary_cross_entropy_with_logits(
            logits, held, reduction="none")).sum(dim=-1)

    def held_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Sum of the per-key Bernoulli entropies."""
        p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        return -(p * p.log() + (1.0 - p) * (1.0 - p).log()).sum(dim=-1)

    # ---- forward paths ----
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames)

    def heads(self, hidden: torch.Tensor):
        return (self.hold_head(hidden), self.turn_head(hidden),
                self.tap_head(hidden), self.value_head(hidden).squeeze(-1))

    @torch.no_grad()
    def step(self, obs: torch.Tensor, hidden: torch.Tensor):
        """
        One control step: (1, C, S, S) and the previous hidden state in.

        Returns the sampled decision, its log-probability, the value estimate,
        the new hidden state, and the frame embedding (reused by the novelty
        and forward-model heads, so the encoder only runs once per step).
        """
        embed = self.encode(obs)
        hidden = self.gru(embed, hidden)
        hold_logits, turn_logits, tap_logits, value = self.heads(hidden)

        bits, log_prob = self.sample_held(hold_logits)
        bits = self.truncate_held(bits, hold_logits)
        turn_dist = torch.distributions.Categorical(logits=turn_logits)
        tap_dist = torch.distributions.Categorical(logits=tap_logits)
        turn = turn_dist.sample()
        tap = tap_dist.sample()
        log_prob = (log_prob + turn_dist.log_prob(turn)
                    + tap_dist.log_prob(tap))
        return (bits, turn, tap), log_prob, value, hidden, embed

    def sequence_log_prob(self, hidden: torch.Tensor,
                          held: torch.Tensor, turn: torch.Tensor,
                          tap: torch.Tensor):
        """
        Score a recorded sequence under the current policy.

        ``hidden`` is (T, B, H) - the recurrent state *before* each step,
        exactly as it was during collection.  Recomputing the states instead of
        storing them would make the update off-policy against the policy that
        produced the data, which is the classic recurrent-PPO trap; storing one
        small vector per step avoids it for a few hundred bytes a step.
        """
        hold_logits, turn_logits, tap_logits, value = self.heads(hidden)
        log_prob = self.held_log_prob(hold_logits, held)
        turn_dist = torch.distributions.Categorical(logits=turn_logits)
        tap_dist = torch.distributions.Categorical(logits=tap_logits)
        log_prob = log_prob + turn_dist.log_prob(turn) + tap_dist.log_prob(tap)
        entropy = (self.held_entropy(hold_logits)
                   + turn_dist.entropy() + tap_dist.entropy())
        return log_prob, entropy, value

    def initial_hidden(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, int(self.cfg.hidden_dim), device=device)


# =============================================================================
# Intrinsic-reward networks
# =============================================================================

class RunningScale:
    """
    Streaming mean and spread of a signal whose magnitude is unknown up front.

    The obvious implementation - Welford's algorithm - has a nasty property
    here: with only a handful of samples its variance estimate is tiny, so the
    first few values of a novelty signal get divided by a near-zero spread and
    come out enormous. Since those first values are exactly what the policy
    learns from, the bot starts by chasing a loud artefact.

    This uses exponential moving statistics with a floor on the spread, and a
    spread that can only *rise* for the first few samples. The result is that a
    novelty bonus starts near 1.0 and stays in that range, which is what makes
    the reward weights in the config mean something.
    """

    def __init__(self, initial: float = 1.0, decay: float = 0.99):
        self.mean = float(initial)
        self.var = float(initial) ** 2
        self.decay = float(decay)
        self.count = 0

    def update(self, x: float) -> None:
        self.count += 1
        self.mean += (1.0 - self.decay) * (x - self.mean)
        # A rising spread for the first samples, then a symmetric estimate.
        if self.count < 8:
            self.var = max(self.var, (x - self.mean) ** 2)
        else:
            self.var += (1.0 - self.decay) * ((x - self.mean) ** 2 - self.var)

    @property
    def std(self) -> float:
        """Spread, never allowed to collapse to zero."""
        return float(max(self.var ** 0.5, 0.25 * abs(self.mean), 1e-2))

    def state(self):
        return (self.mean, self.var, self.count)

    def load(self, state) -> None:
        self.mean, self.var, self.count = (float(state[0]), float(state[1]),
                                           int(state[2]))


class RandomNetworkDistillation(nn.Module):
    """
    RND: a predictor tries to match a fixed randomly-initialised network.

    Prediction error is high where the bot has seen little and low where it has
    seen a lot, which is a per-state novelty signal that needs no visit counts
    and no discretisation.  The predictor is trained online with plain SGD - a
    two-layer MLP does not need an adaptive optimiser, and this keeps its cost
    and memory flat forever.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, lr: float):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        for param in self.target.parameters():
            param.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.optimizer = torch.optim.SGD(self.predictor.parameters(), lr=float(lr))
        self.scale = RunningScale()
        self.last_error = 0.0
        self.last_novelty = 0.0

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        with torch.no_grad():
            target = self.target(x)
        if not train:
            # Only the error is wanted; do not build a graph for it.
            with torch.no_grad():
                error = F.mse_loss(self.predictor(x), target)
            value = float(error)
            self.last_error = value
            self.scale.update(value)
            return error
        prediction = self.predictor(x)
        error = F.mse_loss(prediction, target)
        value = float(error.detach())
        self.last_error = value
        self.scale.update(value)
        self.optimizer.zero_grad(set_to_none=True)
        error.backward()
        self.optimizer.step()
        return error.detach()

    def novelty(self, x: torch.Tensor, train: bool = False) -> float:
        """Scaled novelty: about 1.0 when this state is as new as usual."""
        error = float(self(x, train=train))
        self.last_novelty = error / self.scale.std
        return self.last_novelty

    def train_batch(self, x: torch.Tensor, steps: int = 1) -> float:
        """A few gradient steps on a batch, run during the PPO update."""
        loss_value = 0.0
        for _ in range(max(1, int(steps))):
            with torch.no_grad():
                target = self.target(x)
            prediction = self.predictor(x)
            error = F.mse_loss(prediction, target)
            self.optimizer.zero_grad(set_to_none=True)
            error.backward()
            self.optimizer.step()
            loss_value = float(error.detach())
        return loss_value


class ForwardModel(nn.Module):
    """
    Predicts the next frame embedding from the current one and the action.

    Two things come out of it:

    * **learning progress** - how much better this episode is going than
      previous episodes were, measured on the model's own prediction error.
      Paying for improvement rather than for error is what stops the bot from
      parking in front of the most chaotic thing it can find: once it
      understands a region, the error there stops paying.  This is the term
      that turns "wander around" into "work out what this part of the game
      does", and it is measured across whole episodes because a within-episode
      average just tracks recent noise and pays out forever.

    * **liveness** - a prediction error that suddenly jumps means the world
      changed in a way this model cannot account for.  The reset detector uses
      that, alongside the raw frame difference.
    """

    def __init__(self, embed_dim: int, action_dim: int, hidden: int, lr: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=float(lr))
        self.scale = RunningScale()
        self.last_error = 0.0

    def forward(self, embed: torch.Tensor, action: torch.Tensor,
                next_embed: torch.Tensor, train: bool = True) -> float:
        prediction = self.net(torch.cat([embed, action], dim=-1))
        error = F.mse_loss(prediction, next_embed)
        value = float(error.detach())
        self.last_error = value
        self.scale.update(value)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            error.backward()
            self.optimizer.step()
        return value

    def error_only(self, embed: torch.Tensor, action: torch.Tensor,
                   next_embed: torch.Tensor) -> float:
        """Prediction error without a gradient step, for use inside the loop."""
        with torch.no_grad():
            prediction = self.net(torch.cat([embed, action], dim=-1))
            error = F.mse_loss(prediction, next_embed)
        value = float(error)
        self.last_error = value
        self.scale.update(value)
        return value

    def train_batch(self, embed: torch.Tensor, action: torch.Tensor,
                    next_embed: torch.Tensor, steps: int = 1) -> float:
        loss_value = 0.0
        for _ in range(max(1, int(steps))):
            prediction = self.net(torch.cat([embed, action], dim=-1))
            error = F.mse_loss(prediction, next_embed)
            self.optimizer.zero_grad(set_to_none=True)
            error.backward()
            self.optimizer.step()
            loss_value = float(error.detach())
        return loss_value
