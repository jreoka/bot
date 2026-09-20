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
    # The held-key head is a set of independent Bernoulli bits, conditioned on
    # at most `max_held` of them being set.  Why independent bits rather than a
    # softmax over combinations:
    #
    #   * There are 2**11 combinations. A softmax over them can never be
    #     explored; independent bits let "walk" and "also sprint" be learned
    #     separately and then combined, which is what a player does.
    #   * The gradient of an independent Bernoulli log-probability with respect
    #     to *its own* logit has the right sign for every key, taken or not.
    #
    # The cap is part of the distribution rather than a post-hoc edit of the
    # sample (see `held_log_prob`), because PPO compares the probability the
    # policy assigned to an action during collection with the probability it
    # assigns to that same action during the update. A sample that is modified
    # after its log-probability is recorded makes those two numbers disagree,
    # and the ratio - and therefore the gradient - is meaningless.
    def sample_held(self, logits: torch.Tensor):
        """
        Draw a key-set from the capped distribution, and log its probability.

        Sampling goes through the same category set the update scores with, so
        the recorded log-probability is reproducible under the policy that is
        being optimised.
        """
        masks = self._held_categories(logits)
        scores = self.held_logits_of_masks(logits, masks)
        index = torch.distributions.Categorical(logits=scores).sample()
        bits = masks[index]
        # `held_log_prob` is reused rather than recomputed inline, so sampling
        # and scoring cannot drift apart again.
        return bits, self.held_log_prob(logits, bits)

    # ---- the distribution the log-probabilities are taken under -----------
    #
    # Sampling and scoring *must* be the same distribution, because PPO divides
    # one by the other. They were not: the sampler drew independent Bernoulli
    # bits and then silently dropped keys past the cap, while the update scored
    # the surviving bits as if they had been drawn from those Bernoullis
    # directly. Any step where the cap fired recorded a log-probability the
    # update could not reproduce - measured at up to ~6 nats, a ratio of 400 -
    # so every update was dominated by a truncation artefact instead of by
    # advantage. The symptoms were a fitted KL near -1.2, three quarters of
    # samples outside the trust region, an explained variance of -10^4, and a
    # policy that never moved.
    #
    # The fix is one distribution used by both paths:
    #
    #     P(mask) is proportional to prod_{i in mask} p_i * prod_{i not in mask} (1 - p_i)
    #     ... restricted to |mask| <= max_held, renormalised over the kept masks.
    #
    # which is "independent bits, conditioned on holding at most k keys".  It is
    # what the sampler always meant to draw from, it is exact, and it can be
    # scored cheaply by enumerating the kept masks: C(9, <=4) = 256 of them for
    # a nine-key whitelist.
    def _held_categories(self, logits: torch.Tensor) -> torch.Tensor:
        """Every key-mask with at most ``max_held`` bits set, as (n_cat, n_keys)."""
        n_keys = int(logits.shape[-1])
        cached = getattr(self, "_held_mask_cache", None)
        if (cached is None or cached.shape[-1] != n_keys
                or cached.shape[0] == 0
                or cached.device != logits.device
                or cached.dtype != logits.dtype):
            masks = []
            for mask in range(1 << n_keys):
                if bin(mask).count("1") <= self.max_held:
                    masks.append([(mask >> i) & 1 for i in range(n_keys)])
            cached = torch.tensor(masks, dtype=logits.dtype,
                                  device=logits.device)
            self._held_mask_cache = cached
        return cached

    @staticmethod
    def _mask_codes(masks: torch.Tensor) -> torch.Tensor:
        """(n_cat, n_keys) bits -> (n_cat,) integer codes, for indexing."""
        n_keys = int(masks.shape[-1])
        weights = (1 << torch.arange(n_keys - 1, -1, -1,
                                     device=masks.device, dtype=torch.long))
        return (masks > 0.5).to(torch.long) @ weights

    def held_logits_of_masks(self, logits: torch.Tensor,
                             masks: torch.Tensor) -> torch.Tensor:
        """
        Unnormalised log-probability of every category.

        ``logits`` is (..., n_keys) and ``masks`` is (n_cat, n_keys); the result
        is (..., n_cat).  The per-key term is the Bernoulli log-probability of
        that bit, so this is the independent-bits model restricted to the
        categories that respect the cap.
        """
        n_cat = int(masks.shape[0])
        # Explicit broadcast to (..., n_cat, n_keys), then the per-key Bernoulli
        # log-probability of that bit: -softplus(-z) for a set bit and
        # -softplus(z) for an unset one, which is log sigmoid(z) and
        # log(1 - sigmoid(z)) written the numerically stable way round.
        wide = logits.unsqueeze(-2).expand(*logits.shape[:-1], n_cat,
                                           logits.shape[-1])
        per_key = -(torch.nn.functional.softplus(
            torch.where(masks > 0.5, -wide, wide)))
        return per_key.sum(dim=-1)

    def held_log_prob(self, logits: torch.Tensor,
                      held: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of a key-set under the capped distribution.

        Shared by the sampler and the update, which is the whole point: the
        importance ratio PPO computes is only meaningful when the behaviour
        distribution and the scored distribution are the same one.
        """
        masks = self._held_categories(logits)
        scores = self.held_logits_of_masks(logits, masks)
        log_norm = torch.logsumexp(scores, dim=-1, keepdim=True)
        codes = self._mask_codes(masks)
        target = self._mask_codes((held > 0.5).to(masks.dtype))
        position = (codes.unsqueeze(0) == target.unsqueeze(-1))
        selected = (scores * position.to(scores.dtype)).sum(dim=-1)
        # A mask outside the category set cannot come out of `sample`, but if one
        # ever did, its probability is zero rather than a silently wrong number.
        in_set = position.any(dim=-1)
        return torch.where(in_set, selected - log_norm.squeeze(-1),
                           torch.full_like(selected, -1e9))

    def held_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Entropy of the capped distribution, in nats."""
        masks = self._held_categories(logits)
        scores = self.held_logits_of_masks(logits, masks)
        log_probs = torch.log_softmax(scores, dim=-1)
        return -(log_probs.exp() * log_probs).sum(dim=-1)

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
        # No post-hoc truncation here: the cap is already part of the
        # distribution `sample_held` drew from, so the action needs no edit and
        # its log-probability is exactly the one the update will recompute.
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
