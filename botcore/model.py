"""
The model: how the bot sees, remembers, and decides.

There is exactly one network here, and everything the bot does comes out of it:

    frame stack (C, S, S) -> patch embedding -> tokens (N, E)
                                                  |
                            memory window (M, E) -+-> causal transformer
                                                       (RoPE attention + SwiGLU)
                                                              |
                                        +---------------------+------------------+
                                        |            |             |            |
                                     held keys    turn dir     turn speed      tap
                                        |            |             |            |
                                    value head   next-frame prediction (curiosity)

Every block is a SwiGLU feed-forward and a rotary-position causal
self-attention.  Nothing else is consulted: there is no separate RND network and
no separate forward model, because the transformer already predicts the next
frame's representation as one of its own heads, and that prediction error is the
curiosity signal.

Why a window and not the whole history
--------------------------------------
A decoder-only transformer over everything the bot has ever seen would both cost
more every step and never be trainable on a CPU.  What is kept instead is a
rolling window of the last `mem_tokens` frame representations plus the tokens of
the current frame, attended to causally.  The window is the transformer's memory;
it holds the same kind of history a recurrent state would, it costs a fixed
amount per step, and - unlike a hidden state - every PPO update can recompute it
from the rollout exactly as it was during collection.

Why the pattern is *causal*, and what that buys
-----------------------------------------------
The window in force at step t contains only frames from before t, so the decision
at t cannot see the future.  At update time the stored window is used verbatim,
which is what keeps the importance ratio in PPO meaningful: the same policy,
the same inputs, the same log-probability.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ARCH_NAME, Config


# =============================================================================
# Building blocks
# =============================================================================

class SwiGLU(nn.Module):
    """
    The gated feed-forward used by every block.

        SwiGLU(x) = W2( SiLU(W1 x) * W3 x )

    Two projections where a plain MLP has one, which is why the hidden width is
    kept to a small multiple of the embedding: the gating buys a lot of
    expressiveness per parameter, and the multiply is nearly free compared with
    the matmuls around it.
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        nn.init.orthogonal_(self.gate.weight, gain=1.0)
        nn.init.orthogonal_(self.up.weight, gain=1.0)
        # The output projection starts small so a freshly initialised block is
        # close to the identity: a transformer whose residual branches shout from
        # the first step is one the critic never catches up with.
        nn.init.orthogonal_(self.down.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class RotaryEmbedding(nn.Module):
    """
    Rotary position embedding: rotates query/key pairs by an angle that depends
    only on their *relative* distance.

    That property is what makes the memory window work at all.  The window shifts
    by one token every control step, so the absolute position of a given frame
    changes constantly; with RoPE the attention score between two frames depends
    on how far apart they are, not on where in the window they happen to sit.
    """

    def __init__(self, head_dim: int, max_tokens: int, base: float = 10000.0):
        super().__init__()
        # An even head width is required: the rotation is applied to pairs.
        if head_dim % 2 == 1:
            head_dim -= 1
        self.head_dim = max(2, int(head_dim))
        self.max_tokens = max(2, int(max_tokens))
        self.base = float(base)
        self._cache: Dict[tuple, torch.Tensor] = {}

    def _angles(self, tokens: int, device, dtype) -> torch.Tensor:
        key = (int(tokens), str(device), str(dtype))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(half, device=device,
                                                     dtype=torch.float32)
                                        / max(1, half)))
        positions = torch.arange(int(tokens), device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)          # (tokens, half)
        cos = torch.cos(angles).to(dtype)
        sin = torch.sin(angles).to(dtype)
        # One entry is enough in practice (a handful of shapes), so the cache is
        # bounded rather than growing with the run.
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = (cos, sin)
        return cos, sin

    @staticmethod
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        """(..., half, 2) -> (..., half, 2) rotated by a quarter turn."""
        first, second = x[..., 0], x[..., 1]
        return torch.stack((-second, first), dim=-1)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x is (B, tokens, heads, head_dim) -> the same shape, rotated."""
        tokens = int(x.shape[-3])
        cos, sin = self._angles(tokens, x.device, x.dtype)
        pairs = x.reshape(*x.shape[:-1], self.head_dim // 2, 2)
        rotated = pairs * cos.view(1, tokens, 1, -1, 1) \
            + self._rotate(pairs) * sin.view(1, tokens, 1, -1, 1)
        out = rotated.reshape(*x.shape[:-1], self.head_dim)
        # An odd head width has one dimension that cannot be paired; leave it
        # alone rather than silently dropping it.
        if out.shape[-1] != x.shape[-1]:
            out = torch.cat([out, x[..., self.head_dim:]], dim=-1)
        return out


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask and rotary positions."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = max(1, int(heads))
        self.head_dim = max(2, int(dim) // self.heads)
        inner = self.head_dim * self.heads
        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.out = nn.Linear(inner, dim, bias=False)
        self.dropout = float(dropout)
        nn.init.orthogonal_(self.qkv.weight, gain=1.0)
        nn.init.orthogonal_(self.out.weight, gain=0.5)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, tokens, _dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = rope.apply(query)
        key = rope.apply(key)
        # (B, H, T, D) so the attention weights read naturally as T x T.
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None)
        attended = attended.transpose(1, 2).reshape(batch, tokens, -1)
        return self.out(attended)


class TransformerBlock(nn.Module):
    """Pre-norm attention + SwiGLU, both residual."""

    def __init__(self, dim: int, heads: int, ffn_hidden: int,
                 dropout: float = 0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# =============================================================================
# Encoder: frame stack -> patch tokens
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    (B, C, S, S) -> (B, tokens, embed_dim).

    One strided convolution and nothing else: the transformer does the sequence
    work, so this only has to turn pixels into vectors.  It is deliberately not
    a deep conv stack - a hierarchy of convolutions in front of the transformer
    would mean two models, and the point of this build is that there is one.
    """

    def __init__(self, in_channels: int, embed_dim: int, frame_size: int,
                 patch: int, width: int):
        super().__init__()
        patch = max(2, int(patch))
        self.patch = patch
        self.grid = max(1, (int(frame_size) + patch - 1) // patch)
        self.tokens = self.grid * self.grid
        self.proj = nn.Conv2d(int(in_channels), int(width), kernel_size=patch,
                              stride=patch, bias=True)
        self.norm = nn.LayerNorm(int(width))
        self.out = nn.Linear(int(width), int(embed_dim), bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.orthogonal_(self.out.weight, gain=1.0)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        size = int(frames.shape[-1])
        if size % self.patch != 0:
            # Pad once at the edge rather than re-deriving the grid: the token
            # count downstream must stay fixed for a fixed frame_size.
            pad = self.patch - (size % self.patch)
            frames = F.pad(frames, (0, pad, 0, pad))
        patches = self.proj(frames)
        patches = patches.flatten(2).transpose(1, 2)        # (B, tokens, width)
        return self.out(F.silu(self.norm(patches)))


# =============================================================================
# Policy: one transformer, every head
# =============================================================================

class ActorCritic(nn.Module):
    """
    The whole bot: a SwiGLU + RoPE causal transformer over a memory window, and
    the heads that read its last position.

    Decisions are factored the way a player's are:

        held keys   - independent Bernoulli bits under a top-k cap
        turn        - which way to swing the view
        turn speed  - how far, as a multiple of the step size the bot keeps for
                      itself (this is the mouse speed, and the bot owns it; see
                      `mouse_step`)
        tap/click   - one momentary button

    ``sample`` and ``sequence_log_prob`` must agree exactly, because PPO compares
    the two.  They share every distribution: the capped key-set distribution, and
    the same categorical heads.
    """

    def __init__(self, cfg: Config, head_sizes: Tuple[int, int, int, int],
                 observation_channels: int):
        super().__init__()
        self.cfg = cfg
        self.arch = ARCH_NAME
        self.head_sizes = tuple(int(h) for h in head_sizes)
        self.n_hold, self.n_turn, self.n_speed, self.n_tap = self.head_sizes
        self.n_keys = max(1, self.n_hold - 1)
        self.max_held = max(1, min(int(cfg.max_held_keys), self.n_keys))

        self.embed_dim = int(cfg.embed_dim)
        self.mem_tokens = max(1, int(cfg.mem_tokens))
        self.patch = max(2, int(cfg.patch_size))
        self.patch_embed = PatchEmbedding(
            int(observation_channels), self.embed_dim, int(cfg.frame_size),
            self.patch, int(cfg.patch_width))
        self.frame_tokens = int(self.patch_embed.tokens)

        # The memory slot for a step that has not happened yet. Learned, so the
        # model can decide for itself what "the start of a run" looks like.
        self.bos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.bos, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, int(cfg.attention_heads),
                             int(cfg.ffn_hidden), float(cfg.attention_dropout))
            for _ in range(max(1, int(cfg.transformer_layers)))
        ])
        self.rope = RotaryEmbedding(self.embed_dim // max(
            1, int(cfg.attention_heads)),
            self.mem_tokens + self.frame_tokens + 2)

        self.final_norm = nn.LayerNorm(self.embed_dim)

        h = self.embed_dim
        self.hold_head = nn.Linear(h, self.n_keys)
        self.turn_head = nn.Linear(h, self.n_turn)
        self.speed_head = nn.Linear(h, self.n_speed)
        self.tap_head = nn.Linear(h, self.n_tap)
        self.value_head = nn.Linear(h, 1)

        # A near-uniform start for the decisions: a policy that begins already
        # certain of one action never explores its way out of it.
        for head in (self.hold_head, self.turn_head, self.speed_head,
                     self.tap_head):
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

        # ---- the bot's own mouse speed ----
        #
        # One pixel step the bot keeps for itself, in *world* pixels per unit of
        # speed.  It is a buffer rather than a parameter because it lives outside
        # the differentiable graph: the policy says "turn left at speed 2x" and
        # this says what 1x is worth, which is exactly the kind of quantity a
        # learner should be allowed to move on its own.  `set_mouse_step` pins it
        # when the user knows better (--mouse-turn, or a --calibrate recording).
        self.register_buffer("mouse_step",
                             torch.tensor([float(max(1, cfg.mouse_turn_start))],
                                          dtype=torch.float32))
        self.mouse_step_min = float(max(1, cfg.mouse_turn_min))
        self.mouse_step_max = float(max(self.mouse_step_min,
                                        cfg.mouse_turn_max))

        # ---- the curiosity head ----
        #
        # Predicts the *next* frame's representation. Its error is the novelty
        # signal the reward is built from, so there is no second network to
        # train, store or checkpoint. The input is detached: the reward signal
        # must not be able to change the policy's features to make itself look
        # better.
        self.transition_norm = nn.LayerNorm(h)
        self.transition = nn.Sequential(
            nn.Linear(h, int(cfg.transition_hidden)), nn.SiLU(),
            nn.Linear(int(cfg.transition_hidden), self.embed_dim),
        )
        nn.init.orthogonal_(self.transition[0].weight, gain=1.0)
        nn.init.zeros_(self.transition[0].bias)
        nn.init.orthogonal_(self.transition[2].weight, gain=0.5)
        nn.init.zeros_(self.transition[2].bias)
        self.transition_optimizer = torch.optim.Adam(
            list(self.transition.parameters())
            + list(self.transition_norm.parameters()),
            lr=float(cfg.transition_lr))

    # ---- mouse speed ----
    def mouse_step_value(self) -> int:
        """Pixels per 1x turn step, as an integer, for the input layer."""
        return max(1, int(round(float(self.mouse_step.item()))))

    @torch.no_grad()
    def set_mouse_step(self, pixels: Optional[float]) -> None:
        """Pin the base turn step (the user knows the game better than we do)."""
        if pixels is None:
            return
        value = float(pixels)
        if value <= 0:
            return
        self.mouse_step.fill_(min(max(value, self.mouse_step_min),
                                  self.mouse_step_max))

    @torch.no_grad()
    def nudge_mouse_step(self, factor: float) -> None:
        """
        Move the base turn step, staying inside the configured bounds.

        Used by the session's speed correction: when turns repeatedly fail to
        move the picture, this is the knob that closes the gap. Bounded on both
        ends so a bad estimate cannot turn a small correction into a spin.
        """
        value = float(self.mouse_step.item()) * float(factor)
        self.mouse_step.fill_(min(max(value, self.mouse_step_min),
                                  self.mouse_step_max))

    # ---- the held-key decision ----
    #
    # The held-key head is a set of independent Bernoulli bits, conditioned on
    # at most `max_held` of them being set. Why independent bits rather than a
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
        """Frame stack -> patch tokens."""
        return self.patch_embed(frames)

    def initial_hidden(self, batch: int, device) -> torch.Tensor:
        """The memory window at the start of a run: every slot is the BOS token."""
        return self.bos.detach().expand(int(batch), self.mem_tokens,
                                        self.embed_dim).clone().to(device)

    def advance(self, hidden: torch.Tensor,
                tokens: torch.Tensor) -> torch.Tensor:
        """
        Push one step's frame tokens into the window.

        The window is appended to and rolled; the new step's representation is
        the mean of its own tokens, which is what the next step will attend to.
        Rolling rather than growing is what keeps the per-step cost flat.
        """
        summary = tokens.mean(dim=-2, keepdim=True)
        return torch.cat([hidden, summary], dim=-2)[..., -self.mem_tokens:, :]

    def forward(self, hidden: torch.Tensor, tokens: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the transformer over the memory window plus this step's tokens.

        ``hidden`` is (..., M, E) and ``tokens`` is (..., N, E).  Returns
        (new_hidden, context, features):

        * ``new_hidden`` is the window for the *next* step, with the mean of
          this step's tokens appended (see `advance`);
        * ``context`` is the pooled representation of the window, used for the
          value estimate and the transition head;
        * ``features`` is the last position's output, which is what the decision
          heads read: it has attended to every frame token just encoded.

        ``hidden`` may carry leading batch dimensions (the PPO update passes
        (T, B, M, E)); those are folded into one attention batch, because the
        window of one step of one sequence must never see another's.
        """
        window = self.advance(hidden, tokens)                 # (..., M + 1, E)
        leading = window.shape[:-2]
        rows = 1
        for size in leading:
            rows *= int(size)
        dim = int(window.shape[-1])
        frame_tokens = int(tokens.shape[-2])
        seq = torch.cat([window, tokens], dim=-2)             # (..., M+1+N, E)
        total = int(seq.shape[-2])
        flat = seq.reshape(rows, total, dim)
        # Causal: a frame may only see itself and what came before it.
        mask = torch.ones(total, total, dtype=torch.bool,
                          device=seq.device).tril()
        x = flat
        for block in self.blocks:
            x = block(x, self.rope, mask)
        x = self.final_norm(x)
        x = x.reshape(*leading, total, dim)
        context = x[..., :-frame_tokens, :].mean(dim=-2)
        features = x[..., -1, :]
        return window, context, features

    def heads(self, features: torch.Tensor, context: torch.Tensor):
        """Every decision head, plus the value, from one forward pass."""
        return (self.hold_head(features), self.turn_head(features),
                self.speed_head(features), self.tap_head(features),
                self.value_head(context).squeeze(-1))

    def predict_next(self, context: torch.Tensor) -> torch.Tensor:
        """The curiosity head: what the next frame's representation should be."""
        return self.transition(self.transition_norm(context.detach()))

    @torch.no_grad()
    def step(self, obs: torch.Tensor, hidden: torch.Tensor):
        """
        One control step: (1, C, S, S) and the memory window in.

        Returns the sampled decision, its log-probability, the value estimate,
        the window for the next step, the pooled context (reused by the
        curiosity head), and this frame's token summary (which is what the next
        step's transition error is measured against).
        """
        tokens = self.encode(obs)
        new_hidden, context, features = self.forward(hidden, tokens)
        hold_logits, turn_logits, speed_logits, tap_logits, value = \
            self.heads(features, context)

        bits, log_prob = self.sample_held(hold_logits)
        # No post-hoc truncation here: the cap is already part of the
        # distribution `sample_held` drew from, so the action needs no edit and
        # its log-probability is exactly the one the update will recompute.
        turn_dist = torch.distributions.Categorical(logits=turn_logits)
        speed_dist = torch.distributions.Categorical(logits=speed_logits)
        tap_dist = torch.distributions.Categorical(logits=tap_logits)
        direction = turn_dist.sample()
        speed = speed_dist.sample()
        tap = tap_dist.sample()
        log_prob = (log_prob + turn_dist.log_prob(direction)
                    + speed_dist.log_prob(speed) + tap_dist.log_prob(tap))
        summary = tokens.mean(dim=-2)
        return ((bits, direction, speed, tap), log_prob, value, new_hidden,
                context, summary)

    def value_only(self, obs: torch.Tensor, hidden: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Value of an observation given the window - the GAE bootstrap."""
        tokens = self.encode(obs)
        new_hidden, context, features = self.forward(hidden, tokens)
        _h, _t, _s, _a, value = self.heads(features, context)
        return value, new_hidden

    def sequence_log_prob(self, hidden: torch.Tensor, summary: torch.Tensor,
                          held: torch.Tensor, direction: torch.Tensor,
                          speed: torch.Tensor, tap: torch.Tensor):
        """
        Score a recorded sequence under the current policy.

        ``hidden`` is (T, B, M, E) - the memory window *before* each step,
        exactly as it was during collection - and ``summary`` is (T, B, E), the
        representation of the frame that step saw.  The frame is rebuilt from its
        stored summary rather than re-encoded, so the transformer reads the same
        input in both paths and the same computation is scored as was sampled.
        Recomputing the window instead of storing it would make the update
        off-policy against the policy that produced the data, which is the
        classic recurrent-PPO trap; storing one window per step costs a few
        hundred kilobytes per rollout and removes the problem.
        """
        tokens = summary.unsqueeze(-2)
        _window, context, features = self.forward(hidden, tokens)
        hold_logits, turn_logits, speed_logits, tap_logits, value = \
            self.heads(features, context)
        log_prob = self.held_log_prob(hold_logits, held)
        turn_dist = torch.distributions.Categorical(logits=turn_logits)
        speed_dist = torch.distributions.Categorical(logits=speed_logits)
        tap_dist = torch.distributions.Categorical(logits=tap_logits)
        log_prob = (log_prob + turn_dist.log_prob(direction)
                    + speed_dist.log_prob(speed) + tap_dist.log_prob(tap))
        entropy = (self.held_entropy(hold_logits)
                   + turn_dist.entropy() + speed_dist.entropy()
                   + tap_dist.entropy())
        return log_prob, entropy, value, context


# =============================================================================
# Reward-side statistics
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
