"""
The PPO update, and the guards that keep it from eating the run.

Why the defaults here look different from a typical PPO script: on a CPU the
update competes with the game for the machine, and a long update is directly
harmful.  While an update runs, no frames are being collected, so the bot is
acting on a stale view of the game when it comes back - and if updates grow,
the run degrades in a way that looks exactly like "it stalls after a while".

Three mechanisms keep that bounded:

* Minibatches are whole short sequences, so the backward pass is small and
  cache-friendly and the recurrence is truncated correctly.
* Every update is timed, and the *next* update decides how many minibatches it
  can afford from the measured cost.  A slow machine automatically does fewer
  passes instead of falling further behind.
* The status line separates "time in the game" from "time in the update" and
  says which one is losing, because guessing is how this gets debugged by
  watching a frozen log.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .model import ActorCritic
from .replay import RolloutBuffer, compute_gae, normalise_advantages


class Trainer:
    """Owns the policy, the optimiser, and the measured cost of updating."""

    def __init__(self, cfg: Config, policy: ActorCritic,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.policy = policy
        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=float(cfg.adam_lr), eps=float(cfg.adam_eps))

        # Smoothed cost model, used to size the next update.
        self.seconds_per_minibatch = 0.02
        self.last_update_seconds = 0.0
        self.last_metrics: Dict[str, float] = {}
        self.updates = 0
        self.minibatches_run = 0
        self.budget_hits = 0
        self.skipped_minibatches = 0
        # Running scale of the reward, so the value function fits something of
        # order one whatever the units the intrinsic reward happens to use.
        # Without this the critic is asked to regress returns whose magnitude
        # drifts as the reward terms decay, which is a moving target it never
        # catches - and an uncaught critic means advantages are noise, which
        # means the policy never learns anything at all.
        self.reward_scale = float(cfg.reward_scale)
        self.reward_scale_ready = False

    def _normalise_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
        Rescale rewards so the *return* is of order one.

        Dividing by the mean reward magnitude is not enough: with a discount
        factor near one, a per-step reward of 0.5 becomes a return in the tens,
        and the critic is then asked to regress a target far outside the range
        its output naturally occupies. Fitting that takes many updates, and
        until it does the advantages are dominated by critic error - which is
        the state in which a run "trains" for hours and never learns a policy.
        """
        magnitude = float(np.abs(rewards).mean())
        if magnitude > 0:
            if not self.reward_scale_ready:
                self.reward_scale = magnitude
                self.reward_scale_ready = True
            else:
                self.reward_scale = (0.95 * self.reward_scale
                                     + 0.05 * magnitude)
        # 1/(1-gamma) is the factor by which a steady per-step reward
        # accumulates into a return.
        horizon = 1.0 / max(1e-3, 1.0 - self.cfg.gamma)
        scale = max(self.reward_scale * horizon, float(self.cfg.reward_scale))
        return rewards / scale

    # =====================================================================
    # Collection
    # =====================================================================
    def act(self, observation: np.ndarray, hidden: torch.Tensor
            ) -> Tuple[Tuple[np.ndarray, int, int], float, float,
                       torch.Tensor, np.ndarray, torch.Tensor]:
        """
        One decision.

        Returns the action, its log-probability, the value estimate, the new
        hidden state, the frame embedding and the *pre-step* hidden state.  The
        embedding is handed back so the reward nets can reuse it: the encoder
        runs once per step, not once per consumer.
        """
        obs = torch.as_tensor(observation, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        self.policy.eval()
        with torch.no_grad():
            (held, turn, tap), log_prob, value, new_hidden, embed = \
                self.policy.step(obs, hidden)
        action = (held.squeeze(0).cpu().numpy().astype(np.float32),
                  int(turn.item()), int(tap.item()))
        return (action, float(log_prob.item()), float(value.item()),
                new_hidden, embed.squeeze(0).cpu().numpy(),
                hidden.squeeze(0).cpu().numpy())
    # =====================================================================
    # Update
    # =====================================================================
    def update(self, buffer: RolloutBuffer, last_value: float
               ) -> Dict[str, float]:
        cfg = self.cfg
        if len(buffer) < 2:
            return {}

        rewards = self._normalise_rewards(buffer.rewards())
        values = buffer.values()
        dones = buffer.dones()
        advantages, returns = compute_gae(
            rewards, values, dones, last_value, cfg.gamma, cfg.gae_lambda)
        advantages = normalise_advantages(advantages)

        # Damp the policy gradient while the critic is still mostly guessing.
        #
        # Normalising advantages to unit variance makes the policy step the
        # same size whether the advantage is a real signal or pure critic
        # error. Early in a run the critic is always bad, so an undamped update
        # drives the policy with noise at full strength - it wanders, and in
        # the worst case it locks onto whichever action the noise happened to
        # favour. Scaling by how much of the return the critic actually
        # accounts for means the policy moves gently until the value function
        # has earned the right to steer it.
        confidence = 1.0
        if values.size > 1 and float(returns.var()) > 1e-9:
            explained = 1.0 - float(values.var() / returns.var())
            confidence = float(np.clip(explained, 0.02, 1.0))
            advantages = advantages * confidence

        tensors = buffer.tensors()
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # How many minibatches can this machine afford? Estimate from the last
        # measured cost, with a margin, so a slow update shrinks instead of
        # repeating.
        planned = cfg.update_minibatches() * cfg.epochs_per_update
        if self.seconds_per_minibatch > 1e-6 and cfg.update_seconds_budget > 0:
            affordable = int(cfg.update_seconds_budget
                             / (self.seconds_per_minibatch * 1.3))
            planned = max(cfg.epochs_per_update, min(planned, affordable))

        started = time.perf_counter()
        rng = np.random.default_rng(cfg.seed + self.updates)
        stats: List[Dict[str, float]] = []
        self.policy.train()
        ran = 0

        for epoch in range(int(cfg.epochs_per_update)):
            for _starts, grid in buffer.sequence_batches(
                    cfg.seq_len, cfg.minibatch_size, rng):
                mb_started = time.perf_counter()
                stat = self._minibatch(tensors, grid, adv, ret)
                stats.append(stat)
                ran += 1
                self.minibatches_run += 1

                elapsed = time.perf_counter() - mb_started
                self.seconds_per_minibatch = (
                    0.9 * self.seconds_per_minibatch + 0.1 * elapsed)

                if cfg.update_seconds_budget > 0 and \
                        time.perf_counter() - started >= cfg.update_seconds_budget:
                    self.budget_hits += 1
                    self.skipped_minibatches += max(0, planned - ran)
                    break
            else:
                continue
            break

        self.policy.eval()
        self.last_update_seconds = time.perf_counter() - started
        self.updates += 1

        metrics = self._aggregate(stats, values, returns)
        metrics["seconds"] = self.last_update_seconds
        metrics["minibatches"] = float(ran)
        metrics["critic_confidence"] = confidence
        metrics["budget_hit"] = 1.0 if (cfg.update_seconds_budget > 0 and
                                       self.last_update_seconds
                                       >= cfg.update_seconds_budget) else 0.0
        self.last_metrics = metrics
        return metrics

    def _minibatch(self, tensors: Dict[str, torch.Tensor],
                   grid: torch.Tensor, adv: torch.Tensor, ret: torch.Tensor
                   ) -> Dict[str, float]:
        cfg = self.cfg
        embed = tensors["embed"][grid]        # (T, B, E)
        hidden = tensors["hidden"][grid]      # (T, B, H)  state *before* step t
        held = tensors["held"][grid]
        turn = tensors["turn"][grid]
        tap = tensors["tap"][grid]
        old_logp = tensors["logp"][grid]
        mb_adv = adv[grid]
        mb_ret = ret[grid]

        # Score the recorded actions with the recurrent state that was in force
        # *before* each step - the same state the action was sampled with. The
        # frame embeddings were computed once during collection and are reused
        # here, so the encoder does not run again.
        #
        # This used to advance the GRU one more step and score with the result,
        # which is the state *after* the action: the collection path used
        # `hidden` and the update path used `gru(embed, hidden)`, so `old_logp`
        # and `log_prob` disagreed by construction even before the optimiser
        # ran. That alone was worth a large part of the ratio error.
        hidden_states = hidden

        log_prob, entropy, value = self.policy.sequence_log_prob(
            hidden_states, held, turn, tap)

        ratio = torch.exp(log_prob - old_logp)
        unclipped = ratio * mb_adv
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps,
                              1.0 + cfg.clip_eps) * mb_adv
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(value, mb_ret)
        entropy_bonus = entropy.mean()

        loss = (policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy_bonus)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                   cfg.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = float((old_logp - log_prob).mean().item())
            clip_fraction = float(
                ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item())
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_bonus.item()),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "grad_norm": float(grad_norm),
        }

    @staticmethod
    def _aggregate(stats: List[Dict[str, float]], values: np.ndarray,
                   returns: np.ndarray) -> Dict[str, float]:
        if not stats:
            return {}
        keys = stats[0].keys()
        out = {k: float(np.mean([s[k] for s in stats])) for k in keys}
        # Explained variance: how much of the return the critic actually
        # accounts for. Near zero or negative means the value head is noise,
        # which is the state most runs die in without anyone noticing.
        if values.size > 1 and float(returns.var()) > 1e-9:
            out["explained_variance"] = float(
                1.0 - values.var() / returns.var())
        else:
            out["explained_variance"] = 0.0
        return out

    # =====================================================================
    # Diagnostics
    # =====================================================================
    def nonfinite_parameters(self) -> List[str]:
        bad = []
        for name, param in self.policy.named_parameters():
            if not torch.isfinite(param).all():
                bad.append(name)
        return bad

    def status_line(self) -> str:
        m = self.last_metrics
        if not m:
            return "no update yet"
        return (f"loss {m.get('loss', 0.0):+.3f} "
                f"(pi {m.get('policy_loss', 0.0):+.3f}, "
                f"v {m.get('value_loss', 0.0):.3f}) "
                f"ent {m.get('entropy', 0.0):.2f} "
                f"kl {m.get('approx_kl', 0.0):+.4f} "
                f"clip {100.0 * m.get('clip_fraction', 0.0):.0f}% "
                f"ev {m.get('explained_variance', 0.0):+.2f} "
                f"| {m.get('seconds', 0.0):.2f}s/"
                f"{int(m.get('minibatches', 0))}mb")
