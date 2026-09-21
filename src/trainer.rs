//! The PPO update, and the guards that keep it from eating the run. Ported from
//! `botcore/trainer.py`.
//!
//! Why the defaults look different from a typical PPO script: on a CPU the
//! update competes with the game for the machine, and a long update is directly
//! harmful. While an update runs no frames are being collected, so the bot acts
//! on a stale view of the game when it comes back - and if updates grow, the run
//! degrades in a way that looks exactly like "it stalls after a while".
//!
//! Three mechanisms keep that bounded:
//!
//! * minibatches are whole short sequences, so the backward pass is small and
//!   cache-friendly and attention is truncated to the same window the rollout
//!   used;
//! * every update is timed, and the *next* update decides how many minibatches
//!   it can afford from the measured cost;
//! * the status line separates "time in the game" from "time in the update".

use std::collections::HashMap;
use std::time::Instant;

use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, TensorData};

use crate::config::Config;
use crate::model::{ActorCritic, Transition};
use crate::replay::{RolloutBuffer, compute_gae, mean, normalise_advantages, variance};

/// Where one minibatch's tensors come from.
struct MinibatchTensors<B: Backend> {
    hidden: Tensor<B, 3>,
    summary: Tensor<B, 2>,
    held: Tensor<B, 2>,
    turn: Tensor<B, 1, Int>,
    speed: Tensor<B, 1, Int>,
    tap: Tensor<B, 1, Int>,
    logp: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
    returns: Tensor<B, 1>,
}

pub struct Trainer<B: AutodiffBackend> {
    pub cfg: Config,
    optimizer: OptimizerAdaptor<Adam, Net<B>, B>,
    transition_optimizer: OptimizerAdaptor<Adam, Transition<B>, B>,
    /// Smoothed cost model, used to size the next update.
    pub seconds_per_minibatch: f64,
    pub last_update_seconds: f64,
    pub last_metrics: HashMap<String, f32>,
    pub updates: u64,
    pub minibatches_run: u64,
    pub budget_hits: u64,
    pub skipped_minibatches: u64,
    reward_scale: f32,
    reward_scale_ready: bool,
    device: B::Device,
}

// The optimiser is typed over the module it steps, so the alias keeps the field
// declarations readable.
use crate::model::Net;

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(cfg: &Config, device: &B::Device) -> Self {
        // `torch.optim.Adam(lr=3e-4, eps=1e-5)` with the default betas, plus the
        // `clip_grad_norm_` the Python does by hand before stepping.
        let adam = AdamConfig::new()
            .with_epsilon(cfg.adam_eps)
            .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(
                cfg.grad_clip,
            )));
        let transition = AdamConfig::new()
            .with_epsilon(cfg.adam_eps)
            .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)));
        Self {
            cfg: cfg.clone(),
            optimizer: adam.init::<B, Net<B>>(),
            transition_optimizer: transition.init::<B, Transition<B>>(),
            seconds_per_minibatch: 0.02,
            last_update_seconds: 0.0,
            last_metrics: HashMap::new(),
            updates: 0,
            minibatches_run: 0,
            budget_hits: 0,
            skipped_minibatches: 0,
            reward_scale: cfg.reward_scale,
            reward_scale_ready: false,
            device: device.clone(),
        }
    }

    /// Rescale rewards so the *return* is of order one.
    ///
    /// Dividing by the mean reward magnitude is not enough: with a discount
    /// factor near one, a per-step reward of 0.5 becomes a return in the tens,
    /// and the critic is then asked to regress a target far outside the range
    /// its output naturally occupies.
    fn normalise_rewards(&mut self, rewards: &[f32]) -> Vec<f32> {
        let magnitude = mean(&rewards.iter().map(|r| r.abs()).collect::<Vec<f32>>());
        if magnitude > 0.0 {
            if !self.reward_scale_ready {
                self.reward_scale = magnitude;
                self.reward_scale_ready = true;
            } else {
                self.reward_scale = 0.95 * self.reward_scale + 0.05 * magnitude;
            }
        }
        // 1/(1-gamma) is the factor by which a steady per-step reward
        // accumulates into a return.
        let horizon = 1.0 / (1.0 - self.cfg.gamma).max(1e-3);
        let scale = (self.reward_scale * horizon).max(self.cfg.reward_scale);
        rewards.iter().map(|reward| reward / scale).collect()
    }

    /// One PPO update over the collected rollout.
    pub fn update(
        &mut self,
        policy: &mut ActorCritic<B>,
        buffer: &RolloutBuffer,
        last_value: f32,
    ) -> HashMap<String, f32> {
        let cfg = self.cfg.clone();
        if buffer.len() < 2 {
            return HashMap::new();
        }

        let rewards = self.normalise_rewards(buffer.rewards());
        let values = buffer.values();
        let dones = buffer.dones();
        let (mut advantages, returns) =
            compute_gae(&rewards, values, dones, last_value, cfg.gamma, cfg.gae_lambda);
        normalise_advantages(&mut advantages);

        // Damp the policy gradient while the critic is still mostly guessing.
        //
        // Normalising advantages to unit variance makes the policy step the same
        // size whether the advantage is a real signal or pure critic error. Early
        // in a run the critic is always bad, so an undamped update drives the
        // policy with noise at full strength.
        let mut confidence = 1.0f32;
        let return_variance = variance(&returns);
        if values.len() > 1 && return_variance > 1e-9 {
            let explained = 1.0 - variance(values) / return_variance;
            confidence = explained.clamp(0.02, 1.0);
            for advantage in advantages.iter_mut() {
                *advantage *= confidence;
            }
        }

        // How many minibatches can this machine afford? Estimate from the last
        // measured cost, with a margin, so a slow update shrinks instead of
        // repeating.
        let mut planned = cfg.update_minibatches() * cfg.epochs_per_update;
        if self.seconds_per_minibatch > 1e-6 && cfg.update_seconds_budget > 0.0 {
            let affordable =
                (cfg.update_seconds_budget as f64 / (self.seconds_per_minibatch * 1.3)) as usize;
            planned = planned.min(affordable).max(cfg.epochs_per_update);
        }

        let started = Instant::now();
        let mut rng =
            <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(cfg.seed + self.updates);
        let mut stats: Vec<HashMap<String, f32>> = Vec::new();
        let mut ran = 0usize;

        'epochs: for _epoch in 0..cfg.epochs_per_update {
            for indices in buffer.sequence_batches(cfg.seq_len, cfg.minibatch_size, &mut rng) {
                let minibatch_started = Instant::now();
                let stat = self.minibatch(policy, buffer, &indices, &advantages, &returns);
                stats.push(stat);
                ran += 1;
                self.minibatches_run += 1;

                let elapsed = minibatch_started.elapsed().as_secs_f64();
                self.seconds_per_minibatch = 0.9 * self.seconds_per_minibatch + 0.1 * elapsed;

                if cfg.update_seconds_budget > 0.0
                    && started.elapsed().as_secs_f64() >= cfg.update_seconds_budget as f64
                {
                    self.budget_hits += 1;
                    self.skipped_minibatches += planned.saturating_sub(ran) as u64;
                    break 'epochs;
                }
            }
        }

        self.last_update_seconds = started.elapsed().as_secs_f64();
        self.updates += 1;

        let mut metrics = aggregate::<B>(&stats, values, &returns);
        metrics.insert("seconds".into(), self.last_update_seconds as f32);
        metrics.insert("minibatches".into(), ran as f32);
        metrics.insert("critic_confidence".into(), confidence);
        metrics.insert(
            "budget_hit".into(),
            if cfg.update_seconds_budget > 0.0
                && self.last_update_seconds >= cfg.update_seconds_budget as f64
            {
                1.0
            } else {
                0.0
            },
        );
        self.last_metrics = metrics.clone();
        metrics
    }

    fn minibatch(
        &mut self,
        policy: &mut ActorCritic<B>,
        buffer: &RolloutBuffer,
        indices: &[usize],
        advantages: &[f32],
        returns: &[f32],
    ) -> HashMap<String, f32> {
        let cfg = self.cfg.clone();
        let device = self.device.clone();
        let batch = buffer.gather(indices);
        let rows = batch.rows;
        let embed = buffer.embed_dim;
        let mem = buffer.mem_tokens;
        let n_keys = buffer.n_keys;
        let taken: Vec<f32> = indices.iter().map(|index| advantages[*index]).collect();
        let taken_returns: Vec<f32> = indices.iter().map(|index| returns[*index]).collect();

        let tensors = MinibatchTensors {
            hidden: Tensor::from_data(TensorData::new(batch.hidden, [rows, mem, embed]), &device),
            summary: Tensor::from_data(TensorData::new(batch.summary, [rows, embed]), &device),
            held: Tensor::from_data(TensorData::new(batch.held, [rows, n_keys]), &device),
            turn: Tensor::from_data(TensorData::new(batch.turn, [rows]), &device),
            speed: Tensor::from_data(TensorData::new(batch.speed, [rows]), &device),
            tap: Tensor::from_data(TensorData::new(batch.tap, [rows]), &device),
            logp: Tensor::from_data(TensorData::new(batch.logp, [rows]), &device),
            advantages: Tensor::from_data(TensorData::new(taken, [rows]), &device),
            returns: Tensor::from_data(TensorData::new(taken_returns, [rows]), &device),
        };
        let old_logp_host = tensors
            .logp
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();

        // Score the recorded actions with the memory window that was in force
        // *before* each step - the same window the action was sampled with. The
        // frame each step saw comes from the stored summary, so the transformer
        // reads what it read during collection and the encoder does not run
        // again.
        let scored = policy.sequence_log_prob(
            tensors.hidden,
            tensors.summary,
            tensors.held,
            tensors.turn,
            tensors.speed,
            tensors.tap,
        );

        let log_prob = scored.log_prob.clone();
        let ratio = (log_prob.clone() - tensors.logp).exp();
        let unclipped = ratio.clone() * tensors.advantages.clone();
        let clipped = ratio
            .clone()
            .clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            * tensors.advantages.clone();
        let policy_loss = unclipped.min_pair(clipped).mean().neg();
        let value_error = scored.value.clone() - tensors.returns;
        let value_loss = (value_error.clone() * value_error).mean();
        let entropy_bonus = scored.entropy.clone().mean();

        let loss = policy_loss.clone() + value_loss.clone() * cfg.value_coef
            - entropy_bonus.clone() * cfg.entropy_coef;

        let gradients = loss.backward();
        let parameters = GradientsParams::from_grads(gradients, &policy.net);
        // Cloning a module is a few dozen `Arc` increments: Burn's derive keeps
        // parameters behind shared handles, so the optimiser can take the module
        // by value and hand it back.
        let net = policy.net.clone();
        policy.net = self
            .optimizer
            .step(cfg.adam_lr as f64, net, parameters);

        // Diagnostics come off the host, which keeps them out of the graph.
        let scored_host = log_prob
            .detach()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();
        let rows_f = rows.max(1) as f32;
        let approx_kl = old_logp_host
            .iter()
            .zip(&scored_host)
            .map(|(old, new)| old - new)
            .sum::<f32>()
            / rows_f;
        let clip_fraction = old_logp_host
            .iter()
            .zip(&scored_host)
            .filter(|(old, new)| (((*new - *old).exp()) - 1.0).abs() > cfg.clip_eps)
            .count() as f32
            / rows_f;

        HashMap::from([
            ("loss".to_string(), scalar(&loss)),
            ("policy_loss".to_string(), scalar(&policy_loss)),
            ("value_loss".to_string(), scalar(&value_loss)),
            ("entropy".to_string(), scalar(&entropy_bonus)),
            ("approx_kl".to_string(), approx_kl),
            ("clip_fraction".to_string(), clip_fraction),
            // The gradient norm is not reported: Burn's optimiser clips
            // internally and does not hand the norm back, and the Python only
            // prints it.
            ("grad_norm".to_string(), 0.0),
        ])
    }

    /// Teach the transformer's next-frame prediction on the rollout just
    /// collected.
    ///
    /// This is the only training the curiosity signal needs: there is no
    /// separate RND network and no separate forward model, because the model
    /// already predicts the next frame's representation as one of its own heads.
    ///
    /// Pairs that span an inferred reset are dropped: across a respawn there is
    /// no "next frame" that follows from the last one.
    pub fn train_transition(
        &mut self,
        policy: &mut ActorCritic<B>,
        buffer: &RolloutBuffer,
        minibatch: usize,
        steps: usize,
        budget_seconds: f64,
    ) -> HashMap<String, f32> {
        let cfg = self.cfg.clone();
        let device = self.device.clone();
        if steps == 0 || buffer.len() < 2 {
            return HashMap::new();
        }
        let embed = buffer.embed_dim;
        let contexts = buffer.contexts();
        let summaries = buffer.summaries();
        let resets = buffer.resets();

        // Step t predicts the summary of step t+1.
        let mut predictors: Vec<f32> = Vec::new();
        let mut targets: Vec<f32> = Vec::new();
        for t in 0..buffer.len() - 1 {
            if resets[t + 1] >= 0.5 {
                continue;
            }
            predictors.extend_from_slice(&contexts[t * embed..(t + 1) * embed]);
            targets.extend_from_slice(&summaries[(t + 1) * embed..(t + 2) * embed]);
        }
        let n = predictors.len() / embed;
        if n == 0 {
            return HashMap::new();
        }
        let batch = minibatch.min(n).max(1);
        // A fresh generator per call, with the Python's own seed: its shuffles
        // are then identical on every update, which is what the Python does.
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0);
        let mut order: Vec<usize> = (0..n).collect();
        let started = Instant::now();
        let mut out = HashMap::new();
        let mut batches = 0u32;

        for _ in 0..steps.max(1) {
            rand::seq::SliceRandom::shuffle(order.as_mut_slice(), &mut rng);
            let mut start = 0;
            while start < n {
                let take = batch.min(n - start);
                let mut source = Vec::with_capacity(take * embed);
                let mut target = Vec::with_capacity(take * embed);
                for index in &order[start..start + take] {
                    source.extend_from_slice(&predictors[index * embed..(index + 1) * embed]);
                    target.extend_from_slice(&targets[index * embed..(index + 1) * embed]);
                }
                let input =
                    Tensor::<B, 2>::from_data(TensorData::new(source, [take, embed]), &device);
                let expected =
                    Tensor::<B, 2>::from_data(TensorData::new(target, [take, embed]), &device);
                let prediction = policy.predict_next(input);
                let difference = prediction - expected;
                let loss = (difference.clone() * difference).mean();

                let gradients = loss.backward();
                let parameters = GradientsParams::from_grads(gradients, &policy.net.transition);
                let head = policy.net.transition.clone();
                policy.net.transition =
                    self.transition_optimizer
                        .step(cfg.transition_lr as f64, head, parameters);

                out.insert("transition_loss".to_string(), scalar(&loss));
                batches += 1;
                start += take;
                if budget_seconds > 0.0 && started.elapsed().as_secs_f64() >= budget_seconds {
                    out.insert("transition_batches".to_string(), batches as f32);
                    return out;
                }
            }
        }
        out.insert("transition_batches".to_string(), batches as f32);
        out
    }

    /// Parameters that have gone non-finite, which make every later checkpoint
    /// worthless.
    pub fn nonfinite_parameters(policy: &ActorCritic<B>) -> Vec<String> {
        Self::nonfinite_parameters_from_net(&policy.net)
    }

    /// The same check against a bare parameter tree.
    ///
    /// The learning thread hands back weights rather than a policy, because the
    /// optimiser's state belongs to that thread; this is what lets the session
    /// vet them before they become the ones it plays with.
    pub fn nonfinite_parameters_from_net(net: &Net<B>) -> Vec<String> {
        let mut bad: Vec<String> = Vec::new();
        let check = |name: &str, tensor: Tensor<B, 2>, bad: &mut Vec<String>| {
            let values = tensor.into_data().to_vec::<f32>().unwrap_or_default();
            if values.iter().any(|value| !value.is_finite()) {
                bad.push(name.to_string());
            }
        };
        check("patch_embed.proj_w", net.patch_embed.proj_w.val(), &mut bad);
        check("patch_embed.out", net.patch_embed.out.weight.val(), &mut bad);
        for (index, block) in net.blocks.iter().enumerate() {
            check(&format!("blocks.{index}.qkv"), block.qkv.weight.val(), &mut bad);
            check(
                &format!("blocks.{index}.ffn.gate"),
                block.ffn.gate.weight.val(),
                &mut bad,
            );
            check(
                &format!("blocks.{index}.ffn.down"),
                block.ffn.down.weight.val(),
                &mut bad,
            );
        }
        check("hold_head", net.hold_head.weight.val(), &mut bad);
        check("value_head", net.value_head.weight.val(), &mut bad);
        bad
    }

    pub fn status_line(&self) -> String {
        let m = &self.last_metrics;
        if m.is_empty() {
            return "no update yet".to_string();
        }
        let get = |name: &str| m.get(name).copied().unwrap_or(0.0);
        format!(
            "loss {:+.3} (pi {:+.3}, v {:.3}) ent {:.2} kl {:+.4} clip {:.0}% ev {:+.2} \
             fwd {:.4} | {:.2}s/{}mb",
            get("loss"),
            get("policy_loss"),
            get("value_loss"),
            get("entropy"),
            get("approx_kl"),
            get("clip_fraction") * 100.0,
            get("explained_variance"),
            get("transition_loss"),
            get("seconds"),
            get("minibatches") as i64
        )
    }
}

/// A scalar tensor's value, for the metrics map.
fn scalar<B: Backend>(tensor: &Tensor<B, 1>) -> f32 {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .ok()
        .and_then(|values| values.first().copied())
        .unwrap_or(0.0)
}

fn aggregate<B: Backend>(
    stats: &[HashMap<String, f32>],
    values: &[f32],
    returns: &[f32],
) -> HashMap<String, f32> {
    let mut out = HashMap::new();
    if stats.is_empty() {
        return out;
    }
    for key in stats[0].keys() {
        let total: f32 = stats
            .iter()
            .map(|stat| stat.get(key).copied().unwrap_or(0.0))
            .sum();
        out.insert(key.clone(), total / stats.len() as f32);
    }
    // Explained variance: how much of the return the critic actually accounts
    // for. Near zero or negative means the value head is noise, which is the
    // state most runs die in without anyone noticing.
    let return_variance = variance(returns);
    out.insert(
        "explained_variance".into(),
        if values.len() > 1 && return_variance > 1e-9 {
            1.0 - variance(values) / return_variance
        } else {
            0.0
        },
    );
    out
}
