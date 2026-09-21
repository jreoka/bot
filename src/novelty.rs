//! The learning signal, built entirely from pixels and buttons. Ported from
//! `botcore/novelty.py`.
//!
//! There is no game knowledge anywhere in here: everything is computed from what
//! the bot can see and what it pressed, which is what makes the same reward work
//! on any game.
//!
//! What it pays for, and why each term exists:
//!
//! * **Episodic novelty** - reaching a state not reached since the last reset.
//!   This is the term that means *progress*. It is ramped in over the first few
//!   steps of an episode, because an inferred reset clears the memory: without
//!   that ramp a policy which dies every other step is paid for "new" states
//!   every other step, which is a reward for dying.
//! * **Prediction novelty** - the transformer's own next-frame prediction error,
//!   in units of how much that error usually varies. It is capped, because
//!   uncapped prediction error is largest where the screen is most chaotic.
//! * **Anti-degeneracy** - a charge for pressing nothing that grows the longer
//!   nothing is pressed, plus a charge for repeating one screen.
//!
//! A **learning-progress** term - paying for the prediction error *falling* -
//! was tried and removed: on the synthetic corridor, standing still next to one
//! static screen produced the largest sustained drop in error and earned 61% of
//! what walking forward earned. Curiosity that only pays while it is being
//! reduced rewards finding somewhere quiet to sit.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::config::Config;
use crate::vision::signature_key;

/// What a frame reduces to for novelty and reset detection.
pub type Signature = Vec<f32>;

/// Streaming mean and spread of a signal whose magnitude is unknown up front.
///
/// The obvious implementation - Welford's algorithm - has a nasty property here:
/// with only a handful of samples its variance estimate is tiny, so the first
/// few values of a novelty signal get divided by a near-zero spread and come out
/// enormous. Since those first values are exactly what the policy learns from,
/// the bot starts by chasing a loud artefact.
///
/// Exponential moving statistics with a floor on the spread, and a spread that
/// can only *rise* for the first few samples, is what makes a novelty bonus
/// start near 1.0 and stay in that range.
#[derive(Debug, Clone)]
pub struct RunningScale {
    pub mean: f32,
    pub var: f32,
    pub decay: f32,
    pub count: u64,
}

impl Default for RunningScale {
    fn default() -> Self {
        Self {
            mean: 1.0,
            var: 1.0,
            decay: 0.99,
            count: 0,
        }
    }
}

impl RunningScale {
    pub fn new(initial: f32, decay: f32) -> Self {
        Self {
            mean: initial,
            var: initial * initial,
            decay,
            count: 0,
        }
    }

    pub fn update(&mut self, x: f32) {
        self.count += 1;
        self.mean += (1.0 - self.decay) * (x - self.mean);
        if self.count < 8 {
            // A rising spread for the first samples, then a symmetric estimate.
            self.var = self.var.max((x - self.mean) * (x - self.mean));
        } else {
            self.var += (1.0 - self.decay) * ((x - self.mean) * (x - self.mean) - self.var);
        }
    }

    /// Spread, never allowed to collapse to zero.
    pub fn std(&self) -> f32 {
        self.var.sqrt().max(0.25 * self.mean.abs()).max(1e-2)
    }

    pub fn state(&self) -> (f32, f32, u64) {
        (self.mean, self.var, self.count)
    }

    pub fn load(&mut self, state: (f32, f32, u64)) {
        self.mean = state.0;
        self.var = state.1;
        self.count = state.2;
    }
}

/// Infers "the game just started over" from pixels alone.
///
/// No game exposes this in a way a generic bot can read, so it is inferred from
/// three independent observations, because no single one is reliable in a game
/// nobody has described:
///
/// * **surprise** - the model's prediction of the next frame was far worse than
///   it usually is, and something big changed;
/// * **jump** - a large change the bot did not cause, judged against how much
///   this game normally moves when nothing is pressed;
/// * **revisit** - the bot is back where this episode started and has found
///   nothing new, which is what a respawn looks like.
///
/// Getting this right is not cosmetic. If a death is not recognised as a reset,
/// the novelty memory is never cleared, and dying becomes just another source of
/// new states - so the bot learns to die.
#[derive(Debug, Clone)]
pub struct ResetDetector {
    pub ambient: f32,
    pub ambient_samples: u64,
    pub anchor: Option<Signature>,
    pub last_reset_step: i64,
    pub resets: u64,
    pub last_reason: String,
    /// Typical next-frame prediction error, so the jump test can ask "was this
    /// change surprising?" without reaching into the model.
    pub forward_model_reference: f32,
}

impl ResetDetector {
    pub fn new() -> Self {
        Self {
            ambient: 0.0,
            ambient_samples: 0,
            anchor: None,
            last_reset_step: -1_000_000_000,
            resets: 0,
            last_reason: String::new(),
            forward_model_reference: 1.0,
        }
    }

    /// Record what the world looks like at the start of an episode.
    pub fn note_anchor(&mut self, signature: &[f32]) {
        self.anchor = Some(signature.to_vec());
    }

    /// Learn how much this game moves on its own.
    ///
    /// Only steps where the bot pressed nothing feed this estimate, because
    /// those are the only steps where the change is known not to be caused by
    /// the bot.
    pub fn update_ambient(&mut self, delta: f32, engaged: bool) {
        if engaged {
            return;
        }
        self.ambient_samples += 1;
        self.ambient += 0.02 * (delta - self.ambient);
    }

    pub fn check(
        &mut self,
        cfg: &Config,
        step: u64,
        signature: &[f32],
        delta: f32,
        engaged: bool,
        novel_fraction: f32,
        surprise: f32,
    ) -> (bool, String) {
        if (step as i64) - self.last_reset_step < cfg.reset_cooldown as i64 {
            return (false, String::new());
        }
        let reference = self.forward_model_reference.max(1e-6);

        // A big, unexplained change in the world.
        let startled = delta > cfg.reset_jump && surprise > 2.5 * reference;
        if startled {
            self.fire(step);
            return (
                true,
                if engaged { "surprise" } else { "jump" }.to_string(),
            );
        }

        // A large change the bot did not cause, versus how much this game
        // normally moves on its own.
        if !engaged && delta > cfg.reset_jump && delta > 3.0 * self.ambient.max(1e-3) {
            self.fire(step);
            return (true, "jump".to_string());
        }

        // Back to where this episode started, and nowhere new since.
        if let Some(anchor) = &self.anchor
            && novel_fraction < 0.02
        {
            let distance: f32 = signature
                .iter()
                .zip(anchor)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / signature.len().max(1) as f32;
            if distance < cfg.reset_revisit {
                self.fire(step);
                return (true, "revisit".to_string());
            }
        }
        (false, String::new())
    }

    fn fire(&mut self, step: u64) {
        self.last_reset_step = step as i64;
        self.resets += 1;
    }
}

impl Default for ResetDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the reward for one transition, plus the component breakdown the
/// status line reports.
///
/// It owns no network. The prediction the curiosity terms are built from comes
/// from the policy's own next-frame head, produced by the same forward pass the
/// decision used.
pub struct IntrinsicReward {
    pub cfg: Config,
    /// How many new states per step counts as a full-strength novelty term.
    pub novelty_horizon: f32,
    /// How long the bot has to survive before the per-state novelty bonus is
    /// paid in full.
    pub survival_steps: f32,
    pub action_vector_size: usize,
    /// Streaming estimate of "ordinary" next-frame prediction error.
    pub error_running: RunningScale,
    pub detector: ResetDetector,

    episodic: HashMap<u64, u32>,
    episodic_order: VecDeque<u64>,
    global_visits: HashSet<u64>,
    global_order: VecDeque<u64>,

    pub idle_streak: u64,
    recent: VecDeque<u64>,
    recent_limit: usize,

    prev_signature: Option<Signature>,
    pub prev_summary: Option<Vec<f32>>,
    pub prev_prediction: Option<Vec<f32>>,
    pub episode_steps: u64,
    pub episode_novel: u64,
    pub depth_ema: f32,
    prev_depth: Option<f32>,
    pub depth_rate: f32,

    pub last: HashMap<String, f32>,
    pub totals: HashMap<String, f32>,
    pub steps: u64,
}

impl IntrinsicReward {
    pub fn new(cfg: &Config, action_vector_size: usize) -> Self {
        Self {
            novelty_horizon: (cfg.novelty_bins * 3.0).max(4.0),
            survival_steps: cfg.novelty_survival_steps.max(1.0),
            action_vector_size,
            error_running: RunningScale::default(),
            detector: ResetDetector::new(),
            cfg: cfg.clone(),
            episodic: HashMap::new(),
            episodic_order: VecDeque::new(),
            global_visits: HashSet::new(),
            global_order: VecDeque::new(),
            idle_streak: 0,
            recent: VecDeque::new(),
            recent_limit: 40,
            prev_signature: None,
            prev_summary: None,
            prev_prediction: None,
            episode_steps: 0,
            episode_novel: 0,
            depth_ema: 0.0,
            prev_depth: None,
            depth_rate: 0.25,
            last: HashMap::new(),
            totals: HashMap::new(),
            steps: 0,
        }
    }

    /// Start a new episode: forget what has been visited since the last one, and
    /// re-anchor the reset detector on the new starting view.
    ///
    /// The global visit counts deliberately survive. They describe the whole
    /// run, and re-paying for the same room every episode would teach the bot to
    /// go in circles.
    pub fn reset_episode(&mut self, signature: Option<&[f32]>) {
        self.episodic.clear();
        self.episodic_order.clear();
        self.episode_steps = 0;
        self.episode_novel = 0;
        self.depth_ema = 0.0;
        self.prev_depth = None;
        self.idle_streak = 0;
        self.recent.clear();
        self.prev_summary = None;
        self.prev_prediction = None;
        if let Some(signature) = signature {
            self.detector.note_anchor(signature);
        }
    }

    /// A step the bot did not take (paused). Do not charge it to the bot.
    pub fn note_human(&mut self, signature: Option<&[f32]>) {
        self.idle_streak = 0;
        self.prev_signature = signature.map(|s| s.to_vec());
    }

    /// Reward one transition. Returns `(reward, is_reset, reason)`.
    ///
    /// `summary` is this frame's representation and `prediction` is what the
    /// transformer expected it to be, both produced by the single forward pass
    /// the policy already ran. `has_prediction` is false when no predecessor
    /// made one - the first decision of a run, or the first after the window was
    /// reset - because a prediction of zero is not a prediction of nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        summary: Option<&[f32]>,
        signature: &[f32],
        _action_vector: &[f32],
        engaged: bool,
        step: u64,
        prediction: Option<&[f32]>,
        has_prediction: bool,
    ) -> (f32, bool, String) {
        let cfg = self.cfg.clone();
        self.steps += 1;
        self.episode_steps += 1;

        let mut delta = 0.0f32;
        if let Some(previous) = &self.prev_signature {
            delta = signature
                .iter()
                .zip(previous)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / signature.len().max(1) as f32;
        }
        self.detector.update_ambient(delta, engaged);

        let key = signature_key(signature);
        if self.recent.len() >= self.recent_limit {
            self.recent.pop_front();
        }
        self.recent.push_back(key);

        // ---- episodic novelty: the term that means progress ----
        let visits = self.episodic.get(&key).copied().unwrap_or(0);
        if visits == 0 {
            self.episodic_order.push_back(key);
        }
        self.episodic.insert(key, visits + 1);
        if visits == 0 {
            self.episode_novel += 1;
        }

        // Depth is capped at one "horizon" worth of new states, and smoothed
        // quickly so a step that reaches new ground is visibly different from
        // one that does not.
        let capped = (self.episode_novel as f32).min(self.novelty_horizon);
        self.depth_ema += self.depth_rate * (capped - self.depth_ema);
        let depth = self.depth_ema / self.novelty_horizon;
        let depth_reward = cfg.w_episodic * cfg.w_depth * depth;

        // The per-state bonus is gated on having survived a little while: this
        // is the one piece of the reward a bot can farm by dying.
        let survival = (self.episode_steps as f32 / self.survival_steps).min(1.0);
        let novelty_reward =
            cfg.w_episodic * survival * cfg.novelty_decay.powi(visits.min(64) as i32);

        let mut progress_reward = 0.0f32;
        if let Some(previous) = self.prev_depth {
            let delta_depth = (self.depth_ema - previous) / self.novelty_horizon;
            progress_reward = cfg.w_episodic * cfg.w_depth_progress * delta_depth;
        }
        self.prev_depth = Some(self.depth_ema);
        let episodic = depth_reward + novelty_reward + progress_reward;
        let novel_fraction = if self.episode_steps > 0 {
            self.episode_novel as f32 / self.episode_steps as f32
        } else {
            0.0
        };

        // ---- global counts (diagnostics, and a bounded memory) ----
        if self.global_visits.insert(key) {
            self.global_order.push_back(key);
            if self.global_order.len() > cfg.global_capacity {
                self.evict_global();
            }
        }
        if self.episodic.len() > cfg.episodic_capacity {
            // Bounded memory: keep the most recent half. Cheap, and it only
            // happens after a very long single episode.
            let drop_count = self.episodic.len() / 2;
            for _ in 0..drop_count {
                if let Some(old) = self.episodic_order.pop_front() {
                    self.episodic.remove(&old);
                }
            }
        }

        // ---- the model's own prediction of this frame ----
        let mut error = 0.0f32;
        if has_prediction
            && let (Some(predicted), Some(actual)) = (prediction, summary)
            && predicted.len() == actual.len()
        {
            error = predicted
                .iter()
                .zip(actual)
                .map(|(p, a)| (p - a) * (p - a))
                .sum::<f32>()
                / predicted.len().max(1) as f32;
        }
        if !error.is_finite() {
            error = self.error_running.mean;
        }
        // Surprise is measured on the *spread* of the error, not its level, and
        // the estimate is updated before the reward reads it.
        self.error_running.update(error);
        let spread = self.error_running.std().max(1e-6);

        let surprise = error / spread;
        self.detector.forward_model_reference = self.error_running.mean;
        let (is_reset, reason) =
            self.detector
                .check(&cfg, step, signature, delta, engaged, novel_fraction, surprise);

        // ---- anti-degeneracy ----
        if engaged {
            self.idle_streak = 0;
        } else {
            self.idle_streak += 1;
        }
        let ramp = (1.0 + self.idle_streak as f32 / cfg.idle_ramp_steps.max(1.0))
            .min(cfg.idle_ramp_max);
        let idle_cost = if engaged { 0.0 } else { cfg.w_idle * ramp };

        // Repeating one screen for a long stretch is not novelty, it is a loop.
        let repeats = self.recent.iter().filter(|k| **k == key).count();
        let repeat_ratio = repeats as f32 / self.recent.len().max(1) as f32;
        let mut repetition_cost = 0.0f32;
        if repeat_ratio > 0.75 && self.recent.len() >= self.recent_limit {
            repetition_cost = cfg.w_idle * 2.0 * (repeat_ratio - 0.75) / 0.25;
        }

        // ---- prediction novelty ----
        let novelty_term = (cfg.w_novelty * (error / spread)).min(cfg.w_novelty_cap);
        let reward = episodic + novelty_term - idle_cost - repetition_cost;

        self.last = HashMap::from([
            ("episodic".to_string(), episodic),
            ("novelty".to_string(), novelty_term),
            ("error".to_string(), error),
            ("idle".to_string(), -idle_cost),
            ("repeat".to_string(), -repetition_cost),
            ("total".to_string(), reward),
        ]);
        for (name, value) in &self.last {
            *self.totals.entry(name.clone()).or_insert(0.0) += value;
        }

        // ---- advance state ----
        self.prev_summary = summary.map(|s| s.to_vec());
        self.prev_prediction = prediction.map(|p| p.to_vec());
        self.prev_signature = Some(signature.to_vec());

        if is_reset {
            self.reset_episode(Some(signature));
            self.detector.last_reason = reason.clone();
        }

        (reward.clamp(-cfg.reward_clip, cfg.reward_clip), is_reset, reason)
    }

    /// Drop the oldest quarter of the global visit set, in one pass.
    fn evict_global(&mut self) {
        let drop = (self.global_order.len() / 4).max(1);
        for _ in 0..drop {
            match self.global_order.pop_front() {
                Some(key) => {
                    self.global_visits.remove(&key);
                }
                None => break,
            }
        }
    }

    pub fn mean_reward(&self) -> f32 {
        self.totals.get("total").copied().unwrap_or(0.0) / self.steps.max(1) as f32
    }

    pub fn summary(&self) -> String {
        let n = self.steps.max(1) as f32;
        format!(
            "rwd/step {:+.4} (epi {:+.4}, nov {:+.4}, idle {:+.4})",
            self.totals.get("total").copied().unwrap_or(0.0) / n,
            self.totals.get("episodic").copied().unwrap_or(0.0) / n,
            self.totals.get("novelty").copied().unwrap_or(0.0) / n,
            self.totals.get("idle").copied().unwrap_or(0.0) / n
        )
    }

    pub fn episodic_bonus(&self) -> f32 {
        self.last.get("episodic").copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_running_scale_starts_near_one_and_stays_bounded() {
        let mut scale = RunningScale::default();
        let mut values = Vec::new();
        for step in 0..50 {
            let error = 0.5 + 0.25 * ((step % 7) as f32);
            scale.update(error);
            values.push(error / scale.std().max(1e-6));
        }
        // The first sample is measured against the initial spread of 1.0, so it
        // cannot come out enormous - which is the whole point of the class.
        assert!(values[0] < 2.0, "{values:?}");
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn a_return_to_the_anchor_with_no_novelty_is_a_reset() {
        let cfg = Config::default();
        let mut detector = ResetDetector::new();
        let anchor = vec![0.5f32; 1024];
        detector.note_anchor(&anchor);
        let (is_reset, reason) = detector.check(&cfg, 100, &anchor, 0.0, true, 0.0, 0.0);
        assert!(is_reset);
        assert_eq!(reason, "revisit");
    }

    #[test]
    fn the_cooldown_suppresses_a_second_reset() {
        let cfg = Config::default();
        let mut detector = ResetDetector::new();
        let anchor = vec![0.5f32; 1024];
        detector.note_anchor(&anchor);
        assert!(detector.check(&cfg, 100, &anchor, 0.0, true, 0.0, 0.0).0);
        assert!(!detector.check(&cfg, 101, &anchor, 0.0, true, 0.0, 0.0).0);
    }

    #[test]
    fn paying_the_same_state_twice_pays_less() {
        let mut cfg = Config::default();
        cfg.w_novelty = 0.0;
        cfg.w_depth = 0.0;
        cfg.w_depth_progress = 0.0;
        cfg.w_idle = 0.0;
        let mut reward = IntrinsicReward::new(&cfg, 4);
        let signature = vec![0.25f32; 1024];
        let action = vec![1.0, 0.0, 0.0, 0.0];
        reward.reset_episode(Some(&signature));
        let mut totals = Vec::new();
        for step in 0..20 {
            let (value, _, _) = reward.compute(
                None, &signature, &action, true, step, None, false,
            );
            totals.push(value);
        }
        // Every step is the same state, so the per-visit bonus decays with the
        // repeat count rather than paying full price forever.
        assert!(totals[0] > totals[19], "{totals:?}");
    }
}
