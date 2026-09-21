//! A synthetic game whose progress is known exactly. Ported from
//! `botcore/synth.py`.
//!
//! This exists so that "the bot learns to play" is a statement that can be
//! checked rather than hoped for: `--selftest` runs the identical session,
//! model, reward and trainer against this environment and reports whether the
//! bot moved further into it over time.
//!
//! The game is deliberately the smallest thing that still has the property that
//! matters: a long corridor, the bot can only walk forward or back, going
//! backwards kills it, and dying sends it back to the start. So the only way to
//! see more of the game is to walk forward, and the only way to be rewarded for
//! it is to reach cells it has not reached since the last attempt.
//!
//! The bot is given no advantage here beyond what it gets in a real game: it
//! sees a rendered first-person-ish view of the corridor (never its own
//! coordinates), it presses the same keys, and its reward comes from the same
//! reward module.

use std::time::Instant;

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;

use crate::config::Config;
use crate::keys::{ActionSpace, Keymap};
use crate::model::{Action, ActorCritic};
use crate::session::{EnvAction, EnvStep, Environment, GameSession};
use crate::vision::Observation;

/// A corridor the bot must learn to walk down.
pub struct SyntheticEnv {
    cfg: Config,
    length: i32,
    max_episode_steps: u64,
    frames: Vec<Vec<f32>>,
    observation: Observation,
    cell: i32,
    facing: i32,
    steps: u64,
    pub deaths: u64,
    max_cell_seen: i32,
    total_forward_steps: u64,
    dead_for: i32,
}

impl SyntheticEnv {
    pub const FORWARD_BIT: usize = 0;
    pub const BACKWARD_BIT: usize = 1;

    pub fn new(cfg: &Config, length: i32, max_episode_steps: u64) -> Self {
        let size = cfg.frame_size;
        let channels = cfg.channels();
        Self {
            cfg: cfg.clone(),
            length,
            max_episode_steps,
            frames: Vec::new(),
            observation: Observation::zeros(channels, size, size),
            cell: 0,
            facing: 1,
            steps: 0,
            deaths: 0,
            max_cell_seen: 0,
            total_forward_steps: 0,
            dead_for: 0,
        }
    }

    /// Render what the bot sees: a corridor ahead, brighter the closer the far
    /// wall is, plus side walls. There are no coordinates in the image - the bot
    /// has to work out where it is from the view, like in a real game.
    fn gray_view(&self) -> Vec<f32> {
        let size = self.cfg.frame_size;
        let remaining = (self.length - self.cell) as f32;
        let mut frame = vec![0.0f32; size * size];
        let horizon = (size as f32 * 0.45) as usize;
        let band = ((size as f32 * 0.18) as usize).max(2);
        let brightness = 0.25 + 0.6 / (1.0 + remaining / 3.0);
        for row in horizon..(horizon + band).min(size) {
            for column in 0..size {
                frame[row * size + column] = brightness;
            }
        }

        // Side walls, converging toward the centre with distance.
        let centre = size / 2;
        let spread = ((size as f32 * 0.42 * (remaining / 6.0 + 0.15).min(1.0)) as usize).max(2);
        let left_start = centre.saturating_sub(spread + 6);
        let left_end = centre.saturating_sub(spread);
        let right_start = (centre + spread).min(size);
        let right_end = (centre + spread + 6).min(size);
        for row in 0..size {
            for column in left_start..left_end {
                frame[row * size + column] = 0.35;
            }
            for column in right_start..right_end {
                frame[row * size + column] = 0.35;
            }
        }

        // A facing marker: a small bright block offset by the heading, so
        // turning is visible without revealing the position.
        let marker_x = (centre as i32 + self.facing * (size as f32 * 0.18) as i32).max(0) as usize;
        let marker_y = horizon + band + 2;
        for row in marker_y..(marker_y + 4).min(size) {
            for column in marker_x.saturating_sub(2)..(marker_x + 2).min(size) {
                frame[row * size + column] = 1.0;
            }
        }

        // Walking further in genuinely changes the picture even at a distance.
        let factor = 1.0 + 0.03 * self.cell as f32;
        for row in horizon..(horizon + band).min(size) {
            for column in 0..size {
                frame[row * size + column] =
                    (frame[row * size + column] * factor).clamp(0.0, 1.0);
            }
        }
        frame
    }

    /// A distinct screen the reset detector can see as "something changed".
    fn death_view(&self) -> Vec<f32> {
        let size = self.cfg.frame_size;
        let band = (size / 8).max(2);
        let mut frame = vec![0.05f32; size * size];
        let mut row = 0;
        while row < size {
            for column in 0..size {
                frame[row * size + column] = 0.9;
            }
            row += band;
        }
        frame
    }

    fn build_observation(&mut self) -> Observation {
        let gray = if self.cell < 0 {
            self.death_view()
        } else {
            self.gray_view()
        };
        self.frames.push(gray);
        let stack = self.cfg.frame_stack;
        if self.frames.len() > stack {
            self.frames.remove(0);
        }
        while self.frames.len() < stack {
            let first = self.frames[0].clone();
            self.frames.insert(0, first);
        }
        for (index, frame) in self.frames.iter().enumerate() {
            self.observation
                .channel_mut(index)
                .copy_from_slice(frame);
        }
        // The difference channel: |newest - oldest|, which is what lets the
        // model see motion without inferring it from appearance alone.
        let newest = self.frames[stack - 1].clone();
        let oldest = self.frames[0].clone();
        let difference: Vec<f32> = newest
            .iter()
            .zip(&oldest)
            .map(|(a, b)| (a - b).abs())
            .collect();
        self.observation.channel_mut(stack).copy_from_slice(&difference);
        self.observation.clone()    }

    /// Current goal progress: how far into the corridor the bot is.
    pub fn progress(&self) -> i32 {
        self.cell.max(0)
    }

    pub fn max_cell_seen(&self) -> i32 {
        self.max_cell_seen
    }

    pub fn total_forward_steps(&self) -> u64 {
        self.total_forward_steps
    }
}

impl Environment for SyntheticEnv {
    fn observation_shape(&self) -> (usize, usize, usize) {
        self.observation.shape()
    }

    fn gray_channels(&self) -> usize {
        self.cfg.frame_stack
    }

    fn reset(&mut self, _seed: Option<u64>) -> (Observation, Vec<f32>) {
        self.cell = 0;
        self.facing = 1;
        self.steps = 0;
        self.dead_for = 0;
        self.frames.clear();
        let observation = self.build_observation();
        let signature = crate::vision::gray_signature(
            observation.channel(0),
            observation.height,
            observation.width,
            32,
        );
        (observation, signature)
    }

    fn step(&mut self, action: &EnvAction) -> (Observation, EnvStep) {
        let mask = action.held_mask;
        let turn = action.turn_index;

        if turn == 1 {
            self.facing = -1;
        } else if turn == 2 {
            self.facing = 1;
        }

        if self.cell < 0 {
            // On the death screen. Count it down and respawn, which is the one
            // moment a reset detector should be certain about.
            self.dead_for -= 1;
            if self.dead_for <= 0 {
                self.cell = 0;
                self.facing = 1;
                self.frames.clear();
            }
        } else {
            let forward = mask & (1 << Self::FORWARD_BIT) != 0;
            let backward = mask & (1 << Self::BACKWARD_BIT) != 0;
            if forward && !backward {
                self.cell += 1;
                self.total_forward_steps += 1;
            } else if backward && !forward {
                self.cell -= 1;
            }
            if self.cell >= self.length {
                self.cell = self.length - 1;
            }
            if self.cell < 0 {
                self.deaths += 1;
                self.dead_for = 3;
            }
        }

        let observation = self.build_observation();
        self.steps += 1;
        self.max_cell_seen = self.max_cell_seen.max(self.cell);
        let died = self.cell < 0;
        let truncated = !died && self.steps >= self.max_episode_steps;
        (
            observation,
            EnvStep {
                truncated,
                died,
                progress: self.cell,
            },
        )
    }
}

// =============================================================================
// The learning check
// =============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct Distances {
    pub mean_distance: f64,
    pub best_distance: f64,
    pub mean_steps: f64,
}

/// How far into the world these states got, measured from where it started.
fn peak_distance(cells: &[i32]) -> i32 {
    match (cells.first(), cells.iter().max()) {
        (Some(first), Some(best)) => (best - first).max(0),
        _ => 0,
    }
}

/// A decision that follows the policy's own preference without sampling noise.
///
/// The held-key head is a *set* of binary bits, so there is no single argmax:
/// thresholding every bit at 0.5 would have an undecided policy press every key
/// at once, which looks like a broken agent rather than an undecided one.
/// Instead the bits are ranked by their log-odds and the top-k are taken, where
/// k is the count the distribution favours. That is the mode of the policy in
/// the only sense that is well defined for a set-valued action.
fn greedy_action<B: Backend>(
    policy: &ActorCritic<B>,
    observation: &Observation,
    hidden: Tensor<B, 3>,
    device: &B::Device,
) -> (EnvAction, Tensor<B, 3>) {
    let input = Tensor::<B, 4>::from_data(
        TensorData::new(
            observation.data.clone(),
            [1, observation.channels, observation.height, observation.width],
        ),
        device,
    );
    let tokens = policy.encode(input);
    let (window, context, features) = policy.forward(hidden, tokens);
    let (hold_logits, turn_logits, speed_logits, tap_logits, _value) =
        policy.heads(features, context);

    let logits = hold_logits
        .into_data()
        .to_vec::<f32>()
        .expect("logits are f32");
    let probabilities: Vec<f32> = logits
        .iter()
        .map(|value| 1.0 / (1.0 + (-value).exp()))
        .collect();
    let key_count = logits.len();
    let cap = policy.dims.max_held.min(key_count);
    // k = the number of keys the distribution expects to be held.
    let expected: f32 = probabilities.iter().sum();
    let k = (expected.round() as i64).clamp(0, cap as i64) as usize;

    let mut order: Vec<usize> = (0..key_count).collect();
    order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    let mut mask = 0u32;
    for index in order.iter().take(k) {
        mask |= 1 << *index;
    }

    // The synthetic world only understands left/right/no-turn, so the four
    // directional choices are collapsed onto it. The speed choice is not part of
    // the synthetic world at all, which is the honest arrangement: that game has
    // no view to swing, so there is nothing for a speed to mean there.
    let direction = argmax(
        &turn_logits.into_data().to_vec::<f32>().expect("logits"),
    );
    let turn = if direction == 1 {
        1
    } else if direction == 2 {
        2
    } else {
        0
    };
    let speed = argmax(&speed_logits.into_data().to_vec::<f32>().expect("logits"));
    let tap = argmax(&tap_logits.into_data().to_vec::<f32>().expect("logits"));
    (
        EnvAction {
            held_vks: Vec::new(),
            turn: (0, 0),
            tap_vk: 0,
            index: 0,
            held_mask: mask,
            turn_index: turn,
            speed_index: speed,
            tap_index: tap,
            mouse_step: policy.mouse_step_value(),
        },
        window,
    )
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// Run the policy without sampling noise and measure how far it gets.
pub fn evaluate<B: Backend>(
    policy: &ActorCritic<B>,
    env: &mut dyn Environment,
    device: &B::Device,
    episodes: usize,
    max_steps: usize,
) -> Distances {
    let mut distances = Vec::new();
    let mut steps_taken = Vec::new();
    for _episode in 0..episodes {
        let (mut observation, _signature) = env.reset(None);
        let mut hidden = policy.initial_hidden(1, device);
        let mut cells = Vec::new();
        for _step in 0..max_steps {
            let (action, window) = greedy_action(policy, &observation, hidden, device);
            hidden = window;
            let (next, step) = env.step(&action);
            cells.push(step.progress);
            let finished = step.truncated || step.died;
            observation = next;
            if finished {
                break;
            }
        }
        distances.push(peak_distance(&cells) as f64);
        steps_taken.push(cells.len() as f64);
    }
    let mean = |values: &[f64]| {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    };
    Distances {
        mean_distance: mean(&distances),
        best_distance: distances.iter().copied().fold(0.0, f64::max),
        mean_steps: mean(&steps_taken),
    }
}

/// How far an agent that presses keys at random gets.
///
/// This is the control that makes the self-test mean something. In a small maze
/// a random walk scores well by accident, so "the trained policy got further"
/// only says something if it also beat random by a clear margin.
pub fn random_baseline(
    env: &mut dyn Environment,
    max_held: usize,
    episodes: usize,
    max_steps: usize,
    seed: u64,
) -> Distances {
    use rand::Rng;
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
    let mut distances = Vec::new();
    for _episode in 0..episodes {
        env.reset(None);
        let mut cells = Vec::new();
        for _step in 0..max_steps {
            let mut mask = 0u32;
            for index in 0..max_held {
                if rng.random::<f32>() < 0.25 {
                    mask |= 1 << index;
                }
            }
            let turn = rng.random_range(0..3usize);
            let (_, step) = env.step(&EnvAction {
                held_vks: Vec::new(),
                turn: (0, 0),
                tap_vk: 0,
                index: 0,
                held_mask: mask,
                turn_index: turn,
                speed_index: 0,
                tap_index: 0,
                mouse_step: 1,
            });
            cells.push(step.progress);
            if step.truncated || step.died {
                break;
            }
        }
        distances.push(peak_distance(&cells) as f64);
    }
    let mean = if distances.is_empty() {
        0.0
    } else {
        distances.iter().sum::<f64>() / distances.len() as f64
    };
    Distances {
        mean_distance: mean,
        best_distance: distances.iter().copied().fold(0.0, f64::max),
        mean_steps: 0.0,
    }
}

/// A config for the synthetic runs: the same algorithm, no wall-clock pacing.
pub fn synthetic_train_config(frame_size: usize, rollout_steps: usize, seed: u64) -> Config {
    let base = Config::default();
    Config {
        frame_size,
        frame_stack: 4,
        action_repeat: 2,
        rollout_steps,
        minibatch_size: 256,
        epochs_per_update: 4,
        // A shorter memory window than the shipped default: the synthetic games
        // are decided by the last handful of frames, and the self-test is worth
        // more when it finishes in minutes than when it is marginally better.
        mem_tokens: 8,
        // The synthetic games have no view to swing, so there is nothing for a
        // mouse-speed correction to measure; leaving it on would walk the base
        // step to its ceiling for no reason.
        mouse_adapt_rate: 0.0,
        target_fps: 0.0,
        enable_hotkeys: false,
        torch_threads: base.torch_threads,
        seed,
        checkpoint_dir: String::new(),
        log_json: None,
        max_held_keys: 3,
        novelty_bins: 48.0,
        ..base
    }
}

/// A keymap whose hold order puts the two movement keys in the first two bit
/// positions, so the synthetic world can read them. The bot is never told which
/// key means what - it has to discover that pressing one helps.
pub fn synthetic_action_space(cfg: &Config) -> ActionSpace {
    let mut keymap = Keymap::default_layout();
    keymap.holds = vec!["W".to_string(), "S".to_string(), "D".to_string()];
    keymap.taps = Vec::new();
    keymap.mouse_buttons = Vec::new();
    keymap.vks.clear();
    keymap.vks.insert("W".to_string(), 0x57);
    keymap.vks.insert("S".to_string(), 0x53);
    keymap.vks.insert("D".to_string(), 0x44);
    ActionSpace::build(&keymap, cfg.max_held_keys, &cfg.speed_levels, true, false)
}

/// One check's outcome.
#[derive(Debug, Clone)]
pub struct Check {
    pub name: String,
    pub before: f64,
    pub after: f64,
    pub best: f64,
    pub target_distance: f64,
    pub random_baseline: f64,
    pub improvement: f64,
    pub steps: u64,
    pub solved_at: u64,
    pub seconds: f64,
    pub passed: bool,
}

/// Everything the self-test measured.
#[derive(Debug, Clone, Default)]
pub struct SelfTestResult {
    pub corridor: Option<Check>,
    pub forward_reward_per_step: f64,
    pub backwards_reward_per_step: f64,
    pub backwards_ratio: f64,
    pub control_passed: bool,
    pub idle_reward_per_step: f64,
    pub idle_ratio: f64,
    pub idle_passed: bool,
    pub passed: bool,
    pub ok: bool,
}

/// Train the real algorithm on a game whose progress is known, and report
/// whether it got better.
///
/// Nothing here is special-cased for the synthetic environment: it builds the
/// same config, the same session, the same reward module and the same PPO update
/// that run against a real window. That is the point - this is a measurement of
/// the bot, not of a toy.
pub fn run_learning_check<B: AutodiffBackend>(
    device: B::Device,
    verbose: bool,
    seed: u64,
    steps: u64,
    frame_size: usize,
    rollout_steps: usize,
    corridor_length: i32,
) -> anyhow::Result<SelfTestResult> {
    let cfg = synthetic_train_config(frame_size, rollout_steps, seed);
    cfg.validate()?;
    let mut results = SelfTestResult::default();

    if verbose {
        println!();
        println!("{}", "-".repeat(78));
        println!("  SELF-TEST: can the identical algorithm learn a game it has");
        println!("  never seen, with no game-specific code and no score reading?");
        println!("{}", "-".repeat(78));
    }

    let env = SyntheticEnv::new(&cfg, corridor_length, 300);
    let mut session = GameSession::<B>::new(
        &cfg,
        Box::new(env),
        synthetic_action_space(&cfg),
        None,
        device,
    )?;

    // ---- 1. corridor: the cleanest statement of "make progress" ----
    let control = random_baseline(session.env.as_mut(), cfg.max_held_keys, 5, 300, 12345);
    let before = evaluate(
        &session.acting,
        session.env.as_mut(),
        &session.device,
        5,
        300,
    );
    let started = Instant::now();
    session.reset(Some(seed), false);
    let mut solved_at = 0u64;
    let mut updates_done = 0u64;
    while session.total_steps < steps {
        let action = session.decide();
        let (_reward, done, _info) = session.step(&action);
        if done {
            session.reset(None, true);
        }
        if session.buffer.len() >= cfg.rollout_steps {
            session.bootstrapped_value();
            session.learn();
            updates_done += 1;
            if verbose {
                // The Python's self-test says nothing until the end; a line per
                // update is what makes a run that has stopped learning visible
                // while it is happening.
                println!(
                    "  [selftest] step {:>6}  update {:>3}  {:.1}s  {}  resets {}",
                    session.total_steps,
                    updates_done,
                    started.elapsed().as_secs_f64(),
                    session.reward.summary(),
                    session.reward.detector.resets
                );
            }
            // Stop as soon as the bot is demonstrably playing the game, rather
            // than spending the whole budget proving it twice.
            if updates_done % 4 == 0 {
                let probe = evaluate(
                    &session.acting,
                    session.env.as_mut(),
                    &session.device,
                    3,
                    300,
                );
                if probe.mean_distance > 0.5 * corridor_length as f64 {
                    solved_at = session.total_steps;
                    break;
                }
            }
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let after = evaluate(
        &session.acting,
        session.env.as_mut(),
        &session.device,
        5,
        300,
    );

    let improvement = after.mean_distance - before.mean_distance;
    let baseline = control.mean_distance;
    let reached_target = after.mean_distance >= 0.75 * (corridor_length as f64).max(1.0);
    // "Learned" means it ends up playing the game, or it clearly improved on
    // where it started. Grading purely on improvement would fail a lucky
    // initialisation that already plays well; grading purely on the final score
    // would pass a policy that was handed the answer by its initial weights.
    let learned = reached_target || improvement > 0.25 * (corridor_length as f64).max(1.0);
    let mut passed = learned && after.mean_distance >= 5.0;
    // Beating a random walk is the part that cannot happen by accident.
    passed = passed && after.mean_distance > 1.5 * baseline;

    let check = Check {
        name: "corridor (go forward)".to_string(),
        before: before.mean_distance,
        after: after.mean_distance,
        best: after.best_distance,
        target_distance: corridor_length as f64,
        random_baseline: baseline,
        improvement,
        steps: session.total_steps,
        solved_at,
        seconds: elapsed,
        passed,
    };
    if verbose {
        println!(
            "  {:<24} distance {:5.1} -> {:5.1}   random {:5.1}   {} steps in {:5.1}s   {}",
            check.name,
            check.before,
            check.after,
            baseline,
            check.steps,
            elapsed,
            if passed { "LEARNED" } else { "DID NOT LEARN" }
        );
    }
    results.passed = passed;
    results.corridor = Some(check);

    // ---- 2 and 3. the reward must prefer progress over going backwards ----
    //
    // An absolute threshold would be meaningless here, because it would have to
    // be retuned every time a reward weight changes. What must stay true is the
    // *ordering*: a policy that walks forward has to earn clearly more per step
    // than one that walks backwards.
    if verbose {
        println!();
        println!("  Control: the reward must prefer going forward to going");
        println!("  backwards, or it is not a progress signal.");
    }
    let forward_reward = fixed_policy_reward::<B>(
        &cfg,
        corridor_length,
        seed,
        1 << SyntheticEnv::FORWARD_BIT,
        400,
        &session.device,
    )?;
    let control_reward = fixed_policy_reward::<B>(
        &cfg,
        corridor_length,
        seed,
        1 << SyntheticEnv::BACKWARD_BIT,
        400,
        &session.device,
    )?;
    let ratio = control_reward / forward_reward.max(1e-6);
    let control_passed = ratio < 0.35;
    results.forward_reward_per_step = forward_reward;
    results.backwards_reward_per_step = control_reward;
    results.backwards_ratio = ratio;
    results.control_passed = control_passed;
    if verbose {
        println!(
            "  {:<24} reward/step {:+.4}",
            "always forward", forward_reward
        );
        println!(
            "  {:<24} reward/step {:+.4}   {:.0}% of forward   {}",
            "always backwards",
            control_reward,
            100.0 * ratio,
            if control_passed {
                "correctly lower"
            } else {
                "TOO HIGH: the reward can be farmed by dying"
            }
        );
    }

    let idle_reward = fixed_policy_reward::<B>(
        &cfg,
        corridor_length,
        seed,
        0,
        400,
        &session.device,
    )?;
    let idle_ratio = idle_reward / forward_reward.max(1e-6);
    let idle_passed = idle_ratio < 0.35;
    results.idle_reward_per_step = idle_reward;
    results.idle_ratio = idle_ratio;
    results.idle_passed = idle_passed;
    if verbose {
        println!(
            "  {:<24} reward/step {:+.4}   {:.0}% of forward   {}",
            "never press anything",
            idle_reward,
            100.0 * idle_ratio,
            if idle_passed {
                "correctly lower"
            } else {
                "TOO HIGH: standing still pays"
            }
        );
    }

    results.ok = results.passed && control_passed && idle_passed;
    if verbose {
        println!("{}", "-".repeat(78));
        println!("  RESULT: {}", if results.ok { "PASS" } else { "FAIL" });
        println!("{}", "-".repeat(78));
        println!();
    }
    Ok(results)
}

/// The reward a fixed policy earns per step, which is what the two controls
/// measure: one bit at a time, so "forward" really means forward.
fn fixed_policy_reward<B: AutodiffBackend>(
    cfg: &Config,
    corridor_length: i32,
    seed: u64,
    mask: u32,
    trials: usize,
    device: &B::Device,
) -> anyhow::Result<f64> {
    let env = SyntheticEnv::new(cfg, corridor_length, 300);
    let mut session = GameSession::<B>::new(
        cfg,
        Box::new(env),
        synthetic_action_space(cfg),
        None,
        device.clone(),
    )?;
    session.reset(Some(seed), false);
    let mut held = vec![0.0f32; session.action_space.hold_vks.len()];
    if mask != 0 {
        let index = (mask as f32).log2() as usize;
        if index < held.len() {
            held[index] = 1.0;
        }
    }
    let action = Action {
        held,
        direction: 0,
        speed: 0,
        tap: 0,
    };
    let mut total = 0.0f64;
    for _ in 0..trials {
        let (reward, done, _info) = session.step(&action);
        total += reward as f64;
        if done {
            session.reset(None, false);
        }
    }
    Ok(total / trials.max(1) as f64)
}

/// The action dictionary the environment wants, as the session builds it.
pub fn env_action_for_mask(held_mask: u32, turn_index: usize, mouse_step: i32) -> EnvAction {
    EnvAction {
        held_vks: Vec::new(),
        turn: (0, 0),
        tap_vk: 0,
        index: 0,
        held_mask,
        turn_index,
        speed_index: 0,
        tap_index: 0,
        mouse_step,
    }
}
