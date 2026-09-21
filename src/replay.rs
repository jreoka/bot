//! Storage for one rollout, and the sequence minibatch sampler. Ported from
//! `botcore/replay.py`.
//!
//! Two details matter more than they look:
//!
//! **The memory window is stored, not recomputed.** A transformer policy in PPO
//! needs the window that was actually in force when each action was taken. The
//! obvious shortcut - re-run the encoder over the rollout to rebuild the windows
//! - makes the update score old actions under new inputs, which corrupts the
//! importance ratio. Storing one window per step costs a few hundred kilobytes
//! per rollout and removes the problem.
//!
//! **Memory is a fixed allocation.** The vectors are allocated once at the
//! rollout size and overwritten. Nothing here grows over a run, which is one of
//! the reasons the bot does not degrade after a few hours.
//!
//! This is host memory, not device memory: the update builds its tensors from
//! these slices one minibatch at a time. The Python does the same thing through
//! `numpy` -> `torch.as_tensor`.

use rand::RngExt;
use rand::seq::SliceRandom;

pub struct RolloutBuffer {
    pub capacity: usize,
    pub embed_dim: usize,
    pub mem_tokens: usize,
    pub n_keys: usize,
    size: usize,
    episodes: usize,

    context: Vec<f32>,
    summary: Vec<f32>,
    window: Vec<f32>,
    held: Vec<f32>,
    turn: Vec<i64>,
    speed: Vec<i64>,
    tap: Vec<i64>,
    logp: Vec<f32>,
    value: Vec<f32>,
    reward: Vec<f32>,
    done: Vec<f32>,
    reset: Vec<f32>,
    action_vector: Option<Vec<f32>>,
    action_vector_dim: usize,
}

/// One minibatch, gathered out of the rollout and ready to become tensors.
pub struct Batch {
    pub rows: usize,
    pub hidden: Vec<f32>,
    pub summary: Vec<f32>,
    pub held: Vec<f32>,
    pub turn: Vec<i64>,
    pub speed: Vec<i64>,
    pub tap: Vec<i64>,
    pub logp: Vec<f32>,
}

impl RolloutBuffer {
    pub fn new(
        capacity: usize,
        embed_dim: usize,
        mem_tokens: usize,
        n_keys: usize,
    ) -> Self {
        let mem_tokens = mem_tokens.max(1);
        Self {
            capacity,
            embed_dim,
            mem_tokens,
            n_keys,
            size: 0,
            episodes: 0,
            context: vec![0.0; capacity * embed_dim],
            summary: vec![0.0; capacity * embed_dim],
            window: vec![0.0; capacity * mem_tokens * embed_dim],
            held: vec![0.0; capacity * n_keys],
            turn: vec![0; capacity],
            speed: vec![0; capacity],
            tap: vec![0; capacity],
            logp: vec![0.0; capacity],
            value: vec![0.0; capacity],
            reward: vec![0.0; capacity],
            done: vec![0.0; capacity],
            reset: vec![0.0; capacity],
            action_vector: None,
            action_vector_dim: 0,
        }
    }

    pub fn reset(&mut self) {
        self.size = 0;
        self.episodes = 0;
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn is_full(&self) -> bool {
        self.size >= self.capacity
    }

    pub fn episodes(&self) -> usize {
        self.episodes
    }

    /// Add one transition.
    ///
    /// `terminal` means the environment really ended (window closed, or a
    /// synthetic episode hitting its limit). An *inferred* game reset is not a
    /// terminal step: the game continues, so the value bootstrap must too.
    /// Conflating the two teaches the critic that the world is about to vanish.
    ///
    /// `reset` does mark an inferred reset, because it invalidates the
    /// transition the curiosity head would otherwise learn from: across a
    /// respawn there is no "next frame" that follows from the last one.
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        context: &[f32],
        summary: &[f32],
        window: &[f32],
        held: &[f32],
        direction: usize,
        speed: usize,
        tap: usize,
        log_prob: f32,
        value: f32,
        reward: f32,
        terminal: bool,
        reset: bool,
        action_vector: Option<&[f32]>,
    ) {
        assert!(
            self.size < self.capacity,
            "RolloutBuffer overflow: add() past capacity"
        );
        let index = self.size;
        let embed = self.embed_dim;
        let mem = self.mem_tokens;
        self.context[index * embed..(index + 1) * embed].copy_from_slice(context);
        self.summary[index * embed..(index + 1) * embed].copy_from_slice(summary);
        self.window[index * mem * embed..(index + 1) * mem * embed].copy_from_slice(window);
        self.held[index * self.n_keys..(index + 1) * self.n_keys].copy_from_slice(held);
        self.turn[index] = direction as i64;
        self.speed[index] = speed as i64;
        self.tap[index] = tap as i64;
        self.logp[index] = log_prob;
        self.value[index] = value;
        self.reward[index] = reward;
        self.done[index] = if terminal { 1.0 } else { 0.0 };
        self.reset[index] = if reset { 1.0 } else { 0.0 };
        if let Some(vector) = action_vector {
            if self.action_vector.is_none() {
                self.action_vector_dim = vector.len();
                self.action_vector = Some(vec![0.0; self.capacity * self.action_vector_dim]);
            }
            let dim = self.action_vector_dim;
            if let Some(storage) = self.action_vector.as_mut() {
                storage[index * dim..(index + 1) * dim].copy_from_slice(vector);
            }
        }
        self.size += 1;
        if terminal {
            self.episodes += 1;
        }
    }

    pub fn rewards(&self) -> &[f32] {
        &self.reward[..self.size]
    }

    pub fn values(&self) -> &[f32] {
        &self.value[..self.size]
    }

    pub fn dones(&self) -> &[f32] {
        &self.done[..self.size]
    }

    pub fn contexts(&self) -> &[f32] {
        &self.context[..self.size * self.embed_dim]
    }

    pub fn summaries(&self) -> &[f32] {
        &self.summary[..self.size * self.embed_dim]
    }

    pub fn held(&self) -> &[f32] {
        &self.held[..self.size * self.n_keys]
    }

    pub fn log_probs(&self) -> &[f32] {
        &self.logp[..self.size]
    }

    pub fn resets(&self) -> &[f32] {
        &self.reset[..self.size]
    }

    pub fn windows(&self) -> &[f32] {
        &self.window[..self.size * self.mem_tokens * self.embed_dim]
    }

    /// The (T, B) index grid for one minibatch, flattened row-major.
    ///
    /// A minibatch is a set of whole sequences of consecutive steps. The memory
    /// window stored at each start index seeds the transformer, so the forward
    /// pass sees exactly the history the policy had at the time, and gradients
    /// are cut at the sequence boundary - the standard truncated-BPTT
    /// arrangement, and what makes a transformer PPO update both correct and
    /// cheap.
    pub fn sequence_batches(
        &self,
        seq_len: usize,
        minibatch: usize,
        rng: &mut impl RngExt,
    ) -> Vec<Vec<usize>> {
        let seq_len = seq_len.max(2);
        let n_sequences = self.size / seq_len;
        if n_sequences == 0 {
            return Vec::new();
        }
        let mut starts: Vec<usize> = (0..n_sequences).map(|i| i * seq_len).collect();
        starts.as_mut_slice().shuffle(rng);
        let per_batch = (minibatch / seq_len).max(1);
        let mut batches = Vec::new();
        let mut begin = 0;
        while begin < n_sequences {
            let chunk = &starts[begin..(begin + per_batch).min(n_sequences)];
            begin += per_batch;
            if chunk.is_empty() {
                continue;
            }
            // (T, B) row-major: sequence t of batch entry j is step
            // `chunk[j] + t`.
            let mut indices = Vec::with_capacity(seq_len * chunk.len());
            for step in 0..seq_len {
                for start in chunk {
                    indices.push(start + step);
                }
            }
            batches.push(indices);
        }
        batches
    }

    /// Gather one minibatch's arrays out of the rollout.
    pub fn gather(&self, indices: &[usize]) -> Batch {
        let rows = indices.len();
        let embed = self.embed_dim;
        let mem = self.mem_tokens;
        let n_keys = self.n_keys;
        let mut hidden = Vec::with_capacity(rows * mem * embed);
        let mut summary = Vec::with_capacity(rows * embed);
        let mut held = Vec::with_capacity(rows * n_keys);
        let mut turn = Vec::with_capacity(rows);
        let mut speed = Vec::with_capacity(rows);
        let mut tap = Vec::with_capacity(rows);
        let mut logp = Vec::with_capacity(rows);
        for index in indices {
            hidden.extend_from_slice(&self.window[index * mem * embed..(index + 1) * mem * embed]);
            summary.extend_from_slice(&self.summary[index * embed..(index + 1) * embed]);
            held.extend_from_slice(&self.held[index * n_keys..(index + 1) * n_keys]);
            turn.push(self.turn[*index]);
            speed.push(self.speed[*index]);
            tap.push(self.tap[*index]);
            logp.push(self.logp[*index]);
        }
        Batch {
            rows,
            hidden,
            summary,
            held,
            turn,
            speed,
            tap,
            logp,
        }
    }

    pub fn action_vectors(&self) -> Option<&[f32]> {
        self.action_vector
            .as_ref()
            .map(|storage| &storage[..self.size * self.action_vector_dim])
    }

    pub fn action_vector_dim(&self) -> usize {
        self.action_vector_dim
    }
}

/// Generalised advantage estimation, backwards, as plain arithmetic: this is a
/// few thousand floats, not a tensor job.
///
/// `dones` masks the value bootstrap on a genuine terminal step only. A
/// time-limit truncation is bootstrapped like any other step, because the game
/// did not actually end - treating a segment boundary as a real end is how a
/// value function learns that the world is about to disappear.
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    last_value: f32,
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = rewards.len();
    let mut advantages = vec![0.0f32; n];
    let mut last = 0.0f32;
    for t in (0..n).rev() {
        let next_value = if t == n - 1 { last_value } else { values[t + 1] };
        let not_done = 1.0 - dones[t];
        let delta = rewards[t] + gamma * next_value * not_done - values[t];
        last = delta + gamma * lambda * not_done * last;
        advantages[t] = last;
    }
    let returns: Vec<f32> = advantages
        .iter()
        .zip(values)
        .map(|(advantage, value)| advantage + value)
        .collect();
    (advantages, returns)
}

/// Zero-mean, unit-variance advantages.
///
/// The guard against a degenerate rollout is the point: with a single sample, or
/// with every advantage identical, the standard deviation is zero and dividing
/// by it produces either NaN or - if the code special-cases it - a gradient of
/// exactly zero. That silent zero is the classic way a run "trains" for hours
/// without changing at all.
pub fn normalise_advantages(advantages: &mut [f32]) {
    let n = advantages.len();
    if n == 0 {
        return;
    }
    let mean = advantages.iter().sum::<f32>() / n as f32;
    let variance = advantages
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f32>()
        / n as f32;
    let std = variance.sqrt();
    if std < 1e-6 {
        for value in advantages.iter_mut() {
            *value -= mean;
        }
        return;
    }
    for value in advantages.iter_mut() {
        *value = (*value - mean) / (std + 1e-8);
    }
}

pub fn variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f32>()
        / values.len() as f32
}

pub fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_buffer_refuses_to_overflow() {
        let mut buffer = RolloutBuffer::new(2, 4, 2, 3);
        let zeros = [0.0f32; 4];
        let window = [0.0f32; 8];
        let held = [0.0f32; 3];
        for _ in 0..2 {
            buffer.add(
                &zeros, &zeros, &window, &held, 0, 0, 0, 0.0, 0.0, 0.0, false, false, None,
            );
        }
        assert!(buffer.is_full());
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn sequence_batches_cover_every_sequence_once() {
        let mut buffer = RolloutBuffer::new(8, 2, 1, 1);
        let zeros = [0.0f32; 2];
        let window = [0.0f32; 2];
        let held = [0.0f32; 1];
        for _ in 0..8 {
            buffer.add(
                &zeros, &zeros, &window, &held, 0, 0, 0, 0.0, 0.0, 0.0, false, false, None,
            );
        }
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(1);
        let batches = buffer.sequence_batches(4, 8, &mut rng);
        assert_eq!(batches.len(), 1);
        // (T=4, B=2) row-major: two sequences of four consecutive steps.
        assert_eq!(batches[0], vec![0, 4, 1, 5, 2, 6, 3, 7]);
    }

    #[test]
    fn gae_matches_a_hand_computed_case() {
        // One step, no further value, reward 1: advantage is 1 minus the
        // critic's guess, and the return is the reward.
        let advantages = compute_gae(&[1.0], &[0.25], &[0.0], 0.0, 0.99, 0.95);
        assert!((advantages.0[0] - 0.75).abs() < 1e-6);
        assert!((advantages.1[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalising_a_constant_rollout_does_not_divide_by_zero() {
        let mut advantages = vec![2.0f32; 4];
        normalise_advantages(&mut advantages);
        assert!(advantages.iter().all(|value| value.is_finite()));
        assert!(advantages.iter().all(|value| value.abs() < 1e-6));
    }
}
