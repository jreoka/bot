//! The session: one game, one policy, and the loop that runs them. Ported from
//! `botcore/session.py`.
//!
//! This is the piece that decides whether the bot keeps making progress. It owns
//! the outer loop and, critically, every health check in it: a run that has
//! stopped learning should say so in the log, with a specific reason, rather
//! than leaving the user to notice that the screen has been the same for an
//! hour.
//!
//! The environment is a trait here rather than a duck-typed object. `reset`,
//! `step`, `release_all` and observations are all an environment has to provide,
//! so the same session runs against a real window or the synthetic game - the
//! same arrangement as the Python, with the shape written down.
//!
//! One deliberate departure from the Python: **the update does not run on the
//! thread that plays the game.** It used to, and it meant the bot stood still
//! for as long as PPO took - seconds, every rollout, with no frame captured and
//! nothing pressed. `LearnWorker` moves it to a thread of its own, and the
//! control loop only ever swaps weights in at a rollout boundary. Everything the
//! update needs (the rollout, the weights it was collected with, the bootstrap
//! value) moves with the job, so the arithmetic is the same; what changes is
//! that the game keeps being played while it happens.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;

use crate::config::Config;
use crate::keys::{ActionSpace, Keymap};
use crate::model::{Action, ActorCritic, Net};
use crate::novelty::{IntrinsicReward, Signature};
use crate::replay::RolloutBuffer;
use crate::trainer::Trainer;
use crate::vision::{Observation, gray_signature};

/// What an environment is handed for one step.
///
/// A struct rather than parallel arguments, because the two consumers want
/// different things from it: a real game wants the keys and the pixel delta, the
/// synthetic game wants the held mask and the turn index.
#[derive(Debug, Clone)]
pub struct EnvAction {
    pub held_vks: Vec<u32>,
    pub turn: (i32, i32),
    pub tap_vk: u32,
    pub index: usize,
    pub held_mask: u32,
    pub turn_index: usize,
    pub speed_index: usize,
    pub tap_index: usize,
    pub mouse_step: i32,
}

/// What an environment reports back from one step.
#[derive(Debug, Clone, Default)]
pub struct EnvStep {
    /// The environment genuinely ended (a closed window, or a synthetic episode
    /// hitting its limit). An inferred in-game reset is *not* this.
    pub truncated: bool,
    pub died: bool,
    /// The environment's own progress measure, when it has one. Never read by
    /// the reward - only the self-test looks at it.
    pub progress: i32,
}

/// Input/delivery counters, when an environment can report them.
#[derive(Debug, Clone, Default)]
pub struct InputStats {
    pub focused: bool,
    pub delivered: u64,
    pub skipped_unfocused: u64,
    pub window: String,
    pub non_game: bool,
    pub title: String,
    pub process: String,
}

/// One game, as the session sees it.
pub trait Environment {
    fn observation_shape(&self) -> (usize, usize, usize);
    /// How many channels of the observation are grey frames; the rest are the
    /// difference channel the observation stack computed.
    fn gray_channels(&self) -> usize;
    fn reset(&mut self, seed: Option<u64>) -> (Observation, Signature);
    fn step(&mut self, action: &EnvAction) -> (Observation, EnvStep);
    fn release_all(&mut self) {}
    fn close(&mut self) {}
    fn acquire_focus(&mut self) -> Option<bool> {
        None
    }
    fn suspend(&mut self) {}
    fn resume(&mut self) {}
    fn input_stats(&self) -> Option<InputStats> {
        None
    }
    fn current_observation(&mut self) -> Option<Observation> {
        None
    }
}

/// Tracks the slow-moving quantities that reveal a run going wrong.
pub struct HealthMonitor {
    cfg: Config,
    novel_steps: u64,
    novelty_series: VecDeque<f32>,
    engaged_series: VecDeque<f32>,
    frozen_since: Option<Instant>,
    frozen_reported: bool,
    pub last_delta: f32,
    noop_streak: u64,
    pub longest_noop_streak: u64,
}

impl HealthMonitor {
    pub fn new(cfg: &Config) -> Self {
        Self {
            cfg: cfg.clone(),
            novel_steps: 0,
            novelty_series: VecDeque::new(),
            engaged_series: VecDeque::new(),
            frozen_since: None,
            frozen_reported: false,
            last_delta: 0.0,
            noop_streak: 0,
            longest_noop_streak: 0,
        }
    }

    /// Feed one step. Returns a message the first time a problem appears.
    pub fn note_step(&mut self, delta: f32, engaged: bool, episodic_bonus: f32) -> Option<String> {
        self.last_delta = delta;
        if episodic_bonus > 0.0 {
            self.novel_steps += 1;
        }
        if engaged {
            self.noop_streak = 0;
        } else {
            self.noop_streak += 1;
            self.longest_noop_streak = self.longest_noop_streak.max(self.noop_streak);
        }

        if self.cfg.frozen_warn_seconds <= 0.0 {
            return None;
        }
        if delta < 1e-4 {
            match self.frozen_since {
                None => self.frozen_since = Some(Instant::now()),
                Some(since) => {
                    if !self.frozen_reported
                        && since.elapsed().as_secs_f32() >= self.cfg.frozen_warn_seconds
                    {
                        self.frozen_reported = true;
                        return Some(format!(
                            "[Health] The screen has not changed for {:.0}s. The game has \
                             almost certainly stopped rendering or simulating while \
                             unfocused, so there is nothing to learn from. Fix it in the \
                             game (disable 'pause when unfocused', enable background \
                             running, use borderless windowed) or the window is minimized.",
                            self.cfg.frozen_warn_seconds
                        ));
                    }
                }
            }
        } else {
            self.frozen_since = None;
            self.frozen_reported = false;
        }
        None
    }

    /// Called when the status block is printed; returns diagnoses.
    pub fn end_window(
        &mut self,
        steps: u64,
        engaged_steps: u64,
        update_seconds: f64,
    ) -> Vec<String> {
        let mut messages = Vec::new();
        if steps == 0 {
            return messages;
        }
        let novelty_rate = 1000.0 * self.novel_steps as f32 / steps as f32;
        let engaged_rate = 100.0 * engaged_steps as f32 / steps as f32;
        push_capped(&mut self.novelty_series, novelty_rate, 8);
        push_capped(&mut self.engaged_series, engaged_rate, 8);
        let _ = update_seconds;

        if self.novelty_series.len() >= 4 {
            let series: Vec<f32> = self.novelty_series.iter().copied().collect();
            let recent = &series[series.len() - 2..];
            let earlier = &series[..series.len() - 2];
            let earlier_mean = earlier.iter().sum::<f32>() / earlier.len().max(1) as f32;
            if recent.iter().copied().fold(f32::NEG_INFINITY, f32::max) < 1.0
                && earlier_mean > 5.0
            {
                messages.push(
                    "[Health] New-state rate has flattened: the bot is no longer reaching \
                     situations it has not seen. Either it has exhausted what this action \
                     set can reach (add keys with --calibrate), or it has settled into a \
                     loop. Check the action histogram below: one action dominating means a \
                     loop, many actions with no novelty means saturation."
                        .to_string(),
                );
            }
        }

        if engaged_rate < 15.0 {
            messages.push(format!(
                "[Health] The policy pressed nothing on {:.0}% of steps. Intrinsic reward \
                 can settle on noop because it is never punished by dying. Raise w_idle, or \
                 raise entropy_coef to force exploration back.",
                100.0 - engaged_rate
            ));
        }
        if self.longest_noop_streak > self.cfg.status_every as u64 {
            messages.push(format!(
                "[Health] Longest run of idle decisions: {}. The policy is stuck on noop.",
                self.longest_noop_streak
            ));
        }
        messages
    }

    pub fn start_window(&mut self) {
        self.novel_steps = 0;
        self.longest_noop_streak = 0;
    }
}

fn push_capped(series: &mut VecDeque<f32>, value: f32, cap: usize) {
    if series.len() >= cap {
        series.pop_front();
    }
    series.push_back(value);
}

// =============================================================================
// The learning worker
// =============================================================================
//
// The update used to run on the thread that injects input, and that is the one
// thing this program must not do: while `Trainer::update` is running, no frame
// is captured and no key is pressed, so the bot stops playing for as long as the
// update takes - seconds, every rollout. On a CPU backend that is long enough for
// the game to move on without it.
//
// So the update runs here instead, on a thread of its own, against its own copy
// of the policy. The session keeps collecting with the copy it has; when the
// worker finishes, its weights are swapped in at the next rollout boundary. The
// control loop therefore never waits for training, and the pause is gone rather
// than merely shortened.
//
// What this costs is bounded staleness: while an update runs, the frames being
// collected come from the policy *before* that update, exactly as they did when
// the update ran inline - the difference is that the bot keeps playing instead of
// standing still. The update itself still starts from the same weights the
// rollout was collected with, so PPO's ratio is computed against the behavior
// policy and nothing is silently off-policy.

/// One rollout, handed to the learning thread.
struct LearnJob<B: AutodiffBackend> {
    /// The weights this rollout was collected with: what the update starts from.
    weights: Net<B>,
    buffer: RolloutBuffer,
    /// Value of the observation after the last step, for the GAE tail.
    last_value: f32,
}

/// What the learning thread hands back.
struct LearnResult<B: AutodiffBackend> {
    weights: Net<B>,
    seconds: f64,
    updates: u64,
    metrics: HashMap<String, f32>,
}

enum WorkerRequest<B: AutodiffBackend> {
    Update(LearnJob<B>),
}

/// Owns the learning thread and the last finished update.
struct LearnWorker<B: AutodiffBackend> {
    requests: mpsc::Sender<WorkerRequest<B>>,
    /// One edge per finished update; the update itself travels through `running`.
    results: mpsc::Receiver<()>,
    running: Arc<Mutex<Option<LearnResult<B>>>>,
    /// Updates submitted and not yet finished. Shared with the learning thread
    /// rather than inferred from timestamps, so the control loop can tell
    /// "training" from "idle" without guessing.
    pending: Arc<std::sync::atomic::AtomicUsize>,
    /// A worker whose channel has closed: it failed to start, or its thread
    /// died. Training is disabled rather than being allowed to take the run down,
    /// and the status block says so.
    dead: bool,
}

/// Marks one accepted job as outstanding, however the job's thread ends.
struct PendingGuard(Arc<std::sync::atomic::AtomicUsize>);

impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

impl<B: AutodiffBackend> LearnWorker<B> {
    /// Spawn the thread. `policy` is the weights to train from and `trainer` is
    /// moved outright: the optimiser's state belongs to the training thread, and
    /// with one worker there is nothing to share it with.
    fn spawn(mut trainer: Trainer<B>, mut policy: ActorCritic<B>) -> Option<Self> {
        let (request_tx, request_rx) = mpsc::channel::<WorkerRequest<B>>();
        // Sends nothing: the result itself travels through the slot, and the
        // channel is only the "a result is ready" edge. One lock of that slot per
        // update is cheaper than moving a whole parameter tree through a channel.
        let (result_tx, result_rx) = mpsc::channel::<()>();
        let running: Arc<Mutex<Option<LearnResult<B>>>> = Arc::new(Mutex::new(None));
        let slot = running.clone();
        let pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let outstanding = pending.clone();

        std::thread::Builder::new()
            .name("bot1-learn".to_string())
            .spawn(move || {
                // Recv, not loop-until-stop: the sender is owned by the session,
                // so the thread ends when the run does and cannot outlive it.
                while let Ok(WorkerRequest::Update(job)) = request_rx.recv() {
                    // One outstanding job at a time, so this cannot subtract
                    // another job's count: the session only submits when this is
                    // zero. The guard covers every exit from the iteration,
                    // including the `break` below.
                    let _job_done = PendingGuard(outstanding.clone());
                    // The policy starts from the weights the rollout was
                    // collected with, which are on the job - never from whatever
                    // this thread happens to hold.
                    policy.net = job.weights;
                    let started = Instant::now();
                    let mut metrics = trainer.update(&mut policy, &job.buffer, job.last_value);
                    // Whatever is left of the update budget is shared with the
                    // curiosity head, as before - except that this no longer
                    // steals frames, so a slow machine collects throughout.
                    let spent = started.elapsed().as_secs_f64();
                    let remaining =
                        (trainer.cfg.update_seconds_budget as f64 * 0.5 - spent).max(0.0);
                    if job.buffer.action_vectors().is_some() && remaining > 0.0 {
                        metrics.extend(trainer.train_transition(
                            &mut policy,
                            &job.buffer,
                            trainer.cfg.minibatch_size,
                            2,
                            remaining,
                        ));
                    }
                    let result = LearnResult {
                        weights: policy.net.clone(),
                        seconds: started.elapsed().as_secs_f64(),
                        updates: trainer.updates,
                        metrics,
                    };
                    // The channel read is the guard: it is sent only after the
                    // slot holds the result.
                    let stored = match slot.lock() {
                        Ok(mut held) => {
                            *held = Some(result);
                            true
                        }
                        Err(_) => false,
                    };
                    if !stored || result_tx.send(()).is_err() {
                        break;
                    }
                }
            })
            .ok()?;

        Some(Self {
            requests: request_tx,
            results: result_rx,
            running,
            pending,
            dead: false,
        })
    }

    /// Hand an update over. Never blocks: the worker keeps its own copy of the
    /// weights the job carries.
    fn submit(&mut self, job: LearnJob<B>) {
        if self.dead {
            return;
        }
        self.pending
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if self.requests.send(WorkerRequest::Update(job)).is_err() {
            self.pending
                .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            self.dead = true;
        }
    }

    /// Updates the worker has accepted and not yet finished.
    fn in_flight(&self) -> usize {
        self.pending.load(std::sync::atomic::Ordering::Acquire)
    }

    /// The newest finished update, or `None`.
    ///
    /// A backlog is drained and only its last result kept: applying an update
    /// that a newer one has already superseded would step the optimiser through
    /// weights the main thread never saw.
    fn poll(&mut self) -> Option<LearnResult<B>> {
        let mut latest: Option<LearnResult<B>> = None;
        while let Ok(()) = self.results.try_recv() {
            match self.running.lock() {
                Ok(mut held) => {
                    if let Some(result) = held.take() {
                        latest = Some(result);
                    }
                }
                Err(_) => break,
            }
        }
        latest
    }

    /// Wait up to `timeout`, collecting everything the worker produces.
    ///
    /// Used once, when the run is stopping: one bounded wait is what keeps the
    /// last rollout from being thrown away without making a user who asked the
    /// bot to stop watch it train for another six seconds.
    fn wait_for_update(&mut self, timeout: std::time::Duration) -> Option<LearnResult<B>> {
        let mut latest: Option<LearnResult<B>> = None;
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if self.results.recv_timeout(remaining).is_err() {
                break;
            }
            match self.running.lock() {
                Ok(mut held) => {
                    if let Some(result) = held.take() {
                        latest = Some(result);
                    }
                }
                Err(_) => break,
            }
            if Instant::now() >= deadline {
                break;
            }
        }
        // A timeout does not mean the worker produced nothing: it may have
        // finished as the wait expired, and that result must not be left in the
        // slot to be picked up by a later call.
        latest.or_else(|| self.poll())
    }
}

/// Drives one game with one policy.
pub struct GameSession<B: AutodiffBackend> {
    pub cfg: Config,
    pub action_space: ActionSpace,
    /// The model being trained.
    pub policy: ActorCritic<B>,
    /// The same weights on the non-autodiff backend, for the per-step forward
    /// pass: collection does not need a graph.
    pub acting: ActorCritic<B::InnerBackend>,
    pub reward: IntrinsicReward,
    pub trainer: Trainer<B>,
    pub buffer: RolloutBuffer,
    pub health: HealthMonitor,

    pub env: Box<dyn Environment>,
    pub device: B::Device,
    rng: rand::rngs::StdRng,

    pub observation_shape: (usize, usize, usize),
    pub observation: Observation,
    pub observation_signature: Signature,
    hidden: Tensor<B::InnerBackend, 3>,

    // What the last decision produced, kept between `decide` and `step`.
    pending_log_prob: f32,
    pending_value: f32,
    pending_window: Vec<f32>,
    pending_summary: Vec<f32>,
    pending_context: Vec<f32>,
    pending_prediction: Vec<f32>,
    pending_has_prediction: bool,

    pub env_seconds: f64,
    pub update_seconds: f64,
    /// Set once `run()` has started the learning worker: from then on training
    /// happens off the control loop and never blocks it.
    learn_offloaded: bool,
    learn_worker: Option<LearnWorker<B>>,
    /// Rollouts collected while the worker was still busy, and therefore not
    /// trained on. The reason `trainer.updates` can lag the collector.
    learn_skipped: u64,
    /// Finished updates that a newer one superseded before the session could
    /// apply them. Should stay at zero: only one job is ever outstanding, so
    /// this is a backstop that says so if the accounting ever drifts.
    learn_superseded: u64,
    /// Updates the worker finished and the session accepted.
    learn_updates: u64,
    /// Set when the worker produced non-finite weights, which stops it being
    /// applied. The run continues on the last good weights.
    learn_disabled_after_nan: bool,
    pub episodes: u64,
    pub decisions_made: u64,
    pub total_steps: u64,
    pub last_value: f32,
    started: Instant,
    status_window_steps: u64,
    status_engaged: u64,
    status_novel: u64,
    held_vks: Vec<u32>,
    action_counts: HashMap<String, u64>,
    turn_decisions: u64,
    turn_observations: u64,
    turn_window: VecDeque<bool>,
    /// The `--log` file: one JSON object per update, appended.
    ///
    /// A run that is watched for hours needs a record that survives the scroll
    /// buffer, and a line per update is what makes "when did it stop improving"
    /// a question with an answer.
    log: Option<std::fs::File>,
}

impl<B: AutodiffBackend> GameSession<B> {
    pub fn new(
        cfg: &Config,
        env: Box<dyn Environment>,
        action_space: ActionSpace,
        keymap: Option<&Keymap>,
        device: B::Device,
    ) -> anyhow::Result<Self> {
        let cfg = cfg.clone();
        cfg.validate()?;
        let observation_shape = env.observation_shape();
        let channels = observation_shape.0;

        // The model's own initialisation, so a Rust run starts from the same
        // distribution the Python does. The RNG is seeded from the config:
        // `torch`'s default generator is never seeded in the Python's self-test,
        // so its initial weights differ from run to run - this does not, which is
        // what makes a failed self-test reproducible.
        let mut model_rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(cfg.seed);
        let mut policy = ActorCritic::<B>::new(&cfg, action_space.head_sizes(), channels, &device);
        policy.init(&mut model_rng, &device);

        let buffer = RolloutBuffer::new(
            cfg.rollout_steps,
            cfg.embed_dim,
            cfg.mem_tokens,
            action_space.hold_vks.len(),
        );
        let reward = IntrinsicReward::new(&cfg, action_space.action_vector_size());
        let trainer = Trainer::new(&cfg, &device);
        let acting = ActorCritic::<B::InnerBackend>::from_net(policy.net.valid(), &policy);
        let hidden = acting.initial_hidden(1, &device);
        let health = HealthMonitor::new(&cfg);

        let mut session = Self {
            observation_shape,
            observation: Observation::zeros(
                observation_shape.0,
                observation_shape.1,
                observation_shape.2,
            ),
            observation_signature: Vec::new(),
            hidden,
            pending_log_prob: 0.0,
            pending_value: 0.0,
            pending_window: vec![0.0; cfg.mem_tokens * cfg.embed_dim],
            pending_summary: vec![0.0; cfg.embed_dim],
            pending_context: vec![0.0; cfg.embed_dim],
            pending_prediction: vec![0.0; cfg.embed_dim],
            pending_has_prediction: false,
            env_seconds: 0.0,
            update_seconds: 0.0,
            learn_offloaded: false,
            learn_worker: None,
            learn_skipped: 0,
            learn_superseded: 0,
            learn_updates: 0,
            learn_disabled_after_nan: false,
            episodes: 0,
            decisions_made: 0,
            total_steps: 0,
            last_value: 0.0,
            started: Instant::now(),
            status_window_steps: 0,
            status_engaged: 0,
            status_novel: 0,
            held_vks: Vec::new(),
            action_counts: HashMap::new(),
            turn_decisions: 0,
            turn_observations: 0,
            turn_window: VecDeque::new(),
            // Opened in append mode: a resumed run continues the same log rather
            // than truncating what earlier runs recorded.
            log: cfg.log_json.as_ref().and_then(|path| {
                if let Some(parent) = std::path::Path::new(path).parent()
                    && !parent.as_os_str().is_empty()
                {
                    let _ = std::fs::create_dir_all(parent);
                }
                match std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    Ok(file) => Some(file),
                    Err(error) => {
                        println!("[Log] Could not open '{path}': {error}");
                        None
                    }
                }
            }),
            rng: <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
                cfg.seed ^ 0x9E37_79B9_7F4A_7C15,
            ),
            cfg,
            action_space,
            policy,
            acting,
            reward,
            trainer,
            buffer,
            health,
            env,
            device,
        };

        // The view depth the bot starts from: `--calibrate` measures it and
        // `--mouse-turn` pins it, and either way the bot is free to move it
        // inside the configured bounds afterwards.
        if let Some(keymap) = keymap {
            let clamped = keymap
                .default_mouse_turn
                .clamp(session.cfg.mouse_turn_min, session.cfg.mouse_turn_max);
            session.policy.set_mouse_step(Some(clamped as f32));
            session.acting.mouse_step = session.policy.mouse_step;
        }
        if session.cfg.mouse_turn_pixels != 0 {
            session
                .policy
                .set_mouse_step(Some(session.cfg.mouse_turn_pixels as f32));
            session.acting.mouse_step = session.policy.mouse_step;
        }
        Ok(session)
    }

    /// Rebuild the collection copy of the weights after an update.
    pub fn refresh_acting(&mut self) {
        self.acting =
            ActorCritic::<B::InnerBackend>::from_net(self.policy.net.valid(), &self.policy);
    }

    // =====================================================================
    // The loop
    // =====================================================================

    /// Block until the user presses the start key. Returns false if they quit
    /// instead.
    ///
    /// Nothing at all is sent before this, which is what makes the sequence
    /// "start the script, click the game, press the start key" safe: the bot
    /// cannot type into whichever window happened to have focus at launch.
    pub fn wait_for_start(&mut self, control: &mut dyn crate::control::Control) -> bool {
        if !control.awaiting_start() {
            return true;
        }
        println!();
        println!(
            "[Control] Ready. Click the game window so it has focus, then press the start key."
        );
        println!(
            "[Control] Nothing is being sent until then, so it is safe to use the terminal."
        );
        while control.awaiting_start() && !control.stop_requested() {
            for message in control.service() {
                println!("[Control] {message}");
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        if control.stop_requested() {
            println!("[Control] Quit before starting; nothing was sent.");
            return false;
        }
        true
    }

    /// The observation the environment is showing right now, or the last one.
    fn current_observation(&mut self) -> crate::vision::Observation {
        if let Some(fresh) = self.env.current_observation() {
            return fresh;
        }
        self.observation.clone()
    }

    /// Collect, learn, report, forever (or until `max_decisions`).
    ///
    /// `control` is the thing that can pause, checkpoint and stop the run; a run
    /// with no control simply goes until its budget is spent. `save_fn` is called
    /// when a checkpoint is due.
    ///
    /// Collection and training run on different threads here: the update is
    /// handed to a worker and the next rollout starts immediately, so the bot
    /// never stops playing while it learns.
    pub fn run(
        &mut self,
        mut control: Option<&mut dyn crate::control::Control>,
        max_decisions: Option<u64>,
        mut save_fn: Option<&mut dyn FnMut(&Self, &str)>,
    ) {
        if let Some(inner) = control.as_mut()
            && !self.wait_for_start(*inner)
        {
            return;
        }
        // One attempt at the foreground window, now that the user has said "go".
        // Per the focus policy this never happens again on its own.
        self.focus();
        self.reset(Some(self.cfg.seed), false);
        self.start_learning_worker();
        if !self.learning_is_offloaded() {
            println!(
                "[Learn] The training thread could not be started, so updates will run \
                 inline and the bot will pause while it learns."
            );
        }
        let mut last_status = self.total_steps;

        while max_decisions.is_none_or(|max| self.total_steps < max) {
            if let Some(inner) = control.as_mut() {
                for message in inner.service() {
                    println!("[Control] {message}");
                }
                if inner.paused() {
                    if inner.claim_pause_notice() {
                        println!(
                            "[Control] PAUSED: no input is being sent. Press the pause key \
                             again to resume."
                        );
                    }
                    // Suspending the environment is what actually stops the
                    // input, not merely releasing what is held: without it the
                    // next injected key would land while the user believes they
                    // are paused.
                    self.suspend();
                    while inner.paused() && !inner.stop_requested() {
                        for message in inner.service() {
                            println!("[Control] {message}");
                        }
                        self.release_held();
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    self.resume();
                    self.focus();
                    // The world moved on while paused, so the transformer's
                    // memory window and the pending observation are no longer a
                    // chain.
                    self.observation = self.current_observation();
                    self.observation_signature = self.signature_of(&self.observation.clone());
                    self.hidden = self.acting.initial_hidden(1, &self.device);
                    self.reward
                        .reset_episode(Some(&self.observation_signature.clone()));
                    self.decide();
                }
                let due = inner.checkpoint_due();
                if let (Some(reason), Some(save)) = (due, save_fn.as_mut()) {
                    save(self, &reason);
                }
                if inner.stop_requested() {
                    break;
                }
            }

            while self.buffer.len() < self.cfg.rollout_steps {
                let action = self.decide();
                let (_reward, done, _info) = self.step(&action);
                if done {
                    // `reset` re-decides for the fresh observation, so the
                    // pending action always matches the pending observation. The
                    // rollout survives: an episode boundary is not the end of
                    // the data-collection run.
                    self.reset(None, true);
                }
                let Some(inner) = control.as_mut() else {
                    continue;
                };
                // Checked inside the collection loop as well: a rollout can take
                // a minute at a low control rate, and a checkpoint that waits for
                // a whole rollout is a checkpoint that is missing when the
                // machine loses power.
                for message in inner.service() {
                    println!("[Control] {message}");
                }
                let due = inner.checkpoint_due();
                if let (Some(reason), Some(save)) = (due, save_fn.as_mut()) {
                    save(self, &reason);
                }
                if inner.paused() || inner.stop_requested() {
                    break;
                }
            }

            let stopping = control
                .as_deref()
                .map(|inner| inner.stop_requested())
                .unwrap_or(false);
            if stopping {
                break;
            }
            // A pause caught mid-rollout drops the partial batch, because the
            // transitions either side of it are not a continuous chain.
            let pausing = control
                .as_deref()
                .map(|inner| inner.paused())
                .unwrap_or(false);
            if pausing {
                self.buffer.reset();
                continue;
            }

            // An update that finished while the last rollout was being collected
            // goes in here, at the boundary: every frame of a rollout is then
            // collected with one policy, and no frame is collected with weights
            // the update has already replaced halfway through a window.
            self.apply_trained_weights();

            self.bootstrapped_value();
            if !self.submit_learn() {
                // No worker (a caller driving the loop itself, or a worker that
                // failed to start): train inline, which is the pause this exists
                // to remove.
                self.learn();
            }

            // A NaN anywhere makes every later checkpoint worthless, so it is
            // caught here rather than hours later. With training off the control
            // thread this tests the weights still being collected with, which is
            // the set that would poison the run; the worker's own copy is checked
            // when it is applied.
            let bad = Trainer::<B>::nonfinite_parameters(&self.policy);
            if !bad.is_empty() {
                let shown: Vec<&String> = bad.iter().take(3).collect();
                println!(
                    "[Health] Non-finite weights in {shown:?}. Stopping so the previous \
                     checkpoint stays good."
                );
                break;
            }

            if self.total_steps - last_status >= self.cfg.status_every as u64 {
                last_status = self.total_steps;
                self.status_block();
            }
        }

        // One bounded wait, so the rollout collected just before the stop is not
        // thrown away - and only bounded: a user who asked the bot to stop must
        // not then watch it train for another six seconds.
        self.finish_learning();
    }

    // =====================================================================
    // Environment interaction
    // =====================================================================

    /// Start an episode.
    ///
    /// `keep_rollout` is what separates "this is a new episode" from "this is a
    /// new run". A game that ends every few hundred steps must not throw away
    /// the rollout being collected, or - if episodes are shorter than the
    /// rollout - the buffer would never fill and the bot would collect frames
    /// forever without ever updating. That failure is silent and looks exactly
    /// like a hang.
    pub fn reset(&mut self, seed: Option<u64>, keep_rollout: bool) {
        let (observation, _signature) = self.env.reset(seed);
        self.observation = observation;
        self.observation_signature = self.signature_of(&self.observation);
        self.hidden = self.acting.initial_hidden(1, &self.device);
        self.reward.reset_episode(Some(&self.observation_signature));
        self.episodes += 1;
        if !keep_rollout {
            self.buffer.reset();
            self.last_value = 0.0;
        }
        self.pending_summary = vec![0.0; self.cfg.embed_dim];
        self.pending_prediction = vec![0.0; self.cfg.embed_dim];
        self.pending_has_prediction = false;
        self.decide();
    }

    /// Choose an action for the current observation.
    ///
    /// Everything the later stages need is stashed here: the log-probability and
    /// value of the decision, the memory window in force *before* it (which is
    /// what the transformer PPO update needs), the pooled context and this
    /// frame's summary (which the curiosity head works from), and the model's
    /// prediction of the *next* frame, so the reward at the next step can score
    /// it without another forward pass.
    pub fn decide(&mut self) -> Action {
        let observation = Tensor::<B::InnerBackend, 4>::from_data(
            TensorData::new(
                self.observation.data.clone(),
                [
                    1,
                    self.observation.channels,
                    self.observation.height,
                    self.observation.width,
                ],
            ),
            &self.device,
        );
        let previous_window = self
            .hidden
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();

        let tokens = self.acting.encode(observation);
        let (new_hidden, context, features) =
            self.acting.forward(self.hidden.clone(), tokens.clone());
        let (hold_logits, turn_logits, speed_logits, tap_logits, value) =
            self.acting.heads(features, context.clone());

        // Sampling needs the model and the RNG at the same time; they are
        // separate fields, so the borrows do not overlap.
        let model = &self.acting;
        let held = sample_held(model, hold_logits.clone(), &mut self.rng);
        let direction = sample_categorical(&turn_logits, &mut self.rng);
        let speed = sample_categorical(&speed_logits, &mut self.rng);
        let tap = sample_categorical(&tap_logits, &mut self.rng);
        let held_tensor = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(held.clone(), [1, model.dims.n_keys]),
            &self.device,
        );
        let log_prob = scalar(&model.held_log_prob(hold_logits.clone(), held_tensor))
            + categorical_log_prob(&turn_logits, direction)
            + categorical_log_prob(&speed_logits, speed)
            + categorical_log_prob(&tap_logits, tap);
        let summary = values(tokens.mean_dim(1).reshape([1, self.cfg.embed_dim]));
        let context_host = values(context.clone());
        // What this frame says the next frame should look like; the reward
        // compares it against the real thing one step later.
        let prediction = values(model.predict_next(context));

        self.pending_log_prob = log_prob;
        self.pending_value = scalar(&value);
        self.pending_window = previous_window;
        self.pending_summary = summary;
        self.pending_context = context_host;
        self.pending_prediction = prediction;
        self.pending_has_prediction = true;
        self.hidden = new_hidden;

        Action {
            held,
            direction,
            speed,
            tap,
        }
    }

    /// Translate a factored decision into something an environment can apply.
    ///
    /// The pixel delta is computed here, from the bot's *current* mouse step,
    /// which is the part of the speed decision the bot owns.
    pub fn action_for_env(&self, action: &Action) -> EnvAction {
        let held_vks: Vec<u32> = action
            .held
            .iter()
            .enumerate()
            .filter(|(index, bit)| **bit > 0.5 && *index < self.action_space.hold_vks.len())
            .map(|(index, _)| self.action_space.hold_vks[index])
            .collect();
        let step = self.policy.mouse_step_value();
        let turn = self
            .action_space
            .turn_delta(action.direction, action.speed, step);
        let tap_vk = self
            .action_space
            .taps
            .get(action.tap)
            .map(|tap| tap.vk)
            .unwrap_or(0);
        let mut mask = 0u32;
        for (index, bit) in action.held.iter().take(16).enumerate() {
            if *bit > 0.5 {
                mask |= 1 << index;
            }
        }
        let index = ((mask as usize * self.action_space.n_turn() + action.direction)
            * self.action_space.n_speed()
            + action.speed)
            * self.action_space.n_tap()
            + action.tap;
        EnvAction {
            held_vks,
            turn,
            tap_vk,
            index,
            held_mask: mask,
            turn_index: action.direction,
            speed_index: action.speed,
            tap_index: action.tap,
            mouse_step: step,
        }
    }

    /// Apply a decision and record one transition.
    ///
    /// The same decision is held for `action_repeat` frames: one model
    /// evaluation per repeat frames is a straight throughput win, and holding a
    /// button for several frames is how the input reads to the game anyway.
    pub fn step(&mut self, action: &Action) -> (f32, bool, HashMap<String, f32>) {
        if self.buffer.is_full() {
            // Refuse loudly rather than dropping the transition. Silently
            // discarding rollout data would make the update non-on-policy,
            // which corrupts PPO in a way nothing else in the log would show.
            panic!("the rollout buffer is full; call learn() before stepping again");
        }
        let env_action = self.action_for_env(action);
        let started = Instant::now();
        let (mut observation, mut step_info) = self.env.step(&env_action);
        for _ in 0..self.cfg.action_repeat.saturating_sub(1) {
            if step_info.truncated {
                break;
            }
            let (next, next_info) = self.env.step(&env_action);
            observation = next;
            step_info.truncated |= next_info.truncated;
            step_info.died |= next_info.died;
            step_info.progress = next_info.progress;
        }
        self.env_seconds += started.elapsed().as_secs_f64();

        let engaged = action.held.iter().any(|bit| *bit > 0.5)
            || action.direction != 0
            || action.speed != 0
            || action.tap != 0;
        let signature = self.signature_of(&observation);

        // Reward, novelty and reset detection all reuse the representation and
        // the prediction the policy already produced: one transformer pass per
        // step, for every consumer.
        let action_vector =
            self.action_space
                .describe_action(&action.held, action.direction, action.speed, action.tap);
        let (reward, reset, reason) = self.reward.compute(
            Some(&self.pending_summary),
            &signature,
            &action_vector,
            engaged,
            self.total_steps,
            Some(&self.pending_prediction),
            self.pending_has_prediction,
        );

        self.buffer.add(
            &self.pending_context,
            &self.pending_summary,
            &self.pending_window,
            &action.held,
            action.direction,
            action.speed,
            action.tap,
            self.pending_log_prob,
            self.pending_value,
            reward,
            step_info.truncated,
            reset,
            Some(&action_vector),
        );
        self.held_vks = env_action.held_vks.clone();
        let label = self.action_label(action);
        *self.action_counts.entry(label).or_insert(0) += 1;
        self.note_turn(action, &signature);

        let mut delta = 0.0f32;
        if !self.observation_signature.is_empty() {
            delta = signature
                .iter()
                .zip(&self.observation_signature)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / signature.len().max(1) as f32;
        }
        if let Some(message) = self
            .health
            .note_step(delta, engaged, self.reward.episodic_bonus())
        {
            println!("{message}");
        }

        self.observation = observation;
        self.observation_signature = signature;
        self.decisions_made += 1;
        self.total_steps += 1;
        self.status_window_steps += 1;
        if engaged {
            self.status_engaged += 1;
        }
        if self.reward.episodic_bonus() > 0.0 {
            self.status_novel += 1;
        }
        if reset {
            self.episodes += 1;
            // The next transition is on the far side of a reset, so there is
            // nothing for the curiosity head to learn from across it.
            self.pending_summary = vec![0.0; self.cfg.embed_dim];
            self.pending_prediction = vec![0.0; self.cfg.embed_dim];
        }

        let info = HashMap::from([
            ("reward".to_string(), reward),
            ("reset".to_string(), if reset { 1.0 } else { 0.0 }),
            (
                "reset_reason".to_string(),
                if reason.is_empty() { 0.0 } else { 1.0 },
            ),
        ]);
        (reward, step_info.truncated, info)
    }

    // ---- helpers ----
    fn action_label(&self, action: &Action) -> String {
        let count = action.held.iter().filter(|bit| **bit > 0.5).count();
        let mut parts = vec![if count > 0 {
            format!("{count} key(s)")
        } else {
            "no keys".to_string()
        }];
        let (_name, delta) = self.action_space.direction(action.direction);
        if delta != (0, 0) {
            parts.push("turn".to_string());
        }
        if action.tap != 0 {
            parts.push("tap".to_string());
        }
        parts.join("+")
    }

    /// Watch whether turns actually move the view, and adjust the bot's own
    /// mouse step when they plainly do not.
    ///
    /// This is the half of "the bot sets its own mouse speed" that no policy
    /// gradient can reach. The policy learns *which* speed multiplier is worth
    /// choosing; what it cannot learn is what one pixel of mouse movement is
    /// worth in this game, because that is not a decision but a fact about the
    /// game. So it is measured instead, and deliberately slowly.
    fn note_turn(&mut self, action: &Action, signature: &[f32]) {
        let (_name, delta) = self.action_space.direction(action.direction);
        if delta == (0, 0) {
            return;
        }
        self.turn_decisions += 1;
        let moved = if self.observation_signature.is_empty() {
            0.0
        } else {
            signature
                .iter()
                .zip(&self.observation_signature)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / signature.len().max(1) as f32
        };
        if moved > 0.001 {
            self.turn_observations += 1;
        }

        let rate = self.cfg.mouse_adapt_rate;
        if rate <= 0.0 {
            return;
        }
        if self.turn_window.len() >= 20 {
            self.turn_window.pop_front();
        }
        self.turn_window.push_back(moved > 0.001);
        if self.turn_window.len() < 20 {
            return;
        }
        let moved_share = self.turn_window.iter().filter(|moved| **moved).count() as f32
            / self.turn_window.len() as f32;
        if moved_share <= 0.2 {
            // Repeatedly swinging the view and seeing the same picture: the step
            // is too small for whatever this game reads from the mouse.
            self.policy.nudge_mouse_step(1.0 + rate);
        } else if moved_share >= 0.98 {
            self.policy.nudge_mouse_step(1.0 / (1.0 + rate));
        }
        self.acting.mouse_step = self.policy.mouse_step;
    }

    /// What the bot is currently using as its turn step, and its range.
    pub fn mouse_speed_line(&self) -> String {
        let base = self.policy.mouse_step_value();
        let levels: Vec<f32> = if self.cfg.speed_levels.is_empty() {
            vec![1.0]
        } else {
            self.cfg.speed_levels.clone()
        };
        let rendered = levels
            .iter()
            .map(|level| format!("{}", ((base as f32) * level).round().max(1.0) as i32))
            .collect::<Vec<_>>()
            .join("/");
        format!(
            "mouse {base}px per 1x step, x[{rendered}] across {} speed(s) - {} of {} \
             turn(s) moved the view",
            levels.len(),
            self.turn_observations,
            self.turn_decisions
        )
    }

    /// Coarse fingerprint of an observation.
    ///
    /// Taken from the grey channels the model already sees - the position
    /// channels averaged out - so this is one pass and no colour conversion.
    /// Which channels those are depends on where the observation came from, so
    /// `gray_channels` is read from the environment rather than assumed; getting
    /// it wrong yields a constant fingerprint, which silently removes the entire
    /// episodic reward without raising anything.
    pub fn signature_of(&self, observation: &Observation) -> Signature {
        let count = self.env.gray_channels().max(1);
        let plane = observation.mean_of_channels(count);
        gray_signature(&plane, observation.height, observation.width, 32)
    }

    // =====================================================================
    // Learning
    // =====================================================================

    /// Start the learning worker, if it is not already running.
    ///
    /// Called by `run()`. From here on the update happens on another thread and
    /// the control loop never waits for it. The worker is given its own policy
    /// and the session's own `Trainer` - optimiser state included, since there is
    /// exactly one training thread - so the two only ever meet at a weight swap.
    pub fn start_learning_worker(&mut self) {
        if self.learn_offloaded {
            return;
        }
        let policy = self.policy.clone();
        let trainer = std::mem::replace(&mut self.trainer, Trainer::new(&self.cfg, &self.device));
        match LearnWorker::spawn(trainer, policy) {
            Some(worker) => {
                self.learn_worker = Some(worker);
                self.learn_offloaded = true;
            }
            None => {
                // The trainer went with the failed spawn, so put a fresh one
                // back: an inline update is a pause, but a run with no update at
                // all is a run that learns nothing.
                self.trainer = Trainer::new(&self.cfg, &self.device);
                self.learn_offloaded = false;
            }
        }
    }

    /// Whether training is running off this thread.
    pub fn learning_is_offloaded(&self) -> bool {
        self.learn_offloaded
    }

    /// Hand one unlocked rollout to the learning worker. `true` if it was
    /// accepted; `false` if there is nothing to hand over, in which case the
    /// caller falls through to [`GameSession::learn`].
    ///
    /// The rollout *is* the buffer, so it is moved out and replaced: collection
    /// starts a fresh one with no gap, which is the whole point of the worker.
    ///
    /// At most one update is allowed to be outstanding. If the worker is still
    /// busy, this rollout is dropped rather than queued - and dropped is the
    /// right outcome, not the lazy one: training that cannot keep up with
    /// collection is training on data that is further and further behind the
    /// policy playing the game, and a queue would only make that worse while
    /// growing without bound. The collector keeps playing throughout, which is
    /// the property that matters.
    pub fn submit_learn(&mut self) -> bool {
        if self.buffer.len() < 2 {
            return false;
        }
        let Some(worker) = self.learn_worker.as_mut() else {
            return false;
        };
        if worker.dead {
            return false;
        }
        if worker.in_flight() > 0 {
            self.buffer.reset();
            self.learn_skipped += 1;
            return true;
        }
        let buffer = std::mem::replace(
            &mut self.buffer,
            RolloutBuffer::new(
                self.cfg.rollout_steps,
                self.cfg.embed_dim,
                self.cfg.mem_tokens,
                self.action_space.hold_vks.len(),
            ),
        );
        // Cloned, not moved: the handles are shared, and the session keeps using
        // these weights to collect the next rollout.
        let weights = self.policy.net.clone();
        worker.submit(LearnJob {
            weights,
            buffer,
            last_value: self.last_value,
        });
        true
    }

    /// Swap in a finished update, if there is one.
    pub fn apply_trained_weights(&mut self) {
        // A backlog is drained and only its newest result kept: an update that a
        // later one has already superseded is not one the policy should step
        // through, and the collector is moving on regardless.
        let mut latest: Option<LearnResult<B>> = None;
        let Some(worker) = self.learn_worker.as_mut() else {
            return;
        };
        if let Some(first) = worker.poll() {
            latest = Some(first);
        }
        while let Some(newer) = worker.poll() {
            self.learn_superseded += 1;
            latest = Some(newer);
        }
        if let Some(result) = latest {
            self.accept_result(result);
        }
    }

    /// Apply one finished update and account for it.
    fn accept_result(&mut self, result: LearnResult<B>) {
        let bad = Trainer::<B>::nonfinite_parameters_from_net(&result.weights);
        if !bad.is_empty() {
            let shown: Vec<&String> = bad.iter().take(3).collect();
            self.learn_disabled_after_nan = true;
            println!(
                "[Health] The training thread produced non-finite weights in {shown:?}; its \
                 output is being discarded."
            );
            return;
        }
        self.apply_net(result.weights);
        self.update_seconds += result.seconds;
        // The trainer itself lives on the learning thread, so what it reports is
        // mirrored here: the status block is printed from this one, and a status
        // line describing the last update the *session* applied is the honest
        // one.
        self.trainer.last_metrics = result.metrics.clone();
        self.trainer.last_update_seconds = result.seconds;
        self.trainer.updates = result.updates;
        self.write_log_line(&result.metrics);
        self.learn_updates += 1;
    }

    /// Wait a bounded time for the update already in flight, so the last rollout
    /// is not thrown away when the run stops.
    pub fn finish_learning(&mut self) {
        let Some(worker) = self.learn_worker.as_mut() else {
            return;
        };
        if worker.dead {
            return;
        }
        let timeout = std::time::Duration::from_secs_f64(
            (self.cfg.update_seconds_budget.max(0.0) as f64).min(2.0),
        );
        let Some(result) = worker.wait_for_update(timeout) else {
            return;
        };
        self.accept_result(result);
    }

    /// Apply weights the worker trained, without restarting the collection copy's
    /// non-parameter state.
    ///
    /// The mouse step belongs to the session, not the optimiser: the worker's
    /// copy of it was taken when the job was handed over, and the session may
    /// have moved it since.
    fn apply_net(&mut self, net: Net<B>) {
        let mouse_step = self.policy.mouse_step;
        self.policy.net = net;
        self.policy.mouse_step = mouse_step;
        self.refresh_acting();
    }

    /// True while an update is in flight on the worker.
    pub fn learning_in_flight(&self) -> bool {
        self.learn_worker
            .as_ref()
            .is_some_and(|worker| !worker.dead && worker.in_flight() > 0)
    }

    /// Train a rollout on this thread.
    ///
    /// Kept because `--selftest` drives the session step by step and wants each
    /// update finished before it measures - and because a synchronous reference
    /// path is what makes the worker's arithmetic checkable. The run loop uses
    /// [`GameSession::submit_learn`] instead, so that it never blocks here.
    pub fn learn(&mut self) -> HashMap<String, f32> {
        if self.buffer.len() < 2 {
            return HashMap::new();
        }
        self.learn_offloaded = false;
        let started = Instant::now();
        let mut metrics = self
            .trainer
            .update(&mut self.policy, &self.buffer, self.last_value);

        // Share whatever is left of the update budget with the curiosity head,
        // so a slow machine still collects frames rather than silently spending
        // its whole life training curiosity.
        let spent = started.elapsed().as_secs_f64();
        let remaining = (self.cfg.update_seconds_budget as f64 * 0.5 - spent).max(0.0);
        if self.buffer.action_vectors().is_some() && remaining > 0.0 {
            metrics.extend(self.trainer.train_transition(
                &mut self.policy,
                &self.buffer,
                self.cfg.minibatch_size,
                2,
                remaining,
            ));
        }

        // The collection copy is refreshed once per update, not once per step.
        self.refresh_acting();
        self.update_seconds += started.elapsed().as_secs_f64();
        self.buffer.reset();
        self.write_log_line(&metrics);
        metrics
    }

    /// One JSON object per update, appended to `--log`.
    ///
    /// Written as a line rather than reformatted on exit, so a run that is killed
    /// still has every update it completed.
    fn write_log_line(&mut self, metrics: &HashMap<String, f32>) {
        let Some(file) = self.log.as_mut() else {
            return;
        };
        let mut record = serde_json::Map::new();
        record.insert("step".into(), self.total_steps.into());
        record.insert(
            "seconds".into(),
            serde_json::Value::from(self.started.elapsed().as_secs_f64()),
        );
        for (name, value) in metrics {
            // A NaN or an infinity is not valid JSON, and a log file one bad
            // number away from being unparseable is worse than a missing field.
            let value = if value.is_finite() { *value } else { 0.0 };
            record.insert(name.clone(), serde_json::Value::from(value as f64));
        }
        if let Ok(line) = serde_json::to_string(&serde_json::Value::Object(record)) {
            use std::io::Write;
            let _ = writeln!(file, "{line}");
            let _ = file.flush();
        }
    }

    // =====================================================================
    // Checkpoints
    // =====================================================================

    /// Everything about this run that is not a weight.
    pub fn meta(&self) -> crate::checkpoint::SessionMeta {
        let error = self.reward.error_running.state();
        crate::checkpoint::SessionMeta {
            version: crate::checkpoint::CHECKPOINT_VERSION,
            arch: crate::config::ARCH_NAME.to_string(),
            reward_version: crate::config::REWARD_VERSION,
            step: self.total_steps,
            episodes: self.episodes,
            seconds: self.started.elapsed().as_secs_f64(),
            mouse_step: self.policy.mouse_step,
            head_sizes: self.action_space.head_sizes().to_vec(),
            observation_shape: vec![
                self.observation_shape.0,
                self.observation_shape.1,
                self.observation_shape.2,
            ],
            error_mean: error.0,
            error_var: error.1,
            error_count: error.2,
            ambient: self.reward.detector.ambient,
            reward_steps: self.reward.steps,
            seed: self.cfg.seed,
        }
    }

    /// Write one checkpoint: the weights and the state together.
    pub fn save_checkpoint(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let tensors = self.policy.to_weights();
        let metadata = self.meta().to_metadata()?;
        let path = path.as_ref();
        let temporary = path.with_extension("safetensors.tmp");
        crate::weights::write_safetensors_with_metadata(&temporary, &tensors, &metadata)?;
        std::fs::rename(&temporary, path)?;
        Ok(())
    }

    /// Load one, after checking that it belongs to this build and this action
    /// space. Returns the metadata that was restored.
    pub fn load_checkpoint(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<crate::checkpoint::SessionMeta> {
        let weights = crate::weights::Weights::open(path.as_ref())?;
        let meta = crate::checkpoint::SessionMeta::from_metadata(&weights.metadata())?;
        meta.validate(
            crate::config::ARCH_NAME,
            crate::config::REWARD_VERSION,
            &self.action_space.head_sizes(),
            &[
                self.observation_shape.0,
                self.observation_shape.1,
                self.observation_shape.2,
            ],
        )?;
        self.policy.load(&weights, &self.device)?;
        self.refresh_acting();
        // The bot's own mouse step is part of what it has learned, so it comes
        // back with the weights rather than being reset to the calibration
        // default on every resume.
        self.policy.set_mouse_step(Some(meta.mouse_step));
        self.acting.mouse_step = self.policy.mouse_step;
        self.reward.error_running.load((
            meta.error_mean,
            meta.error_var,
            meta.error_count,
        ));
        self.reward.detector.ambient = meta.ambient;
        self.reward.steps = meta.reward_steps;
        self.total_steps = meta.step;
        self.episodes = meta.episodes;
        Ok(meta)
    }

    /// Value of the current observation, for the GAE tail.
    pub fn bootstrapped_value(&mut self) -> f32 {
        let observation = Tensor::<B::InnerBackend, 4>::from_data(
            TensorData::new(
                self.observation.data.clone(),
                [
                    1,
                    self.observation.channels,
                    self.observation.height,
                    self.observation.width,
                ],
            ),
            &self.device,
        );
        let (value, _hidden) = self.acting.value_only(observation, self.hidden.clone());
        self.last_value = scalar(&value);
        self.last_value
    }

    // =====================================================================
    // Reporting
    // =====================================================================

    pub fn status_block(&mut self) {
        let wall = self.env_seconds + self.update_seconds;
        let env_share = if wall > 0.0 {
            100.0 * self.env_seconds / wall
        } else {
            100.0
        };
        let steps = self.status_window_steps.max(1);
        let elapsed = self.started.elapsed().as_secs_f64();
        let rate = self.total_steps as f64 / elapsed.max(1e-6);
        println!();
        println!(
            "[Step {}] {:.1} decisions/s, {:.0}% of wall clock in the game ({:.0}% \
             training), {} episode(s)",
            self.total_steps,
            rate,
            env_share,
            100.0 - env_share,
            self.episodes
        );
        println!("          {}", self.trainer.status_line());
        if self.learn_offloaded {
            let in_flight = self
                .learn_worker
                .as_ref()
                .map(|worker| worker.in_flight())
                .unwrap_or(0);
            let behind = if self.learn_skipped == 0 {
                String::new()
            } else {
                format!(
                    ", {} rollout(s) dropped because training was still busy",
                    self.learn_skipped
                )
            };
            println!(
                "          learning on its own thread: {} update(s) applied, {} in flight{behind}",
                self.learn_updates, in_flight
            );
            if self.learn_disabled_after_nan {
                println!(
                    "          [Health] The last update produced non-finite weights and was \
                     discarded; training continues from the last good set."
                );
            }
            if self.learn_superseded > 0 {
                // Only one update is ever outstanding, so this cannot happen
                // through the session's own accounting - which is exactly why it
                // is worth saying when it does.
                println!(
                    "          [Health] {} update(s) finished with a newer one already \
                     waiting. Training is running further ahead of collection than the \
                     loop expects.",
                    self.learn_superseded
                );
            }
        }
        println!(
            "          {}  | resets {} (last: {}), {} new states/1000 steps, {}% of \
             steps pressed something",
            self.reward.summary(),
            self.reward.detector.resets,
            if self.reward.detector.last_reason.is_empty() {
                "-".to_string()
            } else {
                self.reward.detector.last_reason.clone()
            },
            self.status_novel * 1000 / steps,
            self.status_engaged * 100 / steps
        );
        println!("          {}", self.mouse_speed_line());
        if let Some(stats) = self.env.input_stats() {
            println!(
                "          input: {} - {} event(s) delivered to {}, {} decision(s) \
                 skipped while unfocused",
                if stats.focused { "focused" } else { "NOT FOCUSED" },
                stats.delivered,
                stats.window,
                stats.skipped_unfocused
            );
        }

        let total: u64 = self.action_counts.values().sum();
        if total > 0 {
            let mut top: Vec<(String, u64)> = self
                .action_counts
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            top.sort_by(|a, b| b.1.cmp(&a.1));
            top.truncate(4);
            let share = 100.0 * top[0].1 as f32 / total as f32;
            let rendered = top
                .iter()
                .map(|(name, count)| {
                    format!("{name} {:.0}%", 100.0 * *count as f32 / total as f32)
                })
                .collect::<Vec<_>>()
                .join(", ");
            println!("          decisions: {rendered}");
            if share > 90.0 {
                println!(
                    "          [Health] One decision type owns almost every step. Raise \
                     entropy_coef to push the policy apart, or lower w_idle if it is 'no keys'."
                );
            }
        }
        self.action_counts.clear();

        for message in self.health.end_window(
            steps,
            self.status_engaged,
            self.trainer.last_update_seconds,
        ) {
            println!("          {message}");
        }
        if self.trainer.last_update_seconds > self.cfg.slow_update_seconds as f64 {
            let advice = if self.learn_offloaded {
                "It no longer stalls collection - it is on its own thread - but it is \
                 still competing for the same CPU, so a long update costs frames instead \
                 of freezing them. Lower rollout_steps, minibatch_size or seq_len, or \
                 raise update_seconds_budget if the machine can spare it."
            } else {
                "Lower rollout_steps, minibatch_size or seq_len, or raise \
                 update_seconds_budget if the machine can spare it."
            };
            println!(
                "          [Health] The last update took {:.1}s. {advice}",
                self.trainer.last_update_seconds
            );
        }
        self.health.start_window();
        self.status_window_steps = 0;
        self.status_engaged = 0;
        self.status_novel = 0;
    }

    pub fn hold_keys(&self) -> Vec<u32> {
        self.held_vks.clone()
    }

    pub fn release_held(&mut self) {
        self.env.release_all();
        self.held_vks.clear();
    }

    pub fn close(&mut self) {
        self.release_held();
        self.env.close();
    }

    pub fn suspend(&mut self) {
        self.env.suspend();
    }

    pub fn resume(&mut self) {
        self.env.resume();
    }

    pub fn focus(&mut self) {
        if self.env.acquire_focus() == Some(false) {
            println!(
                "[Control] The game window could not be brought forward (Windows allows \
                 that only from the foreground app). Click the game - input starts on its \
                 own once the game has focus."
            );
        }
    }
}

fn values<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .expect("activations are f32")
}

fn scalar<B: Backend>(tensor: &Tensor<B, 1>) -> f32 {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .ok()
        .and_then(|values| values.first().copied())
        .unwrap_or(0.0)
}

/// Draw a key-set from the capped distribution.
fn sample_held<B: Backend>(
    model: &ActorCritic<B>,
    logits: Tensor<B, 2>,
    rng: &mut impl rand::RngExt,
) -> Vec<f32> {
    let scores = values(model.held_logits_of_masks(logits));
    let index = sample_from_scores(&scores, rng);
    model.held.masks[index].clone()
}

/// Sample one index from unnormalised scores.
fn sample_from_scores(scores: &[f32], rng: &mut impl rand::RngExt) -> usize {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    let draw: f32 = rng.random::<f32>() * total;
    let mut cumulative = 0.0f32;
    for (index, value) in exps.iter().enumerate() {
        cumulative += value;
        if draw < cumulative {
            return index;
        }
    }
    exps.len().saturating_sub(1)
}

/// Draw one categorical index from a `(1, classes)` logit row.
fn sample_categorical<B: Backend>(logits: &Tensor<B, 2>, rng: &mut impl rand::RngExt) -> usize {
    sample_from_scores(&values(logits.clone()), rng)
}

/// Log-probability of one category of a `(1, classes)` logit row.
fn categorical_log_prob<B: Backend>(logits: &Tensor<B, 2>, index: usize) -> f32 {
    let scores = values(logits.clone());
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    (exps.get(index).copied().unwrap_or(0.0) / total)
        .max(1e-45)
        .ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Flex};

    type TestBackend = Autodiff<Flex>;

    /// A session on the synthetic game, with a rollout short enough to fill in a
    /// test's worth of steps.
    fn synthetic_session(rollout_steps: usize) -> GameSession<TestBackend> {
        let device = Default::default();
        let cfg = crate::synth::synthetic_train_config(48, rollout_steps, 7);
        let env = crate::synth::SyntheticEnv::new(&cfg, 60, 300);
        GameSession::<TestBackend>::new(
            &cfg,
            Box::new(env),
            crate::synth::synthetic_action_space(&cfg),
            None,
            device,
        )
        .expect("a synthetic session")
    }

    /// One rollout's worth of real transitions, so the update has data that is
    /// not degenerate.
    fn fill_rollout(session: &mut GameSession<TestBackend>) {
        session.reset(Some(1), false);
        while session.buffer.len() < session.cfg.rollout_steps {
            let action = session.decide();
            let (_reward, done, _info) = session.step(&action);
            if done {
                session.reset(None, true);
            }
        }
    }

    /// Poll the session until the in-flight update has been applied. Generous,
    /// because a cold CPU backend's first update is its slowest.
    fn wait_for_one_update(session: &mut GameSession<TestBackend>) {
        let deadline = Instant::now() + std::time::Duration::from_secs(300);
        while session.learn_updates == 0 {
            session.apply_trained_weights();
            if session.learn_updates > 0 {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "the learning worker produced nothing in five minutes"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// The whole point of the worker: `learn()` blocks the calling thread, and
    /// `submit_learn()` must not.
    #[test]
    fn submitting_an_update_does_not_train_on_the_calling_thread() {
        let mut session = synthetic_session(64);
        fill_rollout(&mut session);
        session.bootstrapped_value();
        session.start_learning_worker();
        assert!(session.learning_is_offloaded());

        let before = session.policy.net.clone();
        let started = Instant::now();
        assert!(session.submit_learn(), "the rollout should be accepted");
        let submission = started.elapsed();

        // Training starts from the weights the rollout was collected with, so
        // nothing about the session's own policy may change here.
        assert_eq!(
            session.policy.net.bos.val().into_data().to_vec::<f32>().unwrap(),
            before.bos.val().into_data().to_vec::<f32>().unwrap(),
            "the collection weights must not move while the update is in flight"
        );
        // The synchronous path on the same data takes far longer than a channel
        // send; this is the property that removes the pause, so it is asserted
        // rather than assumed.
        assert!(
            submission < std::time::Duration::from_millis(200),
            "submitting must not train on the collecting thread (took {submission:?})"
        );
        // And the buffer is free again, so collection carries straight on.
        assert_eq!(session.buffer.len(), 0);
        assert_eq!(session.buffer.capacity, 64);

        // The worker does eventually produce weights, and the session takes them.
        wait_for_one_update(&mut session);
        assert_eq!(session.learn_updates, 1, "the update should have landed");
        assert!(!session.learning_in_flight());
    }

    /// A fingerprint of every policy parameter.
    ///
    /// One element is not enough to tell whether an update happened: a single
    /// gradient step moves a weight by ~1e-7, which is real but below the
    /// precision a lone `f32` shows once the optimiser has rounded it. Summing
    /// the whole tree makes that movement visible while staying comparable
    /// exactly between two runs that did identical arithmetic.
    fn weight_fingerprint(session: &GameSession<TestBackend>) -> f64 {
        session
            .policy
            .to_weights()
            .iter()
            .flat_map(|(_, _, values)| values.iter())
            .map(|value| *value as f64)
            .sum()
    }

    /// The worker must not quietly do nothing, or quietly do something else.
    #[test]
    fn the_worker_updates_the_same_weights_the_inline_path_would() {
        let inline = {
            let mut session = synthetic_session(64);
            fill_rollout(&mut session);
            session.bootstrapped_value();
            let before = weight_fingerprint(&session);
            session.learn();
            let after = weight_fingerprint(&session);
            assert!(
                (after - before).abs() > 0.0,
                "an inline update must move the weights"
            );
            after
        };

        let offloaded = {
            let mut session = synthetic_session(64);
            fill_rollout(&mut session);
            session.bootstrapped_value();
            session.start_learning_worker();
            assert!(session.submit_learn());
            wait_for_one_update(&mut session);
            weight_fingerprint(&session)
        };

        assert_eq!(
            inline, offloaded,
            "a rollout collected and trained under the same seed must produce the \
             same weights whether it is updated on the control thread or a worker"
        );
    }

    /// The loop the real run uses, driven for several rollouts: the collector
    /// must keep playing the whole time, and the weights it plays with must stay
    /// finite and keep moving.
    #[test]
    fn collection_continues_while_updates_are_in_flight() {
        let mut session = synthetic_session(32);
        session.start_learning_worker();

        let mut submissions = 0usize;
        let mut applied = 0usize;
        let mut dropped = 0usize;
        let mut previous = weight_fingerprint(&session);
        for round in 0..6 {
            fill_rollout(&mut session);
            session.bootstrapped_value();
            if session.submit_learn() {
                submissions += 1;
            } else {
                // The worker was still busy, so the rollout was let go rather
                // than queued: the queue is bounded by design.
                dropped += 1;
            }

            // The point of the whole change: frames keep being collected while an
            // update runs on the other thread.
            let steps_before = session.total_steps;
            for _ in 0..8 {
                let action = session.decide();
                let (_reward, done, _info) = session.step(&action);
                if done {
                    session.reset(None, true);
                }
            }
            assert!(
                session.total_steps >= steps_before + 8,
                "collection must not stall while an update is in flight"
            );

            // Let the worker land, then apply it at a rollout boundary, exactly
            // as `run()` does.
            let deadline = Instant::now() + std::time::Duration::from_secs(300);
            while session.learning_in_flight() {
                assert!(Instant::now() < deadline, "an update never finished");
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            session.apply_trained_weights();

            let bad = Trainer::<TestBackend>::nonfinite_parameters(&session.policy);
            assert!(bad.is_empty(), "round {round} produced non-finite weights: {bad:?}");

            if session.learn_updates as usize > applied {
                let now = weight_fingerprint(&session);
                assert!(now.is_finite(), "round {round}: fingerprint {now} is not finite");
                assert!(
                    (now - previous).abs() > 0.0,
                    "round {round}: applying an update must move the weights"
                );
                previous = now;
                applied = session.learn_updates as usize;
            }
        }
        assert!(submissions > 0, "no rollout was ever submitted");
        assert!(applied > 0, "the worker never applied an update");
        assert!(
            applied <= submissions,
            "the worker applied more updates ({applied}) than were submitted ({submissions})"
        );
        assert_eq!(
            session.learn_skipped as usize, dropped,
            "every rollout refused by the bounded queue must be reported as dropped"
        );
    }

    /// The parts of the fix that stay on the control loop have to be cheap, or
    /// the pause just moves rather than disappears.
    ///
    /// Two things still run there: the policy clone that gives the worker its
    /// starting weights (per update) and `to_weights`, which copies the
    /// parameters out for a checkpoint (per checkpoint). Both are on the loop, so
    /// both are measured rather than assumed - the assumption being that Burn's
    /// parameters are behind `Arc` handles, which a few thousand `Arc` increments
    /// would confirm and a deep copy would refute.
    #[test]
    fn the_work_left_on_the_control_loop_is_cheap() {
        let session = synthetic_session(64);

        let started = Instant::now();
        let clone = session.policy.clone();
        let clone_seconds = started.elapsed().as_secs_f64();
        // Keep it alive so the clone is not optimised away.
        assert!(clone.net.bos.val().dims()[0] == 1);

        // Warm the allocator, then time the real thing.
        let _ = session.policy.to_weights();
        let started = Instant::now();
        let tensors = session.policy.to_weights();
        let weights_seconds = started.elapsed().as_secs_f64();

        println!(
            "  policy clone {:.3}ms, to_weights {:.3}ms ({} tensors)",
            clone_seconds * 1000.0,
            weights_seconds * 1000.0,
            tensors.len()
        );
        assert!(
            clone_seconds < 0.010,
            "the worker's starting weights must be a handle copy, not a weight copy \
             (took {:.1}ms)",
            clone_seconds * 1000.0
        );
        assert!(
            weights_seconds < 0.100,
            "copying the weights for a checkpoint must not stall the loop (took {:.1}ms)",
            weights_seconds * 1000.0
        );
    }
}
