//! The real game, wrapped as an environment. Ported from `botcore/game.py`.
//!
//! This is the only module that knows the bot is driving a window, and it knows
//! nothing about *which* game: it grabs the client area as pixels, applies a
//! factored action (keys held, mouse turn, one tap or click), and reports what
//! changed. The learning layer above it never sees a window handle.
//!
//! Three deliberate choices, all inherited:
//!
//! * **The observation is grey frames plus a difference channel.** No colour, no
//!   audio: colour triples the model's input cost for very little control
//!   information.
//! * **Held keys are level-triggered, not edge-triggered.** The bot says which
//!   buttons should be down and this module presses what is new and releases what
//!   is gone, which is what lets it walk continuously instead of tapping a key
//!   every 33 ms.
//! * **The mouse delta arrives already decided.** Which way to swing and how far
//!   are the bot's decisions; this module only sends the resulting pixels.

use std::collections::BTreeSet;
use std::time::Instant;

use crate::capture::{FrameStack, Pacer};
use crate::config::Config;
use crate::keys::ActionSpace;
use crate::platform::{self, Handle};
use crate::session::{EnvAction, EnvStep, Environment, InputStats};
use crate::vision::{Observation, gray_signature};

/// What the bot is driving, for the status line.
#[derive(Debug, Clone, Default)]
struct WindowLabel {
    window: String,
    title: String,
    process: String,
    non_game: bool,
}

/// A live game window as an environment.
///
/// `step` takes the action [`crate::session::GameSession`] built and returns the
/// next observation. `truncated` is true only when the window has gone away: an
/// in-game reset is the reward module's business, not this one's.
pub struct RealGameEnv {
    cfg: Config,
    handle: Handle,
    dry_run: bool,
    grabber: platform::FrameGrabber,
    stack: FrameStack,
    pacer: Pacer,
    input: platform::Injector,
    label: WindowLabel,
    shape: (usize, usize, usize),
    gray_channels: usize,
    held: BTreeSet<u32>,
    steps: u64,
    dead_captures: u64,
    pub window_gone: bool,
    pub paused_by_user: bool,
    last_capture_ms: f64,
    total_capture_seconds: f64,
}

impl RealGameEnv {
    /// Build the environment, refusing outright to drive a window that must not
    /// be driven.
    pub fn new(
        cfg: &Config,
        handle: Handle,
        action_space: &ActionSpace,
        dry_run: bool,
    ) -> anyhow::Result<Self> {
        let _ = action_space;
        // The one guard that stops this program typing its actions into a shell
        // prompt. Window selection refuses the bot's own terminal before it gets
        // here; this is the backstop for every other way in - a stale handle in a
        // config file, a script, a checkpoint.
        if !dry_run {
            let window = platform::current::window_entry(handle);
            let risk = platform::window_risk(window.as_ref());
            if !risk.is_empty() {
                anyhow::bail!(
                    "refusing to drive {}: {risk}",
                    platform::describe_window(window.as_ref())
                );
            }
        }
        let stack = FrameStack::new(cfg.frame_size, cfg.frame_stack);
        let shape = (stack.channels, cfg.frame_size, cfg.frame_size);
        let mut env = Self {
            cfg: cfg.clone(),
            handle,
            dry_run,
            grabber: platform::FrameGrabber::new(handle),
            stack,
            pacer: Pacer::new(cfg.target_fps, cfg.capture_delay, cfg.action_repeat),
            input: platform::Injector::new(handle, cfg.focus_policy, cfg.clip_cursor),
            label: WindowLabel::default(),
            shape,
            gray_channels: cfg.frame_stack,
            held: BTreeSet::new(),
            steps: 0,
            dead_captures: 0,
            window_gone: false,
            paused_by_user: false,
            last_capture_ms: 0.0,
            total_capture_seconds: 0.0,
        };
        env.label = env.describe();
        Ok(env)
    }

    /// Title/process of the driven window.
    ///
    /// Read at startup and refreshed when the performance line prints, not on
    /// every step: the handle cannot change mid-run, the title can (a game
    /// renames its window per level), and resolving a process name is far too
    /// much work to do twenty times a second.
    fn describe(&self) -> WindowLabel {
        let pid = platform::current::get_window_pid(self.handle);
        let process = platform::current::get_process_name(pid);
        let window = platform::current::window_entry(self.handle);
        WindowLabel {
            window: platform::describe_window(window.as_ref()),
            title: window.map(|w| w.title).unwrap_or_default(),
            non_game: platform::current::is_non_game_process(&process.to_lowercase()),
            process,
        }
    }

    /// Grab one frame and turn it into an observation.
    ///
    /// A failed grab yields a black frame rather than an error: a grab that
    /// returned nothing is not a game event, and the loop above decides what to
    /// do about it (the session's health monitor says so if it keeps happening).
    fn grab(&mut self) -> Observation {
        let started = Instant::now();
        let frame = self.grabber.grab();
        let elapsed = started.elapsed().as_secs_f64();
        self.last_capture_ms = elapsed * 1000.0;
        self.total_capture_seconds += elapsed;
        match frame {
            Some(frame) => {
                self.dead_captures = 0;
                self.stack.push(&frame).clone()
            }
            None => {
                self.dead_captures += 1;
                Observation::zeros(self.shape.0, self.shape.1, self.shape.2)
            }
        }
    }

    /// Level-triggered: press what is new, release what is gone.
    pub fn set_held(&mut self, vks: &[u32]) {
        let target: BTreeSet<u32> = vks.iter().copied().filter(|vk| *vk != 0).collect();
        if target == self.held {
            return;
        }
        if !self.dry_run {
            self.input.begin_action();
            let released: Vec<u32> = self.held.difference(&target).copied().collect();
            for vk in released {
                self.input.release_vk(vk);
            }
            let pressed: Vec<u32> = target.difference(&self.held).copied().collect();
            for vk in pressed {
                self.input.press_vk(vk);
            }
        }
        self.held = target;
    }

    pub fn mean_capture_ms(&self) -> f64 {
        1000.0 * self.total_capture_seconds / self.steps.max(1) as f64
    }

    pub fn window_label(&self) -> String {
        if self.label.window.is_empty() {
            platform::describe_window(platform::current::window_entry(self.handle).as_ref())
        } else {
            self.label.window.clone()
        }
    }

    pub fn is_dry_run(&self) -> bool {
        self.dry_run
    }

    /// The inner injector, so a caller can inspect the delivery counters.
    pub fn injector(&self) -> &platform::Injector {
        &self.input
    }
}

impl Environment for RealGameEnv {
    fn observation_shape(&self) -> (usize, usize, usize) {
        self.shape
    }

    fn gray_channels(&self) -> usize {
        self.gray_channels
    }

    fn reset(&mut self, _seed: Option<u64>) -> (Observation, Vec<f32>) {
        self.release_all();
        self.stack.clear();
        self.pacer.reset();
        self.steps = 0;
        self.paused_by_user = false;
        let mut observation = None;
        // Fill the stack before the first decision, so the model does not start
        // from a screen it has never seen changing.
        for _ in 0..self.cfg.frame_stack {
            let grabbed = self.grab();
            observation = Some(grabbed.clone());
            if self.dead_captures > 0 {
                break;
            }
        }
        let observation = observation.unwrap_or_else(|| {
            Observation::zeros(self.shape.0, self.shape.1, self.shape.2)
        });
        let plane = observation.mean_of_channels(self.gray_channels);
        let signature = gray_signature(&plane, observation.height, observation.width, 32);
        (observation, signature)
    }

    fn step(&mut self, action: &EnvAction) -> (Observation, EnvStep) {
        let focused = self.input.focused();
        if !self.dry_run && !focused {
            // Nothing is injected into a window that is not in front: the keys
            // would land in whatever is - for a bot launched from a terminal,
            // the terminal, which then keeps stealing the keyboard back.
            self.release_all();
        } else if self.paused_by_user {
            self.release_all();
        } else {
            self.set_held(&action.held_vks);
            if !self.dry_run {
                self.input.begin_action();
                if action.turn != (0, 0) {
                    self.input.mouse_move(action.turn.0, action.turn.1);
                }
                if action.tap_vk != 0 {
                    self.input.tap_vk(action.tap_vk, self.cfg.tap_seconds);
                }
                self.input.end_action();
            }
        }

        self.pacer.tick();
        let observation = self.grab();
        self.steps += 1;
        let gone = !platform::current::is_window(self.handle);
        if gone {
            self.window_gone = true;
        }
        if let Some(line) = self.pacer.report(15.0) {
            println!("{line}");
            self.label = self.describe();
            if !self.dry_run {
                // Printed next to the rate on purpose: the rate alone says the
                // loop is alive, and this line says whether it is playing.
                println!("{}", self.input.status_line());
            }
        }
        (
            observation,
            EnvStep {
                truncated: gone,
                died: false,
                progress: 0,
            },
        )
    }

    fn release_all(&mut self) {
        if !self.dry_run {
            self.input.begin_action();
            let held: Vec<u32> = self.held.iter().copied().collect();
            for vk in held {
                self.input.release_vk(vk);
            }
            // Closed, not left open: `end_action` is where the cursor a mouse
            // move claimed is let go, and this path runs when the bot is
            // suspending or stopping - exactly when the user needs their mouse
            // back.
            self.input.end_action();
        }
        self.held.clear();
    }

    fn close(&mut self) {
        if !self.held.is_empty() {
            // One last attempt at the foreground window, so the key-ups below
            // actually reach the game rather than a window that took over.
            self.input.acquire_focus(true);
        }
        self.release_all();
    }

    fn acquire_focus(&mut self) -> Option<bool> {
        if self.dry_run || self.input.policy == crate::config::FocusPolicy::Never {
            // A dry run touches nothing, and "never" means the user brings the
            // game forward: neither is a failure to report.
            return Some(true);
        }
        Some(self.input.acquire_focus(true))
    }

    fn suspend(&mut self) {
        self.release_all();
        self.input.suspended = true;
    }

    fn resume(&mut self) {
        self.input.suspended = false;
    }

    fn input_stats(&self) -> Option<InputStats> {
        Some(InputStats {
            focused: self.input.focused(),
            delivered: self.input.events_sent,
            skipped_unfocused: self.input.skipped_unfocused,
            window: self.window_label(),
            non_game: self.label.non_game,
            title: self.label.title.clone(),
            process: self.label.process.clone(),
        })
    }

    fn current_observation(&mut self) -> Option<Observation> {
        // The session keeps the last observation it saw, which is what the
        // Python falls back to as well.
        None
    }
}
