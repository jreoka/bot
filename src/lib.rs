//! bot1, in Rust: a game-agnostic bot that learns to play from pixels.
//!
//! The tree mirrors the Python's `botcore/` one module at a time, so the two can
//! be read side by side while the port is in progress:
//!
//! ```text
//! config.rs    every tunable, with the reasoning next to it
//! model.rs     the SwiGLU + RoPE transformer: every head, and the next-frame
//!              prediction the reward reads
//! weights.rs   reading the Python's weights, which is what the parity test
//!              needs to prove this computes the same thing
//! keys.rs      the calibrated whitelist and the factored action space
//! vision.rs    the observation, and the coarse fingerprint the reward keys on
//! replay.rs    fixed-size rollout storage and sequence minibatching
//! novelty.rs   the intrinsic reward and the reset detector (owns no net)
//! trainer.rs   PPO plus the curiosity head's own optimiser
//! session.rs   the loop, the health checks, and the environment trait
//! synth.rs     the synthetic game used by --selftest
//! capture.rs   frames into observations, and the control-rate pacer
//! diagnostics.rs  the checks: what the bot sees, what it may press, how fast
//! game.rs      the real window as an environment: the only module that knows
//!              there is a window at all
//! control.rs   Ctrl+C, the start/pause/save/quit keys, the checkpoint timer
//! checkpoint.rs  what a run writes so that stopping is not losing
//! platform/    one policy, three mechanisms: which window to drive, and how to
//!              capture it and press keys into it (Win32 / X11 / CoreGraphics)
//! ```
//!
//! Still to come: frame capture and input injection, hotkeys, calibration,
//! diagnostics, and the session's checkpointing and CLI.

pub mod capture;
pub mod checkpoint;
pub mod config;
pub mod control;
pub mod diagnostics;
pub mod game;
pub mod keys;
pub mod model;
pub mod novelty;
pub mod platform;
pub mod replay;
pub mod session;
pub mod synth;
pub mod trainer;
pub mod vision;
pub mod weights;

pub use config::Config;
pub use model::{Action, ActorCritic};
pub use session::{Environment, GameSession};
