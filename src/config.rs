//! Every tunable, ported from `botcore/config.py`.
//!
//! Field names and defaults match the Python exactly, and so do the
//! `BOT_<FIELD>` environment overrides, so the same configuration file and the
//! same environment drive either build. Two rules from the Python are kept
//! because they are what makes "runs for hours without stalling" a property of
//! the configuration rather than a hope:
//!
//! * anything that grows without bound gets a capacity here;
//! * anything that loops gets a time or iteration budget here.

use serde::{Deserialize, Serialize};

/// The shape of the checkpoint format. Bumped whenever a saved run stops being
/// loadable.
pub const CHECKPOINT_VERSION: u32 = 3;
/// Bumped whenever the reward function changes meaning.
pub const REWARD_VERSION: u32 = 3;
/// Bumped when the model architecture changes in a way that invalidates weights.
pub const ARCH_NAME: &str = "swiglu-rope-transformer-v1";

/// How hard the bot may fight for the foreground window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FocusPolicy {
    /// Bring the game forward when a run starts or resumes, then leave the
    /// desktop alone.
    Once,
    /// Re-assert focus on every action.
    Always,
    /// Never touch focus; the user brings the game forward.
    Never,
}

/// Process priority class for the bot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    BelowNormal,
    Normal,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    // =====================================================================
    // Window / capture
    // =====================================================================
    /// `None` = auto-detect a likely game window (or prompt). `Some` = that hwnd.
    pub window_hwnd: Option<usize>,
    /// True = silently take the largest likely game window; false = ask.
    pub prefer_game_window: bool,

    // =====================================================================
    // Observation
    // =====================================================================
    pub frame_size: usize,
    pub frame_stack: usize,
    /// How many control steps one decision is repeated for.
    pub action_repeat: usize,

    // =====================================================================
    // Control rate
    // =====================================================================
    pub target_fps: f32,
    pub capture_delay: f32,
    pub tap_seconds: f32,

    // =====================================================================
    // Model
    // =====================================================================
    pub embed_dim: usize,
    pub mem_tokens: usize,
    pub patch_size: usize,
    pub patch_width: usize,
    pub transformer_layers: usize,
    pub ffn_hidden: usize,
    pub attention_heads: usize,
    pub attention_dropout: f32,
    pub seq_len: usize,

    // =====================================================================
    // PPO
    // =====================================================================
    pub rollout_steps: usize,
    pub minibatch_size: usize,
    pub epochs_per_update: usize,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip_eps: f32,
    pub value_coef: f32,
    pub entropy_coef: f32,
    pub adam_lr: f32,
    pub adam_eps: f32,
    pub grad_clip: f32,
    pub reward_scale: f32,

    // =====================================================================
    // Learning signal
    // =====================================================================
    pub w_episodic: f32,
    pub w_depth: f32,
    pub w_depth_progress: f32,
    pub novelty_decay: f32,
    pub novelty_survival_steps: f32,
    pub w_idle: f32,
    pub idle_ramp_steps: f32,
    pub idle_ramp_max: f32,
    pub novelty_bins: f32,

    // =====================================================================
    // Reward normalisation
    // =====================================================================
    pub scale_floor: f32,
    pub reward_clip: f32,

    // =====================================================================
    // Episode / reset detection
    // =====================================================================
    pub reset_jump: f32,
    pub reset_revisit: f32,
    pub reset_cooldown: usize,

    // =====================================================================
    // Liveness / stall guard
    // =====================================================================
    pub frozen_warn_seconds: f32,
    pub update_seconds_budget: f32,
    pub status_every: usize,
    pub slow_update_seconds: f32,

    // =====================================================================
    // Novelty memory caps (all fixed-size; nothing here grows forever)
    // =====================================================================
    pub episodic_capacity: usize,
    pub global_capacity: usize,

    // =====================================================================
    // The transformer's own curiosity head
    // =====================================================================
    pub transition_hidden: usize,
    pub transition_lr: f32,
    pub w_novelty: f32,
    pub w_novelty_cap: f32,
    pub error_decay: f32,

    // =====================================================================
    // Input budget
    // =====================================================================
    pub max_held_keys: usize,

    /// The base step the speed multipliers apply to, in pixels.
    pub mouse_turn_pixels: i32,
    pub mouse_turn_start: i32,
    pub mouse_turn_min: i32,
    pub mouse_turn_max: i32,
    pub speed_levels: Vec<f32>,
    pub mouse_adapt_rate: f32,

    /// Virtual keys the recorder refuses to learn and the bot refuses to press.
    pub blocked_vks: Vec<u32>,

    // ---- keymap / calibration -------------------------------------------
    pub keymap_path: String,
    pub calibrate_key: String,
    pub calibrate_min_hold: f32,

    // =====================================================================
    // Checkpointing
    // =====================================================================
    pub checkpoint_dir: String,
    pub checkpoint_interval_sec: f32,
    pub checkpoint_keep: usize,
    pub resume: bool,
    pub final_model_path: String,

    // =====================================================================
    // Control / housekeeping
    // =====================================================================
    pub hotkey_pause: String,
    pub hotkey_save: String,
    pub hotkey_quit: String,
    pub enable_hotkeys: bool,
    pub wait_for_start: bool,
    pub focus_policy: FocusPolicy,
    /// Threads the tensor backend may use. 1 in the Python build, for the reason
    /// recorded in `config.py`: this model is small enough that the per-op
    /// thread hand-off costs more than the parallelism saves. Kept as its own
    /// name so `BOT_TORCH_THREADS` still means something.
    pub torch_threads: usize,
    pub priority: Priority,
    pub preview: bool,
    pub show_model: bool,
    pub log_json: Option<String>,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            window_hwnd: None,
            prefer_game_window: true,

            frame_size: 96,
            frame_stack: 4,
            action_repeat: 2,

            target_fps: 20.0,
            capture_delay: 0.0,
            tap_seconds: 0.05,

            embed_dim: 96,
            mem_tokens: 16,
            patch_size: 16,
            patch_width: 64,
            transformer_layers: 2,
            ffn_hidden: 192,
            attention_heads: 4,
            attention_dropout: 0.0,
            seq_len: 16,

            rollout_steps: 1024,
            minibatch_size: 256,
            epochs_per_update: 4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_eps: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            adam_lr: 3e-4,
            adam_eps: 1e-5,
            grad_clip: 0.5,
            reward_scale: 1.0,

            w_episodic: 3.0,
            w_depth: 1.0,
            w_depth_progress: 0.5,
            novelty_decay: 0.5,
            novelty_survival_steps: 12.0,
            w_idle: 0.02,
            idle_ramp_steps: 25.0,
            idle_ramp_max: 4.0,
            novelty_bins: 32.0,

            scale_floor: 1e-3,
            reward_clip: 10.0,

            reset_jump: 0.35,
            reset_revisit: 0.02,
            reset_cooldown: 8,

            frozen_warn_seconds: 120.0,
            update_seconds_budget: 6.0,
            status_every: 2000,
            slow_update_seconds: 3.0,

            episodic_capacity: 20000,
            global_capacity: 200000,

            transition_hidden: 192,
            transition_lr: 3e-4,
            w_novelty: 1.0,
            w_novelty_cap: 2.0,
            error_decay: 0.99,

            max_held_keys: 4,

            mouse_turn_pixels: 0,
            mouse_turn_start: 20,
            mouse_turn_min: 4,
            mouse_turn_max: 120,
            speed_levels: vec![0.25, 0.5, 1.0, 2.0, 4.0],
            mouse_adapt_rate: 0.08,

            blocked_vks: vec![0x70, 0x72, 0x5B, 0x5C, 0x5D, 0x12, 0x09],

            keymap_path: "keymap.json".to_string(),
            calibrate_key: "f8".to_string(),
            calibrate_min_hold: 0.12,

            checkpoint_dir: "checkpoints".to_string(),
            checkpoint_interval_sec: 300.0,
            checkpoint_keep: 3,
            resume: true,
            final_model_path: "gamebot.pt".to_string(),

            hotkey_pause: "f8".to_string(),
            hotkey_save: "f9".to_string(),
            hotkey_quit: "f10".to_string(),
            enable_hotkeys: true,
            wait_for_start: true,
            focus_policy: FocusPolicy::Once,
            torch_threads: 1,
            priority: Priority::BelowNormal,
            preview: false,
            show_model: false,
            log_json: None,
            seed: 0,
        }
    }
}

impl Config {
    /// Input channels the model sees: the grey stack plus one difference.
    pub fn channels(&self) -> usize {
        self.frame_stack + 1
    }

    /// Patch tokens one frame becomes, given `frame_size` and `patch_size`.
    pub fn frame_tokens(&self) -> usize {
        let step = self.patch_size.max(2);
        let grid = self.frame_size.div_ceil(step).max(1);
        grid * grid
    }

    /// Minibatches per epoch, given the rollout and minibatch size.
    pub fn update_minibatches(&self) -> usize {
        let mb = self.minibatch_size.max(self.seq_len);
        (self.rollout_steps / mb).max(1)
    }

    /// Fail loudly here rather than three hours into a run.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.frame_size >= 40,
            "frame_size {} is too small; the patch embedding needs at least 40 \
             (96 is the default)",
            self.frame_size
        );
        anyhow::ensure!(self.frame_stack >= 1, "frame_stack must be at least 1");
        anyhow::ensure!(self.action_repeat >= 1, "action_repeat must be at least 1");
        anyhow::ensure!(self.seq_len >= 2, "seq_len must be at least 2");
        anyhow::ensure!(
            self.rollout_steps >= self.seq_len,
            "rollout_steps {} must be at least seq_len {}",
            self.rollout_steps,
            self.seq_len
        );
        anyhow::ensure!(self.max_held_keys >= 1, "max_held_keys must be at least 1");
        anyhow::ensure!(self.embed_dim >= 8, "embed_dim must be at least 8");
        anyhow::ensure!(
            self.embed_dim % 4 == 0,
            "embed_dim must be divisible by 4"
        );
        anyhow::ensure!(self.mem_tokens >= 1, "mem_tokens must be at least 1");
        anyhow::ensure!(self.patch_size >= 2, "patch_size must be at least 2");
        anyhow::ensure!(
            self.transformer_layers >= 1,
            "transformer_layers must be at least 1"
        );
        anyhow::ensure!(
            self.ffn_hidden >= self.embed_dim,
            "ffn_hidden must be at least embed_dim"
        );
        anyhow::ensure!(self.attention_heads >= 1, "attention_heads must be at least 1");
        anyhow::ensure!(
            self.embed_dim % self.attention_heads == 0,
            "embed_dim {} must be divisible by attention_heads {}",
            self.embed_dim,
            self.attention_heads
        );
        anyhow::ensure!(
            (self.embed_dim / self.attention_heads) % 2 == 0,
            "embed_dim / attention_heads must be even for rotary position \
             embedding (got {})",
            self.embed_dim / self.attention_heads
        );
        anyhow::ensure!(
            !self.speed_levels.is_empty(),
            "speed_levels must list at least one multiplier"
        );
        anyhow::ensure!(
            self.speed_levels.iter().all(|s| *s > 0.0),
            "speed_levels must all be positive"
        );
        anyhow::ensure!(
            self.mouse_turn_max >= self.mouse_turn_min,
            "mouse_turn_max must be >= mouse_turn_min"
        );
        anyhow::ensure!(
            self.mouse_turn_min <= self.mouse_turn_start
                && self.mouse_turn_start <= self.mouse_turn_max,
            "mouse_turn_start {} must lie between mouse_turn_min {} and \
             mouse_turn_max {}",
            self.mouse_turn_start,
            self.mouse_turn_min,
            self.mouse_turn_max
        );
        Ok(())
    }

    /// Apply `BOT_<FIELD>` environment overrides, for quick experiments.
    ///
    /// Only the fields the Python build exposes this way are honoured; the
    /// coercion follows the field's own type, as `_env` does there.
    pub fn env_overrides(mut self) -> Self {
        macro_rules! env_num {
            ($field:ident, $ty:ty) => {
                if let Ok(raw) = std::env::var(concat!("BOT_", stringify!($field)).to_uppercase())
                    && let Ok(parsed) = raw.trim().parse::<$ty>()
                {
                    self.$field = parsed;
                }
            };
        }
        macro_rules! env_bool {
            ($field:ident) => {
                if let Ok(raw) = std::env::var(concat!("BOT_", stringify!($field)).to_uppercase()) {
                    self.$field = matches!(
                        raw.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    );
                }
            };
        }
        macro_rules! env_string {
            ($field:ident) => {
                if let Ok(raw) = std::env::var(concat!("BOT_", stringify!($field)).to_uppercase())
                    && !raw.is_empty()
                {
                    self.$field = raw;
                }
            };
        }

        env_num!(frame_size, usize);
        env_num!(frame_stack, usize);
        env_num!(action_repeat, usize);
        env_num!(target_fps, f32);
        env_num!(capture_delay, f32);
        env_num!(tap_seconds, f32);
        env_num!(embed_dim, usize);
        env_num!(mem_tokens, usize);
        env_num!(patch_size, usize);
        env_num!(patch_width, usize);
        env_num!(transformer_layers, usize);
        env_num!(ffn_hidden, usize);
        env_num!(attention_heads, usize);
        env_num!(attention_dropout, f32);
        env_num!(seq_len, usize);
        env_num!(rollout_steps, usize);
        env_num!(minibatch_size, usize);
        env_num!(epochs_per_update, usize);
        env_num!(gamma, f32);
        env_num!(gae_lambda, f32);
        env_num!(clip_eps, f32);
        env_num!(value_coef, f32);
        env_num!(entropy_coef, f32);
        env_num!(adam_lr, f32);
        env_num!(adam_eps, f32);
        env_num!(grad_clip, f32);
        env_num!(reward_scale, f32);
        env_num!(w_episodic, f32);
        env_num!(w_depth, f32);
        env_num!(w_depth_progress, f32);
        env_num!(novelty_decay, f32);
        env_num!(novelty_survival_steps, f32);
        env_num!(w_idle, f32);
        env_num!(idle_ramp_steps, f32);
        env_num!(idle_ramp_max, f32);
        env_num!(novelty_bins, f32);
        env_num!(scale_floor, f32);
        env_num!(reward_clip, f32);
        env_num!(reset_jump, f32);
        env_num!(reset_revisit, f32);
        env_num!(reset_cooldown, usize);
        env_num!(frozen_warn_seconds, f32);
        env_num!(update_seconds_budget, f32);
        env_num!(status_every, usize);
        env_num!(slow_update_seconds, f32);
        env_num!(episodic_capacity, usize);
        env_num!(global_capacity, usize);
        env_num!(transition_hidden, usize);
        env_num!(transition_lr, f32);
        env_num!(w_novelty, f32);
        env_num!(w_novelty_cap, f32);
        env_num!(error_decay, f32);
        env_num!(max_held_keys, usize);
        env_num!(mouse_turn_start, i32);
        env_num!(mouse_turn_min, i32);
        env_num!(mouse_turn_max, i32);
        env_num!(mouse_adapt_rate, f32);
        env_num!(calibrate_min_hold, f32);
        env_num!(checkpoint_interval_sec, f32);
        env_num!(checkpoint_keep, usize);
        env_num!(torch_threads, usize);
        env_num!(seed, u64);

        env_bool!(prefer_game_window);
        env_bool!(enable_hotkeys);
        env_bool!(wait_for_start);
        env_bool!(resume);
        env_bool!(preview);
        env_bool!(show_model);

        env_string!(keymap_path);
        env_string!(calibrate_key);
        env_string!(checkpoint_dir);
        env_string!(final_model_path);
        env_string!(hotkey_pause);
        env_string!(hotkey_save);
        env_string!(hotkey_quit);
        if let Ok(raw) = std::env::var("BOT_LOG_JSON") {
            self.log_json = if raw.is_empty() { None } else { Some(raw) };
        }

        if let Ok(raw) = std::env::var("BOT_FOCUS_POLICY") {
            self.focus_policy = match raw.trim().to_ascii_lowercase().as_str() {
                "always" => FocusPolicy::Always,
                "never" => FocusPolicy::Never,
                _ => FocusPolicy::Once,
            };
        }
        if let Ok(raw) = std::env::var("BOT_PRIORITY") {
            self.priority = match raw.trim().to_ascii_lowercase().as_str() {
                "normal" => Priority::Normal,
                "high" => Priority::High,
                _ => Priority::BelowNormal,
            };
        }
        self
    }

    /// One line describing the shape of the model and the run.
    pub fn describe(&self) -> String {
        let speeds = self
            .speed_levels
            .iter()
            .map(|s| format!("{s}"))
            .collect::<Vec<_>>()
            .join("/");
        // The base step *in force*: `--mouse-turn` pins it, so printing
        // `mouse_turn_start` while a pin is set would describe a number the bot
        // is not using.
        let base = if self.mouse_turn_pixels > 0 {
            self.mouse_turn_pixels
        } else {
            self.mouse_turn_start
        };
        format!(
            "obs {}x{}x{} grey (+diff), repeat {}, swiglu+rope transformer {}d x {}L \
             x {}H (ffn {}, window {} frames, {} patch token(s)/frame), turn speed \
             x[{}] from a {}px base, rollout {} x {} epochs in {}-step minibatches",
            self.frame_stack,
            self.frame_size,
            self.frame_size,
            self.action_repeat,
            self.embed_dim,
            self.transformer_layers,
            self.attention_heads,
            self.ffn_hidden,
            self.mem_tokens,
            self.frame_tokens(),
            speeds,
            base,
            self.rollout_steps,
            self.epochs_per_update,
            self.minibatch_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_python_build() {
        let cfg = Config::default();
        assert_eq!(cfg.frame_tokens(), 36);
        assert_eq!(cfg.channels(), 5);
        assert_eq!(cfg.update_minibatches(), 4);
        cfg.validate().expect("shipped defaults must be valid");
    }

    #[test]
    fn validation_refuses_a_bad_rotary_width() {
        let cfg = Config {
            attention_heads: 5,
            ..Default::default()
        };
        // 96 / 5 is not even, and not even an integer, so this must be refused
        // rather than silently producing a broken rotation.
        assert!(cfg.validate().is_err());
    }
}
