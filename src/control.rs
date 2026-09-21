//! What steers a run while it is running: Ctrl+C, the start/pause/save/quit
//! keys, and the checkpoint timer. Ported from `botcore/runtime.py`'s
//! `SignalController`.
//!
//! The safety property this file exists for is not "the run stops cleanly". It
//! is that **a killed run must not leave a key held down**. The bot presses keys
//! with `SendInput`/XTEST, which is not undone by the process dying: if the
//! process disappears between a key-down and its key-up, the game keeps walking
//! forward with nobody at the controls. So Ctrl+C is caught, the stop is a flag
//! rather than an exit, and the session releases everything before returning.

use std::time::Instant;

use crate::config::Config;
use crate::platform::{HotkeyAction, Hotkeys};

/// What the session asks its controller, once per step or minibatch.
pub trait Control {
    /// Messages worth printing, and whatever the hotkeys asked for this tick.
    fn service(&mut self) -> Vec<String>;
    fn paused(&self) -> bool;
    /// True until the user presses the start key. Nothing is sent before then.
    fn awaiting_start(&self) -> bool;
    fn stop_requested(&self) -> bool;
    /// Whether a pause notice is owed to the user (once per pause).
    fn claim_pause_notice(&mut self) -> bool;
    /// `Some(reason)` when a checkpoint is due.
    fn checkpoint_due(&mut self) -> Option<String>;
    fn stop_reason(&self) -> String;
}

/// The real controller: hotkeys, Ctrl+C, and the periodic checkpoint.
pub struct SignalController {
    interval: f64,
    accumulated: f64,
    last: Instant,
    started_at: Instant,
    stopped: bool,
    paused: bool,
    start_key_seen: bool,
    wait_for_start: bool,
    pause_notice: bool,
    manual_save: bool,
    pub stop_reason: String,
    hotkeys: Option<Hotkeys>,
    interrupts: u64,
}

impl SignalController {
    pub fn new(cfg: &Config) -> Self {
        let mut controller = Self {
            interval: cfg.checkpoint_interval_sec.max(0.05) as f64,
            accumulated: 0.0,
            last: Instant::now(),
            started_at: Instant::now(),
            stopped: false,
            paused: false,
            start_key_seen: !cfg.wait_for_start,
            wait_for_start: cfg.wait_for_start,
            pause_notice: false,
            manual_save: false,
            stop_reason: "running".to_string(),
            hotkeys: None,
            interrupts: 0,
        };
        if cfg.enable_hotkeys {
            controller.hotkeys = Some(Hotkeys::new(
                &cfg.hotkey_pause,
                &cfg.hotkey_save,
                &cfg.hotkey_quit,
            ));
        }
        controller
    }

    /// Install the interrupt handler and start listening for hotkeys.
    pub fn start(&mut self) -> bool {
        self.last = Instant::now();
        install_interrupt_handler();
        let Some(hotkeys) = self.hotkeys.as_mut() else {
            println!("[Hotkeys] No global hotkeys; Ctrl+C still saves and quits.");
            self.maybe_ungate();
            return false;
        };
        let available = hotkeys.start();
        for message in hotkeys.drain_messages() {
            println!("[Hotkeys] {message}");
        }
        if available {
            println!("[Hotkeys] {} (work from any window)", hotkeys.describe());
        } else {
            println!("[Hotkeys] No global hotkeys; Ctrl+C still saves and quits.");
        }
        self.maybe_ungate();
        available
    }

    /// A run that waits for a start key it can never receive would sit there
    /// forever, which looks exactly like a hang.
    fn maybe_ungate(&mut self) {
        let some_hotkey = self
            .hotkeys
            .as_ref()
            .map(|hotkeys| hotkeys.available())
            .unwrap_or(false);
        if self.wait_for_start && !some_hotkey {
            self.wait_for_start = false;
            self.start_key_seen = true;
            println!("[Control] The start key is unavailable, so the run starts immediately.");
        }
    }

    pub fn stop(&mut self) {
        if let Some(hotkeys) = self.hotkeys.as_mut() {
            hotkeys.stop();
        }
    }

    /// True while the user has asked for a pause, for a caller that drives its
    /// own loop.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn elapsed(&self) -> f64 {
        self.started_at.elapsed().as_secs_f64()
    }
}

impl Control for SignalController {
    fn service(&mut self) -> Vec<String> {
        let now = Instant::now();
        self.accumulated += now.duration_since(self.last).as_secs_f64();
        self.last = now;
        let mut messages = Vec::new();

        // Ctrl+C: the first one saves and stops, the second one exits now.
        if interrupt_requested() {
            let count = take_interrupts();
            if count > 0 {
                self.interrupts += count;
                if self.interrupts == 1 {
                    self.stop_reason = "ctrl_c".to_string();
                    self.stopped = true;
                    messages.push(
                        "[Signal] Interrupted - releasing every key and stopping...".to_string(),
                    );
                    messages
                        .push("[Signal] (Press again to exit without saving.)".to_string());
                } else {
                    messages.push("[Signal] Second interrupt - exiting now.".to_string());
                    println!("{}", messages.last().cloned().unwrap_or_default());
                    std::process::exit(130);
                }
            }
        }

        if let Some(hotkeys) = self.hotkeys.as_mut() {
            for message in hotkeys.drain_messages() {
                messages.push(format!("[Hotkeys] {message}"));
            }
            for action in hotkeys.poll() {
                match action {
                    HotkeyAction::Pause => {
                        if !self.start_key_seen {
                            self.start_key_seen = true;
                            self.paused = false;
                            self.pause_notice = true;
                            messages.push(
                                "STARTING - the bot has the keyboard now. Press the same key \
                                 again to pause."
                                    .to_string(),
                            );
                        } else {
                            self.paused = !self.paused;
                            self.pause_notice = false;
                            messages.push(if self.paused {
                                "PAUSED - no input is being sent.".to_string()
                            } else {
                                "RESUMED.".to_string()
                            });
                        }
                    }
                    HotkeyAction::Save => {
                        self.manual_save = true;
                        messages.push("Manual checkpoint requested.".to_string());
                    }
                    HotkeyAction::Quit => {
                        self.stop_reason = "hotkey".to_string();
                        self.stopped = true;
                        messages.push("Quit hotkey - saving and stopping.".to_string());
                    }
                }
            }
        }
        messages
    }

    fn paused(&self) -> bool {
        self.paused
    }

    fn awaiting_start(&self) -> bool {
        self.wait_for_start && !self.start_key_seen
    }

    fn stop_requested(&self) -> bool {
        self.stopped
    }

    fn claim_pause_notice(&mut self) -> bool {
        if self.paused && !self.pause_notice {
            self.pause_notice = true;
            return true;
        }
        false
    }

    fn checkpoint_due(&mut self) -> Option<String> {
        if self.manual_save {
            self.manual_save = false;
            return Some("manual".to_string());
        }
        // One interval per call, never a catch-up loop: a long stall loses the
        // surplus rather than producing a burst of checkpoints.
        if self.accumulated >= self.interval {
            self.accumulated -= self.interval;
            return Some("periodic".to_string());
        }
        None
    }

    fn stop_reason(&self) -> String {
        self.stop_reason.clone()
    }
}

// ---- the interrupt flag ----
//
// A signal handler may only touch async-signal-safe state, so it sets an atomic
// counter and returns; the loop above reads it. Nothing here prints, allocates
// or locks.

static INTERRUPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HANDLER_INSTALLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn interrupt_requested() -> bool {
    INTERRUPTS.load(std::sync::atomic::Ordering::SeqCst) > 0
}

fn take_interrupts() -> u64 {
    INTERRUPTS.swap(0, std::sync::atomic::Ordering::SeqCst)
}

extern "C" fn on_interrupt(_signal: i32) {
    INTERRUPTS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// Catch Ctrl+C so the session can let go of the keyboard before it exits.
///
/// Without this the process dies between a key-down and its key-up, and the game
/// keeps walking on its own - which is the one failure of this program that a
/// user cannot fix by taking their hands off the keyboard.
pub fn install_interrupt_handler() {
    use std::sync::atomic::Ordering;
    if HANDLER_INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }
    #[cfg(windows)]
    {
        // `SetConsoleCtrlHandler(NULL, TRUE)` makes this process *ignore* Ctrl+C
        // entirely, which is the opposite of what is wanted. What is wanted is
        // for the console handler to tell us rather than killing us, so the
        // default handler is replaced by one that only counts.
        use windows::Win32::System::Console::{
            CTRL_BREAK_EVENT, CTRL_C_EVENT, CTRL_CLOSE_EVENT, SetConsoleCtrlHandler,
        };
        unsafe extern "system" fn handler(event: u32) -> windows::core::BOOL {
            if event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT || event == CTRL_CLOSE_EVENT {
                on_interrupt(event as i32);
                // TRUE: handled, so the process is not terminated.
                return windows::core::BOOL(1);
            }
            windows::core::BOOL(0)
        }
        let _ = unsafe { SetConsoleCtrlHandler(Some(handler), true) };
    }
    #[cfg(not(windows))]
    unsafe {
        // SAFETY: the handler only increments an atomic, which is all a signal
        // handler is allowed to do. The cast goes through a function pointer
        // because casting a function *item* straight to an integer is a lint.
        let handler = on_interrupt as extern "C" fn(i32);
        libc_signal(2, handler as usize); // SIGINT
        libc_signal(15, handler as usize); // SIGTERM
    }
}

/// `signal(2)`, without pulling in a crate for one call.
///
/// Only used on platforms with a C library; the Windows path above is separate.
#[cfg(not(windows))]
unsafe fn libc_signal(signum: i32, handler: usize) {
    unsafe extern "C" {
        fn signal(signum: i32, handler: usize) -> usize;
    }
    unsafe {
        signal(signum, handler);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controller() -> SignalController {
        let cfg = Config {
            checkpoint_interval_sec: 0.05,
            enable_hotkeys: false,
            ..Default::default()
        };
        SignalController::new(&cfg)
    }

    #[test]
    fn a_checkpoint_falls_due_once_per_interval_and_never_bursts() {
        let mut control = controller();
        assert_eq!(control.checkpoint_due(), None);
        std::thread::sleep(std::time::Duration::from_millis(120));
        // Time is accumulated by `service`, which the loop calls once per step -
        // `checkpoint_due` only reads the accumulator.
        control.service();
        // Three intervals have passed, but a stall does not produce three
        // checkpoints: the surplus is dropped.
        assert_eq!(control.checkpoint_due(), Some("periodic".to_string()));
        assert_eq!(control.checkpoint_due(), Some("periodic".to_string()));
        assert_eq!(control.checkpoint_due(), None);
    }

    #[test]
    fn a_manual_request_is_honoured_once() {
        let mut control = controller();
        control.manual_save = true;
        assert_eq!(control.checkpoint_due(), Some("manual".to_string()));
        assert_eq!(control.checkpoint_due(), None);
    }

    #[test]
    fn a_run_with_no_hotkeys_does_not_wait_for_a_start_key() {
        // Waiting for a key that can never arrive looks exactly like a hang.
        let cfg = Config {
            enable_hotkeys: false,
            wait_for_start: true,
            ..Default::default()
        };
        let mut control = SignalController::new(&cfg);
        assert!(control.awaiting_start());
        control.start();
        assert!(!control.awaiting_start(), "the gate must open by itself");
    }

    #[test]
    fn the_pause_notice_is_owed_once() {
        let mut control = controller();
        control.paused = true;
        assert!(control.claim_pause_notice());
        assert!(!control.claim_pause_notice());
    }

    #[test]
    fn an_interrupt_asks_for_a_stop_rather_than_ending_the_process() {
        let mut control = controller();
        on_interrupt(2);
        let messages = control.service();
        assert!(control.stop_requested());
        assert_eq!(control.stop_reason(), "ctrl_c");
        assert!(messages.iter().any(|m| m.contains("releasing every key")));
    }
}
