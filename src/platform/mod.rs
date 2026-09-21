//! The platform layer: one policy, three mechanisms.
//!
//! Everything in this module is platform-neutral *policy* - which windows are
//! safe to drive, how a remembered window is re-identified after a restart, what
//! the user is asked when nothing looks like a game. The *mechanism* lives in a
//! per-OS submodule behind the same function names:
//!
//! ```text
//! platform::windows   Win32: EnumWindows, PrintWindow, SendInput, RegisterHotKey
//! platform::linux     X11: XQueryTree, XGetImage, XTEST, XGrabKey
//! platform::macos     CoreGraphics: CGWindowList, CGEventPost, CGEventTap
//! platform::unsupported   everything else, with a readable refusal
//! ```
//!
//! `platform::current` re-exports whichever one this build selected, so the call
//! sites above this layer never mention an OS. That is what makes the rest of
//! the bot - the model, the reward, PPO, the observation pipeline - portable
//! without a single `cfg` in it: it only ever sees [`WindowInfo`] and the
//! functions in this file.
//!
//! What is *not* seamless, and is stated here rather than discovered later:
//!
//! * **Wayland** has no protocol for per-window capture or for listening to
//!   another application's global window list, and no way to inject input into
//!   one window. A Wayland session can be driven only through the portal (a
//!   full-screen share the user approves) or through `/dev/uinput`, which is
//!   kernel-level and needs the device. `platform::linux` reports which session
//!   it found and refuses clearly rather than pretending.
//! * **macOS** requires two separate TCC grants - Screen Recording for capture
//!   and Accessibility for input injection - and neither can be granted
//!   programmatically. Without them the capture API returns a desktop picture
//!   instead of the window and the input API does nothing, both without an
//!   error. [`crate::platform::macos::permissions`] exists to make that visible
//!   before a run starts rather than as a silent no-op mid-run.
//! * **Elevated windows**: on Windows a non-elevated process cannot send input
//!   to an elevated one (UIPI), and the equivalent exists on macOS with SIP. No
//!   amount of porting changes that.

use std::collections::HashSet;

/// A window handle, in whatever form the platform uses one.
///
/// Win32 `HWND` is a pointer, X11 `Window` is a 32-bit id, macOS `CGWindowID` is
/// a 32-bit id. All three fit, and all three are only ever compared or passed
/// back to the platform.
pub type Handle = isize;

/// One top-level window, in the shape every consumer wants it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowInfo {
    pub handle: Handle,
    pub title: String,
    pub pid: u32,
    pub process: String,
    /// `(left, top, width, height)` of the whole window frame.
    pub rect: (i32, i32, i32, i32),
    /// `(width, height)` of the area that gets captured.
    pub client: (i32, i32),
    pub minimized: bool,
    /// This process, or an ancestor of it. Never driven.
    pub own: bool,
    /// This process's own terminal/console window. Never driven.
    pub console: bool,
}

/// What the caller knows or remembers about the window it wants.
#[derive(Debug, Clone, Default)]
pub struct TargetRequest {
    /// `--window HANDLE`.
    pub configured: Option<Handle>,
    pub prefer_game_window: bool,
    pub allow_prompt: bool,
    /// The handle a previous `--calibrate` recorded.
    pub remembered_handle: Option<Handle>,
    /// The title it recorded, which survives the handle.
    pub remembered_title: String,
}

#[cfg(windows)]
pub mod windows;
#[cfg(windows)]
pub use windows as current;

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "linux")]
pub use linux as current;

#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "macos")]
pub use macos as current;

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
pub mod unsupported;
#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
pub use unsupported as current;

/// What a hotkey asks the run to do. Platform-neutral: the key that carries it
/// is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyAction {
    Pause,
    Save,
    Quit,
}

impl HotkeyAction {
    pub fn name(self) -> &'static str {
        match self {
            HotkeyAction::Pause => "pause",
            HotkeyAction::Save => "save",
            HotkeyAction::Quit => "quit",
        }
    }
}

/// The stateful pieces every backend provides: something that grabs frames from
/// a window, something that presses keys into one, the global hotkeys, and the
/// calibration recorder.
///
/// Each backend defines types with these names and these methods, so a call site
/// writes `platform::FrameGrabber` and never mentions an OS. There is no trait
/// here on purpose: the methods take and return plain data, and the cross-check
/// for each target is what proves every backend really has them.
pub use current::{FrameGrabber, Hotkeys, Injector, KeyRecorder};

/// The platform-neutral stub, used by the backends that do not exist yet.
pub mod unimplemented;

/// A cheap heuristic used only to decide whether to prompt.
///
/// It refuses this process's own windows outright, however large they are, and
/// it refuses the programs a developer already has open (browsers, editors,
/// chat). Both refusals are about the same failure: auto-detection that picks the
/// biggest window on the desktop picks the terminal the bot was started from.
///
/// The process tables differ per OS; the policy does not.
pub fn looks_like_game(window: &WindowInfo) -> bool {
    if window.own || window.console {
        return false;
    }
    let process = window.process.to_lowercase();
    let title = window.title.to_lowercase();
    if current::is_shell_process(&process) || current::is_non_game_process(&process) {
        return false;
    }
    if current::SHELL_TITLES
        .iter()
        .any(|shell| title.contains(shell))
    {
        return false;
    }
    let (width, height) = (window.rect.2, window.rect.3);
    if width < 480 || height < 320 {
        return false;
    }
    true
}

/// Why this window must not be driven, or an empty string when it looks safe.
///
/// Only self-inflicted cases are refused: the bot's own terminal is a guaranteed
/// way to send every keystroke into a shell instead of a game. A browser is a
/// wrong guess, not a hazard, so it is warned about rather than blocked.
pub fn window_risk(window: Option<&WindowInfo>) -> String {
    let Some(window) = window else {
        return String::new();
    };
    if window.handle == 0 {
        return "it has no window handle".to_string();
    }
    if window.handle == current::console_window() {
        return "it is this process's own console window, so the bot would type its \
                actions into a shell"
            .to_string();
    }
    if window.own || window.console || current::window_is_own_process(window.handle) {
        return "it belongs to this process or to the shell that launched it (the \
                terminal/IDE), so the bot would type its actions into a shell prompt \
                instead of the game"
            .to_string();
    }
    String::new()
}

/// `'Title' (process.exe) handle=N`, for logs.
pub fn describe_window(window: Option<&WindowInfo>) -> String {
    match window {
        None => "(no window)".to_string(),
        Some(window) => format!(
            "'{}' ({}) hwnd={}",
            if window.title.is_empty() {
                "?"
            } else {
                &window.title
            },
            if window.process.is_empty() {
                "?"
            } else {
                &window.process
            },
            window.handle
        ),
    }
}

/// Loose title comparison, because a game's title changes with its level.
///
/// Either title containing the other counts, and so does a shared first word of
/// at least three characters: "Minecraft 26.3 - Singleplayer" matches "Minecraft
/// 26.3 - Multiplayer". A handle that the OS has recycled onto an unrelated
/// window fails all three checks, which is the whole point of checking - a stale
/// handle is a wrong window.
pub fn titles_match(remembered: &str, current: &str) -> bool {
    let normalise = |text: &str| {
        text.to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let a = normalise(remembered);
    let b = normalise(current);
    if a.is_empty() || b.is_empty() {
        return false;
    }
    if a.contains(&b) || b.contains(&a) {
        return true;
    }
    let first_a = a.split(' ').next().unwrap_or("");
    let first_b = b.split(' ').next().unwrap_or("");
    first_a.chars().count() >= 3 && first_a == first_b
}

pub fn format_window_table(windows: &[WindowInfo]) -> String {
    let mut lines = Vec::new();
    for (index, window) in windows.iter().enumerate() {
        let (width, height) = (window.rect.2, window.rect.3);
        let mut notes: Vec<&str> = Vec::new();
        if window.minimized {
            notes.push("minimized");
        }
        if window.own || window.console {
            notes.push("the terminal the bot runs in - never driven");
        } else if current::is_non_game_process(&window.process.to_lowercase()) {
            notes.push("not a game");
        }
        let flag = if notes.is_empty() {
            String::new()
        } else {
            format!("  ({})", notes.join("; "))
        };
        let title: String = window.title.chars().take(44).collect();
        lines.push(format!(
            "  [{:2}] {:5}x{:<5} hwnd={:<10} {:<24} {}{}",
            index + 1,
            width,
            height,
            window.handle,
            if window.process.is_empty() {
                "?"
            } else {
                &window.process
            },
            title,
            flag
        ));
    }
    lines.join("\n")
}

/// Return a likely game handle, or `None` when nothing convincing is found.
pub fn find_game_window(prefer_largest: bool) -> Option<Handle> {
    for window in current::enumerate_windows(MIN_AREA) {
        if looks_like_game(&window) {
            return if prefer_largest {
                Some(window.handle)
            } else {
                None
            };
        }
    }
    None
}

/// 64 x 64, the Python's default floor for "worth listing at all".
pub const MIN_AREA: i32 = 64 * 64;

/// Decide which window to play.
///
/// Order: an explicit handle, then the window a previous `--calibrate` recorded -
/// by handle, then by title - then the window the user is looking at, then the
/// largest plausible game, then the picker.
///
/// The calibrated window comes second on purpose: it is the only evidence the
/// bot has that the user has already pointed at the game. Ignoring it is how a
/// run ends up driving the terminal it was launched from - which is the largest
/// window on most desktops, and the one window that must never be driven,
/// because every action would be typed into a shell prompt.
pub fn resolve_target_window(request: &TargetRequest) -> Option<WindowInfo> {
    if let Some(configured) = request.configured {
        let window = current::window_entry(configured)?;
        let risk = window_risk(Some(&window));
        if !risk.is_empty() {
            println!(
                "[Window] WARNING: {} - {}.",
                describe_window(Some(&window)),
                risk
            );
            println!(
                "[Window] Pass the game's handle instead: --list-windows, then --window HWND."
            );
        }
        return Some(window);
    }

    if let Some(remembered) = request.remembered_handle {
        match current::window_entry(remembered) {
            None => println!(
                "[Window] The window calibrated earlier (handle {remembered}) is not open \
                 any more; looking for the game."
            ),
            Some(window) => {
                if window.own || window.console {
                    println!(
                        "[Window] The handle calibrated earlier now points at {} - that is \
                         this terminal, not the game. Ignoring it.",
                        describe_window(Some(&window))
                    );
                } else if !request.remembered_title.is_empty()
                    && !titles_match(&request.remembered_title, &window.title)
                {
                    println!(
                        "[Window] handle {remembered} is now '{}', not '{}'; ignoring the \
                         remembered handle, because handles get reused.",
                        window.title, request.remembered_title
                    );
                } else if request.remembered_title.is_empty()
                    && current::is_non_game_process(&window.process.to_lowercase())
                {
                    println!(
                        "[Window] The calibrated handle {remembered} now belongs to {}, \
                         which is not a game; ignoring it.",
                        window.process
                    );
                } else {
                    println!(
                        "[Window] Using the window you calibrated against: {}",
                        describe_window(Some(&window))
                    );
                    return Some(window);
                }
            }
        }
    }

    let windows = current::enumerate_windows(MIN_AREA);
    // The handle is often gone (a game gets a new one every launch, and the OS
    // hands old numbers out again), but the title usually survives, so the
    // calibration still identifies the window it was recorded against.
    if !request.remembered_title.is_empty() {
        for window in &windows {
            if looks_like_game(window) && titles_match(&request.remembered_title, &window.title) {
                println!(
                    "[Window] Found the game you calibrated against, by title: {}",
                    describe_window(Some(window))
                );
                return Some(window.clone());
            }
        }
    }

    if request.prefer_game_window {
        let mut ordered: Vec<WindowInfo> = Vec::new();
        let foreground = current::window_entry(current::foreground_window());
        if let Some(foreground) = &foreground
            && looks_like_game(foreground)
        {
            ordered.push(foreground.clone());
        }
        for window in &windows {
            if looks_like_game(window) && !ordered.iter().any(|w| w.handle == window.handle) {
                ordered.push(window.clone());
            }
        }
        if let Some(chosen) = ordered.first() {
            let is_foreground = foreground
                .as_ref()
                .is_some_and(|f| f.handle == chosen.handle);
            if is_foreground {
                println!(
                    "[Window] Using the window in front: {}",
                    describe_window(Some(chosen))
                );
            } else {
                println!(
                    "[Window] Auto-selected the largest likely game: {} ({}x{})",
                    describe_window(Some(chosen)),
                    chosen.rect.2,
                    chosen.rect.3
                );
            }
            println!(
                "[Window] This terminal, any open browser/editor, and this process's own \
                 windows are never auto-selected. Use --pick or --window HWND for a \
                 different window."
            );
            return Some(chosen.clone());
        }
        println!(
            "[Window] Nothing on screen looks like a game (terminals, browsers and editors \
             are excluded on purpose)."
        );
    }

    if !request.allow_prompt {
        return None;
    }
    let chosen = pick_window_interactive(&windows)?;
    let risk = window_risk(Some(&chosen));
    if !risk.is_empty() {
        println!(
            "[Window] WARNING: {} - {}.",
            describe_window(Some(&chosen)),
            risk
        );
        println!(
            "[Window] Driving it would send every action into that program, not into a game."
        );
    }
    println!("[Window] Using {}", describe_window(Some(&chosen)));
    Some(chosen)
}

/// Ask which window to drive, on standard input.
pub fn pick_window_interactive(windows: &[WindowInfo]) -> Option<WindowInfo> {
    use std::io::Write;
    if windows.is_empty() {
        println!("[Window] No usable windows found. Start the game first.");
        return None;
    }
    println!();
    println!("{}", "-".repeat(78));
    println!("  WHICH WINDOW SHOULD THE BOT PLAY?");
    println!("{}", "-".repeat(78));
    println!("{}", format_window_table(windows));
    println!("{}", "-".repeat(78));
    loop {
        print!("  Number (blank = cancel): ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            println!();
            return None;
        }
        let line = line.trim();
        if line.is_empty() {
            return None;
        }
        match line.parse::<usize>() {
            Ok(index) if index >= 1 && index <= windows.len() => {
                return Some(windows[index - 1].clone());
            }
            Ok(_) => println!("  That number is not in the list."),
            Err(_) => println!("  Please type the number shown in brackets."),
        }
    }
}

/// What the platform layer cannot do here, said once, in one place.
pub fn capability_report() -> Vec<String> {
    current::capability_report()
}

/// Give the user their cursor back, whatever is holding it.
///
/// The injector releases its own confinement on the way out, including from a
/// panic. This is the version for the exits that do not unwind - the second
/// Ctrl+C, and the release path that cleans up after a run that died - because a
/// cursor clip is desktop-wide state and nothing else here will undo it.
pub fn free_cursor_clip() {
    current::free_cursor_clip();
}

/// Whether this build can actually drive a window on this machine.
pub fn is_supported() -> bool {
    current::is_supported()
}

/// The set of process ids belonging to this process's own shell chain.
pub fn own_process_ids() -> HashSet<u32> {
    current::own_process_ids()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(process: &str, title: &str, width: i32, height: i32) -> WindowInfo {
        WindowInfo {
            handle: 1,
            title: title.to_string(),
            pid: 42,
            process: process.to_string(),
            rect: (0, 0, width, height),
            client: (width, height),
            minimized: false,
            own: false,
            console: false,
        }
    }

    #[test]
    fn this_processs_own_windows_are_never_a_likely_game() {
        let mut own = window("game.exe", "Some Game", 1280, 720);
        own.own = true;
        assert!(!looks_like_game(&own));
        own.own = false;
        own.console = true;
        assert!(!looks_like_game(&own));
    }

    #[test]
    fn a_small_window_is_not_a_game() {
        assert!(!looks_like_game(&window("game.exe", "Tooltip", 200, 100)));
    }

    #[test]
    fn titles_match_the_way_the_python_says() {
        // A level change keeps the first words and the handle stays valid.
        assert!(titles_match(
            "Minecraft 26.3 - Singleplayer",
            "Minecraft 26.3 - Multiplayer"
        ));
        assert!(titles_match("Minecraft", "Minecraft 26.3"));
        assert!(!titles_match("Minecraft 26.3", "Notepad"));
        assert!(!titles_match("", "Minecraft"));
        assert!(!titles_match("ab one", "ab two"));
    }
}
