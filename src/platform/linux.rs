//! The Linux/X11 mechanism behind [`crate::platform`].
//!
//! Everything the Windows build does, done with X11: list windows, capture one,
//! inject input into it, grab hotkeys. Two of the pieces need no display at all -
//! the process-name lookup (`/proc/<pid>/exe`) and the walk up the parent chain
//! that identifies the shell the bot was launched from.
//!
//! **Which session, stated once.** Under X11 the bot can do what the Windows
//! build does. Under Wayland it can do none of it, and the reasons are the design
//! of the protocol rather than missing code:
//!
//! * there is no protocol for enumerating another application's windows, so the
//!   bot cannot offer a picker or re-find the window it calibrated against;
//! * there is no protocol for capturing one window - the portal shares a screen
//!   or a region the user picks, and the compositor may not even let the client
//!   know which window is in it;
//! * there is no protocol for injecting input into another client. The only route
//!   is `/dev/uinput`, which is kernel-level: it needs the device node and it
//!   types into whatever has focus, with no per-window check available at all -
//!   which is exactly the safety property this bot is built around.
//!
//! A Wayland build is therefore a *different* program with a different safety
//! story, not a port of this one. This module refuses clearly instead of
//! half-working.
//!
//! **Two honest limits of the X11 path**, both stated to the user rather than
//! discovered mid-run:
//!
//! * `GetImage` on a window reads that window's own contents, but a window that
//!   is *occluded* on a server with no compositing manager may return whatever
//!   happens to be in the framebuffer there. Games are normally played in front,
//!   so this is a note rather than a blocker; `--check-capture` exists to catch
//!   it.
//! * Mouse-look is injected with `XTestFakeInput`, which generates *core* pointer
//!   motion. A game that reads raw device motion through XInput2 may ignore it,
//!   in which case the view will not turn however large the delta is. That is the
//!   same problem `--mouse-turn` exists for on Windows, with a different cause.

use std::collections::HashSet;

use x11rb::connection::Connection;
use x11rb::protocol::Event;
use x11rb::protocol::xproto::{
    AtomEnum, ClientMessageEvent, ConnectionExt as _, EventMask, GrabMode, ImageFormat,
    ModMask, Window,
};
use x11rb::rust_connection::RustConnection;

use super::{Handle, WindowInfo};

/// Terminals and shells. The group that matters: a bot that drives its own
/// terminal types its actions into a shell prompt.
const TERMINAL_PROCESSES: [&str; 21] = [
    "gnome-terminal",
    "gnome-terminal-server",
    "konsole",
    "xterm",
    "xterm-256color",
    "kitty",
    "alacritty",
    "wezterm",
    "wezterm-gui",
    "foot",
    "footclient",
    "terminator",
    "tilix",
    "xfce4-terminal",
    "mate-terminal",
    "lxterminal",
    "urxvt",
    "st",
    "tmux",
    "screen",
    "bash",
];

const APP_PROCESSES: [&str; 16] = [
    "firefox",
    "firefox-bin",
    "chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
    "brave",
    "opera",
    "vivaldi",
    "code",
    "code-oss",
    "codium",
    "sublime_text",
    "discord",
    "slack",
    "zoom",
];

pub const SHELL_TITLES: [&str; 3] = ["program manager", "settings", "task manager"];

pub fn is_shell_process(process: &str) -> bool {
    TERMINAL_PROCESSES.contains(&process)
}

pub fn is_non_game_process(process: &str) -> bool {
    TERMINAL_PROCESSES.contains(&process) || APP_PROCESSES.contains(&process)
}

/// Is this session X11, where the bot can actually work?
pub fn is_x11() -> bool {
    let session = std::env::var("XDG_SESSION_TYPE").unwrap_or_default();
    let wayland = std::env::var("WAYLAND_DISPLAY")
        .map(|display| !display.is_empty())
        .unwrap_or(false);
    if wayland || session.eq_ignore_ascii_case("wayland") {
        return false;
    }
    // No session type recorded: fall back to the presence of a display. The
    // connection itself is the real test.
    std::env::var("DISPLAY")
        .map(|display| !display.is_empty())
        .unwrap_or(false)
}

/// Open a connection to the display, or say why not.
fn connect() -> Result<(RustConnection, usize), String> {
    if !is_x11() {
        return Err(capability_report()
            .first()
            .cloned()
            .unwrap_or_else(|| "no X11 display".to_string()));
    }
    x11rb::connect(None).map_err(|error| format!("could not open the X11 display: {error}"))
}

pub fn is_supported() -> bool {
    connect().is_ok()
}

/// Nothing here confines a cursor to a window, so there is nothing to give back.
///
/// X11 has the mechanism - `XGrabPointer` with `confine_to` - but a grab is
/// exclusive: it collides with the one a game already holds on its own window
/// and can leave the pointer frozen for every other client if it fails
/// half-way. Confining the pointer is left to the game, and the injector says so
/// once at the first action rather than this being silent.
pub fn free_cursor_clip() {}

pub fn capability_report() -> Vec<String> {
    let mut report = Vec::new();
    if !is_x11() {
        report.push(
            "This is a Wayland session. Wayland gives an application no way to list another \
             program's windows, capture one window, or send input to one window - so the bot \
             cannot do here what it does on Windows, and it cannot even check that the window \
             it would drive is the right one."
                .to_string(),
        );
        report.push(
            "Two honest routes: log in to an X11 session (GNOME: pick 'GNOME on Xorg' at the \
             login screen; KDE: start a Plasma X11 session), where this build works; or wait \
             for a Wayland build, which needs the screen-cast portal for capture and \
             /dev/uinput for input, and which gives up the focus check that keeps the bot from \
             typing into your terminal."
                .to_string(),
        );
        return report;
    }
    match connect() {
        Err(error) => report.push(error),
        Ok((connection, screen_num)) => {
            match x11rb::protocol::xtest::get_version(&connection, 2, 2) {
                Ok(cookie) => {
                    if let Err(error) = cookie.reply() {
                        report.push(format!(
                            "The XTEST extension is not answering ({error}); input injection \
                             will not work."
                        ));
                    }
                }
                Err(_) => report.push(
                    "The XTEST extension is missing, so the bot cannot send input to a window. \
                     On some minimal window managers it has to be enabled explicitly."
                        .to_string(),
                ),
            }
            let screen = &connection.setup().roots[screen_num];
            report.push(format!(
                "X11 display ready: {}x{} root window, screen {screen_num}.",
                screen.width_in_pixels, screen.height_in_pixels
            ));
        }
    }
    report
}

// ---- processes, which need no display at all ----

/// Executable name owning `pid`, from `/proc`, or an empty string.
pub fn get_process_name(pid: u32) -> String {
    if pid == 0 {
        return String::new();
    }
    std::fs::read_link(format!("/proc/{pid}/exe"))
        .ok()
        .and_then(|path| path.file_name().map(|name| name.to_string_lossy().to_string()))
        .unwrap_or_default()
}

/// The parent of `pid` from `/proc/<pid>/stat`, or 0.
fn parent_of(pid: u32) -> u32 {
    let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return 0;
    };
    // The second field is the command name in parentheses and may contain
    // spaces, so the fields after it are found from the last ')'.
    let Some(close) = stat.rfind(')') else {
        return 0;
    };
    let mut fields = stat[close + 1..].split_whitespace();
    let _state = fields.next();
    fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0)
}

/// This process plus every ancestor: the shell, the terminal, the IDE.
pub fn own_process_ids() -> HashSet<u32> {
    let mut ids = HashSet::new();
    let own = std::process::id();
    ids.insert(own);
    let mut pid = own;
    // Depth cap: never loop forever, whatever /proc says.
    for _ in 0..16 {
        let parent = parent_of(pid);
        if parent == 0 || ids.contains(&parent) {
            break;
        }
        ids.insert(parent);
        pid = parent;
    }
    ids
}

pub fn window_is_own_process(handle: Handle) -> bool {
    let Some(pid) = window_pid(handle) else {
        return false;
    };
    own_process_ids().contains(&pid)
}

/// X11 has no console window: the terminal is an ordinary window, and the
/// parent-chain check above is what identifies it.
pub fn console_window() -> Handle {
    0
}

// ---- windows ----

fn atom(connection: &RustConnection, name: &[u8]) -> Option<u32> {
    connection
        .intern_atom(false, name)
        .ok()?
        .reply()
        .ok()
        .map(|reply| reply.atom)
}

fn property(connection: &RustConnection, window: Window, name: &[u8]) -> Option<Vec<u8>> {
    let atom = atom(connection, name)?;
    let reply = connection
        .get_property(false, window, atom, AtomEnum::ANY, 0, 1024)
        .ok()?
        .reply()
        .ok()?;
    (!reply.value.is_empty()).then_some(reply.value)
}

/// A window's title, preferring what a modern toolkit sets.
fn window_name(connection: &RustConnection, window: Window) -> String {
    for property_name in [&b"_NET_WM_NAME"[..], &b"WM_NAME"[..]] {
        if let Some(value) = property(connection, window, property_name) {
            let text = String::from_utf8_lossy(&value)
                .trim_end_matches('\0')
                .to_string();
            if !text.trim().is_empty() {
                return text;
            }
        }
    }
    String::new()
}

/// The pid a window belongs to, from `_NET_WM_PID`.
fn window_pid_from(connection: &RustConnection, window: Window) -> u32 {
    let Some(value) = property(connection, window, b"_NET_WM_PID") else {
        return 0;
    };
    // A CARDINAL, in native byte order.
    match value.len() {
        4 => u32::from_ne_bytes([value[0], value[1], value[2], value[3]]),
        8 => u64::from_ne_bytes([
            value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
        ]) as u32,
        _ => 0,
    }
}

fn window_pid(handle: Handle) -> Option<u32> {
    let (connection, _) = connect().ok()?;
    let pid = window_pid_from(&connection, handle as Window);
    (pid != 0).then_some(pid)
}

/// Is this window mapped, and therefore something a user can be looking at?
fn is_viewable(connection: &RustConnection, window: Window) -> bool {
    let Ok(cookie) = connection.get_window_attributes(window) else {
        return false;
    };
    let Ok(attributes) = cookie.reply() else {
        return false;
    };
    // Menus, tooltips and other unmanaged popups set this, and driving one is
    // never what the user meant.
    if attributes.override_redirect {
        return false;
    }
    attributes.map_state == x11rb::protocol::xproto::MapState::VIEWABLE
}

/// One window, in the shape [`crate::platform`] wants, or `None`.
fn entry(connection: &RustConnection, root: Window, window: Window) -> Option<WindowInfo> {
    if !is_viewable(connection, window) {
        return None;
    }
    let title = window_name(connection, window);
    if title.trim().is_empty() {
        return None;
    }
    let geometry = connection.get_geometry(window).ok()?.reply().ok()?;
    if geometry.width == 0 || geometry.height == 0 {
        return None;
    }
    // Absolute screen coordinates: the window may have been reparented into a
    // frame by the window manager, so its own x/y are relative to that frame.
    let placed = connection
        .translate_coordinates(window, root, 0, 0)
        .ok()?
        .reply()
        .ok()?;
    let pid = window_pid_from(connection, window);
    let own = pid != 0 && own_process_ids().contains(&pid);
    Some(WindowInfo {
        handle: window as Handle,
        title,
        pid,
        process: get_process_name(pid),
        rect: (
            placed.dst_x as i32,
            placed.dst_y as i32,
            geometry.width as i32,
            geometry.height as i32,
        ),
        client: (geometry.width as i32, geometry.height as i32),
        minimized: false,
        own,
        console: false,
    })
}

/// Every viewable, titled, large-enough top-level window, largest first.
pub fn enumerate_windows(min_area: i32) -> Vec<WindowInfo> {
    let Ok((connection, screen_num)) = connect() else {
        return Vec::new();
    };
    let root = connection.setup().roots[screen_num].root;
    let Ok(cookie) = connection.query_tree(root) else {
        return Vec::new();
    };
    let Ok(tree) = cookie.reply() else {
        return Vec::new();
    };
    let mut found: Vec<WindowInfo> = tree
        .children
        .iter()
        .filter_map(|child| entry(&connection, root, *child))
        .filter(|window| window.rect.2.saturating_mul(window.rect.3) >= min_area)
        .collect();
    found.sort_by(|a, b| {
        let area = |w: &WindowInfo| w.rect.2.saturating_mul(w.rect.3);
        area(b).cmp(&area(a))
    });
    found
}

pub fn window_entry(handle: Handle) -> Option<WindowInfo> {
    let (connection, screen_num) = connect().ok()?;
    let root = connection.setup().roots[screen_num].root;
    entry(&connection, root, handle as Window)
}

pub fn client_size(handle: Handle) -> (i32, i32) {
    let Ok((connection, _)) = connect() else {
        return (0, 0);
    };
    match connection
        .get_geometry(handle as Window)
        .map(|cookie| cookie.reply())
    {
        Ok(Ok(geometry)) => (geometry.width as i32, geometry.height as i32),
        _ => (0, 0),
    }
}

pub fn is_minimized(_handle: Handle) -> bool {
    // A minimized window on X11 is simply not viewable, which
    // `enumerate_windows` already filters on.
    false
}

pub fn is_window(handle: Handle) -> bool {
    let Ok((connection, _)) = connect() else {
        return false;
    };
    connection
        .get_window_attributes(handle as Window)
        .map(|cookie| cookie.reply().is_ok())
        .unwrap_or(false)
}

pub fn window_title(handle: Handle) -> String {
    let Ok((connection, _)) = connect() else {
        return String::new();
    };
    window_name(&connection, handle as Window)
}

pub fn get_window_pid(handle: Handle) -> u32 {
    window_pid(handle).unwrap_or(0)
}

/// The window the user is looking at.
///
/// `_NET_ACTIVE_WINDOW` from the window manager is the answer when there is one,
/// because `GetInputFocus` returns the window the *WM* gave focus to, which on a
/// reparenting window manager is the frame rather than the application's window.
pub fn foreground_window() -> Handle {
    let Ok((connection, screen_num)) = connect() else {
        return 0;
    };
    let root = connection.setup().roots[screen_num].root;
    if let Some(value) = property(&connection, root, b"_NET_ACTIVE_WINDOW")
        && value.len() == 4
    {
        let window = u32::from_ne_bytes([value[0], value[1], value[2], value[3]]);
        if window != 0 {
            return window as Handle;
        }
    }
    let Ok(cookie) = connection.get_input_focus() else {
        return 0;
    };
    let Ok(focus) = cookie.reply() else {
        return 0;
    };
    // Walk up to the top-level window the user is actually looking at.
    let mut window = focus.focus;
    for _ in 0..8 {
        let Ok(cookie) = connection.query_tree(window) else {
            break;
        };
        let Ok(tree) = cookie.reply() else {
            break;
        };
        if tree.parent == 0 || tree.parent == root {
            break;
        }
        window = tree.parent;
    }
    window as Handle
}

// =============================================================================
// Frame capture
// =============================================================================

/// Captures a window with `GetImage`.
///
/// There is no device context or bitmap to keep alive on X11, so unlike the
/// Win32 grabber this holds a connection and a buffer. One connection per
/// grabber, reused: opening a connection per frame would cost far more than the
/// grab itself.
pub struct FrameGrabber {
    connection: Option<RustConnection>,
    handle: Handle,
    pixels: Vec<u8>,
    pub last_error: Option<String>,
    pub frames: u64,
    total_seconds: f64,
}

impl FrameGrabber {
    pub fn new(handle: Handle) -> Self {
        match connect() {
            Ok((connection, _)) => Self {
                connection: Some(connection),
                handle,
                pixels: Vec::new(),
                last_error: None,
                frames: 0,
                total_seconds: 0.0,
            },
            Err(error) => Self {
                connection: None,
                handle,
                pixels: Vec::new(),
                last_error: Some(error),
                frames: 0,
                total_seconds: 0.0,
            },
        }
    }

    pub fn grab(&mut self) -> Option<crate::capture::Frame> {
        let started = std::time::Instant::now();
        let result = self.grab_inner();
        self.frames += 1;
        self.total_seconds += started.elapsed().as_secs_f64();
        result
    }

    fn grab_inner(&mut self) -> Option<crate::capture::Frame> {
        let connection = self.connection.as_ref()?;
        let window = self.handle as Window;
        let geometry = connection.get_geometry(window).ok()?.reply().ok()?;
        let (width, height) = (geometry.width, geometry.height);
        if width == 0 || height == 0 {
            self.last_error = Some("window has no client area".to_string());
            return None;
        }
        // A 24- or 32-bit TrueColor visual delivers four bytes per pixel in
        // BGRX order on a little-endian server, which is what the observation
        // pipeline wants. Anything else - a 16-bit visual, a paletted one -
        // would need a conversion this refuses to guess at.
        if geometry.depth != 24 && geometry.depth != 32 {
            self.last_error = Some(format!(
                "window depth {} is not a 24- or 32-bit TrueColor visual",
                geometry.depth
            ));
            return None;
        }
        let reply = match connection
            .get_image(ImageFormat::Z_PIXMAP, window, 0, 0, width, height, !0)
            .map(|cookie| cookie.reply())
        {
            Ok(Ok(reply)) => reply,
            Ok(Err(error)) => {
                self.last_error = Some(format!("GetImage failed: {error}"));
                return None;
            }
            Err(error) => {
                self.last_error = Some(format!("GetImage request failed: {error}"));
                return None;
            }
        };
        let expected = width as usize * height as usize * 4;
        if reply.data.len() < expected {
            self.last_error = Some(format!(
                "GetImage returned {} bytes, expected {expected}",
                reply.data.len()
            ));
            return None;
        }
        // BGRX -> BGRA: the pipeline reads the first three bytes, and making the
        // fourth opaque is cheaper than trusting whatever the server left there.
        self.pixels.clear();
        self.pixels.extend_from_slice(&reply.data[..expected]);
        for pixel in self.pixels.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
        self.last_error = None;
        Some(crate::capture::Frame::new(
            width as usize,
            height as usize,
            self.pixels.clone(),
        ))
    }

    pub fn mean_ms(&self) -> f64 {
        1000.0 * self.total_seconds / self.frames.max(1) as f64
    }
}

// =============================================================================
// Input injection
// =============================================================================

/// X11 keysyms, which is what a key *is* here.
///
/// A keymap's names are portable (`W`, `SPACE`, `F8`); the number behind a name
/// is not. On Windows that number is a virtual-key code, here it is a keysym, and
/// the keycode that actually gets sent is looked up from the keyboard mapping at
/// run time.
pub fn keysym_for(name: &str) -> Option<u32> {
    let upper = name.trim().to_uppercase();
    let named: [(&str, u32); 22] = [
        ("SPACE", 0x0020),
        ("ESC", 0xFF1B),
        ("TAB", 0xFF09),
        ("ENTER", 0xFF0D),
        ("BACKSPACE", 0xFF08),
        ("UP", 0xFF52),
        ("DOWN", 0xFF54),
        ("LEFT", 0xFF51),
        ("RIGHT", 0xFF53),
        ("INSERT", 0xFF63),
        ("DELETE", 0xFFFF),
        ("HOME", 0xFF50),
        ("END", 0xFF57),
        ("PAGEUP", 0xFF55),
        ("PAGEDOWN", 0xFF56),
        ("LSHIFT", 0xFFE1),
        ("RSHIFT", 0xFFE2),
        ("LCTRL", 0xFFE3),
        ("RCTRL", 0xFFE4),
        ("LALT", 0xFFE9),
        ("RALT", 0xFFEA),
        ("SHIFT", 0xFFE1),
    ];
    if let Some((_, keysym)) = named.iter().find(|(key, _)| *key == upper) {
        return Some(*keysym);
    }
    let bytes = upper.as_bytes();
    if bytes.len() == 1 {
        let byte = bytes[0];
        // The keysym of an unshifted letter is its *lowercase* code: the key
        // without shift produces 'w'. Sending the keycode found from that keysym
        // is what presses the W key.
        if byte.is_ascii_uppercase() {
            return Some(byte.to_ascii_lowercase() as u32);
        }
        if byte.is_ascii_digit() {
            return Some(byte as u32);
        }
    }
    let punctuation: [(&str, u32); 11] = [
        (";", 0x003B),
        ("=", 0x003D),
        (",", 0x002C),
        ("-", 0x002D),
        (".", 0x002E),
        ("/", 0x002F),
        ("`", 0x0060),
        ("[", 0x005B),
        ("\\", 0x005C),
        ("]", 0x005D),
        ("'", 0x0027),
    ];
    if let Some((_, keysym)) = punctuation.iter().find(|(key, _)| *key == upper) {
        return Some(*keysym);
    }
    if let Some(rest) = upper.strip_prefix('F')
        && let Ok(number) = rest.parse::<u32>()
        && (1..=24).contains(&number)
    {
        return Some(0xFFBE + number - 1);
    }
    None
}

/// Turn the code `parse_hotkey` produced into an X11 keysym.
///
/// `parse_hotkey` speaks virtual-key codes, because that is the vocabulary the
/// keymap file and the user's `--hotkeys` flags are written in. The mapping to
/// keysyms is small and explicit: letters, digits and the function keys.
pub fn keysym_from_code(code: u32) -> Option<u32> {
    match code {
        0x30..=0x39 => Some(code),
        0x41..=0x5A => Some((code as u8).to_ascii_lowercase() as u32),
        0x70..=0x87 => Some(0xFFBE + (code - 0x6F) - 1),
        0x20 => Some(0x0020),
        0x1B => Some(0xFF1B),
        0x09 => Some(0xFF09),
        0x0D => Some(0xFF0D),
        0x25 => Some(0xFF51),
        0x26 => Some(0xFF52),
        0x27 => Some(0xFF53),
        0x28 => Some(0xFF54),
        _ => None,
    }
}

/// The keycode that produces `keysym` at its first (unshifted) level.
fn keycode_for(connection: &RustConnection, keysym: u32) -> Option<u8> {
    let setup = connection.setup();
    let min = setup.min_keycode;
    let count = setup.max_keycode - setup.min_keycode + 1;
    let reply = connection
        .get_keyboard_mapping(min, count)
        .ok()?
        .reply()
        .ok()?;
    let per = reply.keysyms_per_keycode as usize;
    if per == 0 {
        return None;
    }
    for index in 0..count as usize {
        if reply.keysyms.get(index * per) == Some(&keysym) {
            return Some(min + index as u8);
        }
    }
    None
}

/// Sends real keyboard and mouse input to the focused window, through XTEST.
///
/// The same contract as the Win32 injector: nothing is sent unless the game is
/// the front window, releases are forced so a key can never be left down, and
/// every delivered event is counted, because "the bot is deciding" and "the bot
/// is playing" look identical in every other number.
pub struct Injector {
    connection: Option<RustConnection>,
    screen_num: usize,
    handle: Handle,
    pub policy: crate::config::FocusPolicy,
    pub suspended: bool,
    pub enabled: bool,
    pub blocked: u64,
    pub failures: u64,
    pub skipped_unfocused: u64,
    pub events_sent: u64,
    pub keys_sent: u64,
    pub mouse_sent: u64,
    pub last_error: Option<String>,
    focused: bool,
    focus_attempted: bool,
    focus_warned: bool,
    focus_notice_at: std::time::Instant,
    pub transient_vks: HashSet<u32>,
    /// Accepted for the shared call site; confinement is not implemented on X11,
    /// and `begin_action` says so once rather than pretending.
    pub clip_cursor: bool,
    clip_notice: bool,
    keycode_cache: std::collections::HashMap<u32, u8>,
    announced: bool,
}

impl Injector {
    pub fn new(handle: Handle, policy: crate::config::FocusPolicy, clip_cursor: bool) -> Self {
        let (connection, screen_num, error) = match connect() {
            Ok((connection, screen_num)) => (Some(connection), screen_num, None),
            Err(error) => (None, 0, Some(error)),
        };
        Self {
            connection,
            screen_num,
            handle,
            policy,
            suspended: false,
            enabled: true,
            blocked: 0,
            failures: 0,
            skipped_unfocused: 0,
            events_sent: 0,
            keys_sent: 0,
            mouse_sent: 0,
            last_error: error,
            focused: false,
            focus_attempted: false,
            focus_warned: false,
            focus_notice_at: std::time::Instant::now(),
            transient_vks: HashSet::new(),
            clip_cursor,
            clip_notice: false,
            keycode_cache: std::collections::HashMap::new(),
            announced: false,
        }
    }

    pub fn focused(&self) -> bool {
        foreground_window() == self.handle
    }

    /// Bring the game window forward, once, using the EWMH protocol.
    ///
    /// On X11 this is a *request* to the window manager, which may refuse it -
    /// and a refusal is not an error, it is the desktop doing its job. That is
    /// why the caller is told whether it worked.
    pub fn acquire_focus(&mut self, force: bool) -> bool {
        use x11rb::protocol::xproto::{ClientMessageData, ConfigureWindowAux, StackMode};
        if self.focused() {
            self.focused = true;
            self.focus_warned = false;
            return true;
        }
        if self.policy == crate::config::FocusPolicy::Never {
            return false;
        }
        if !force && self.policy == crate::config::FocusPolicy::Once && self.focus_attempted {
            return false;
        }
        self.focus_attempted = true;
        let Some(connection) = self.connection.as_ref() else {
            return false;
        };
        let root = connection.setup().roots[self.screen_num].root;
        let window = self.handle as Window;
        let mut asked = false;
        if let Some(atom) = atom(connection, b"_NET_ACTIVE_WINDOW") {
            // Source indication 2: a pager. Window managers are allowed to ignore
            // anything else from another client.
            let event = ClientMessageEvent::new(32, window, atom, ClientMessageData::from([2u32, 0, 0, 0, 0]));
            asked = connection
                .send_event(
                    false,
                    root,
                    EventMask::SUBSTRUCTURE_REDIRECT | EventMask::SUBSTRUCTURE_NOTIFY,
                    event,
                )
                .is_ok();
        }
        // Raising is a separate, weaker request: some window managers honour it
        // when they will not move focus.
        let raised = connection
            .configure_window(window, &ConfigureWindowAux::new().stack_mode(StackMode::ABOVE))
            .is_ok();
        let _ = connection.flush();
        // Give the window manager a moment to act before asking who has focus.
        std::thread::sleep(std::time::Duration::from_millis(30));
        self.focused = self.focused();
        if self.focused {
            self.focus_warned = false;
            return true;
        }
        if !asked && !raised {
            self.last_error = Some("no window manager answered the focus request".to_string());
        }
        false
    }

    pub fn begin_action(&mut self) {
        self.focused = self.focused();
        if self.focused {
            if self.focus_warned {
                self.focus_warned = false;
                println!("[Input] The game window is focused again; input resumed.");
            }
            // Said once, at the first action of a real run: this is the one gap
            // in this backend that shows up as input landing in the wrong place
            // rather than as nothing happening.
            if self.clip_cursor && !self.clip_notice {
                self.clip_notice = true;
                println!(
                    "[Input] Cursor confinement is not implemented on X11, so a game that \
                     frees the cursor (an inventory, a pause menu) can still have the pointer \
                     walk out of the window and a click land outside it. --no-clip-cursor \
                     silences this."
                );
            }
            return;
        }
        if self.policy == crate::config::FocusPolicy::Always && self.acquire_focus(true) {
            return;
        }
        if !self.focus_warned {
            self.focus_warned = true;
            self.focus_notice_at = std::time::Instant::now();
            println!(
                "[Input] The game window is not focused, so nothing is being sent. Click the \
                 game - input resumes on its own. (The terminal can be used normally \
                 meanwhile.)"
            );
        } else if self.focus_notice_at.elapsed().as_secs_f64() >= 60.0 {
            self.focus_notice_at = std::time::Instant::now();
            println!(
                "[Input] Still sending nothing: the game window has not been the front window \
                 for {} step(s). Click the game to let the bot play it.",
                self.skipped_unfocused
            );
        }
    }

    pub fn end_action(&mut self) {
        self.focused = false;
    }

    pub fn status_line(&self) -> String {
        let mut line = format!(
            "[Input] game window {} - {} key event(s) and {} mouse event(s) delivered, \
             {} step(s) skipped while unfocused",
            if self.focused() { "focused" } else { "NOT focused" },
            self.keys_sent,
            self.mouse_sent,
            self.skipped_unfocused
        );
        if self.failures > 0 {
            line += &format!(", {} inject failure(s)", self.failures);
        }
        if let Some(error) = &self.last_error {
            line += &format!(" ({error})");
        }
        line
    }

    fn may_inject(&mut self, what: &str, forcing: bool) -> bool {
        if self.connection.is_none() {
            if !self.announced {
                self.announced = true;
                self.failures += 1;
                println!(
                    "[Input] BLOCKED injection ({what}): {}",
                    self.last_error.clone().unwrap_or_default()
                );
            }
            return false;
        }
        if !self.enabled {
            self.blocked += 1;
            if self.blocked <= 5 {
                println!(
                    "[Input] BLOCKED injection ({what}): input is disabled in this mode. This \
                     is a bug."
                );
            }
            return false;
        }
        if self.suspended {
            return false;
        }
        if !forcing && !self.focused {
            self.skipped_unfocused += 1;
            return false;
        }
        true
    }

    /// The X11 keycode for a keysym, looked up once and kept.
    fn keycode(&mut self, keysym: u32) -> Option<u8> {
        if let Some(cached) = self.keycode_cache.get(&keysym) {
            return Some(*cached);
        }
        let found = keycode_for(self.connection.as_ref()?, keysym)?;
        self.keycode_cache.insert(keysym, found);
        Some(found)
    }

    fn fake(&mut self, kind: u8, detail: u8, root_x: i16, root_y: i16) -> bool {
        let Some(connection) = self.connection.as_ref() else {
            return false;
        };
        let root = connection.setup().roots[self.screen_num].root;
        x11rb::protocol::xtest::fake_input(
            connection,
            kind,
            detail,
            0, // CurrentTime
            root,
            root_x,
            root_y,
            0,
        )
        .map(|cookie| cookie.check().is_ok())
        .unwrap_or(false)
            && connection.flush().is_ok()
    }

    fn send_key(&mut self, keysym: u32, up: bool, forcing: bool) {
        let what = format!("key 0x{keysym:02X} {}", if up { "up" } else { "down" });
        if !self.may_inject(&what, forcing) {
            return;
        }
        let Some(keycode) = self.keycode(keysym) else {
            self.failures += 1;
            self.last_error = Some(format!("no keycode produces keysym 0x{keysym:02X}"));
            return;
        };
        let kind = if up {
            x11rb::protocol::xproto::KEY_RELEASE_EVENT
        } else {
            x11rb::protocol::xproto::KEY_PRESS_EVENT
        };
        if self.fake(kind, keycode, 0, 0) {
            self.keys_sent += 1;
            self.events_sent += 1;
        } else {
            self.failures += 1;
            self.last_error = Some("XTestFakeInput(key) failed".to_string());
        }
    }

    fn send_button(&mut self, button: u8, up: bool, forcing: bool) {
        let what = format!("button {button} {}", if up { "up" } else { "down" });
        if !self.may_inject(&what, forcing) {
            return;
        }
        let kind = if up {
            x11rb::protocol::xproto::BUTTON_RELEASE_EVENT
        } else {
            x11rb::protocol::xproto::BUTTON_PRESS_EVENT
        };
        if self.fake(kind, button, 0, 0) {
            self.mouse_sent += 1;
            self.events_sent += 1;
        } else {
            self.failures += 1;
            self.last_error = Some("XTestFakeInput(button) failed".to_string());
        }
    }

    /// Relative mouse motion, as an absolute move from the current position.
    ///
    /// `XTestFakeInput` moves the pointer to a position rather than by a delta,
    /// so the current position is read first. The catch worth knowing: this
    /// produces *core* pointer events, and a game reading raw device motion
    /// through XInput2 may ignore them entirely - in which case the view will not
    /// turn however large the delta is. See the module comment.
    pub fn mouse_move(&mut self, dx: i32, dy: i32) {
        if !self.may_inject("mouse move", false) {
            return;
        }
        let Some(connection) = self.connection.as_ref() else {
            return;
        };
        let root = connection.setup().roots[self.screen_num].root;
        let (width, height) = (
            connection.setup().roots[self.screen_num].width_in_pixels as i32,
            connection.setup().roots[self.screen_num].height_in_pixels as i32,
        );
        let Ok(cookie) = connection.query_pointer(root) else {
            self.failures += 1;
            return;
        };
        let Ok(pointer) = cookie.reply() else {
            self.failures += 1;
            return;
        };
        let target_x = (pointer.root_x as i32 + dx).clamp(0, (width - 1).max(0)) as i16;
        let target_y = (pointer.root_y as i32 + dy).clamp(0, (height - 1).max(0)) as i16;
        if self.fake(
            x11rb::protocol::xproto::MOTION_NOTIFY_EVENT,
            0,
            target_x,
            target_y,
        ) {
            self.mouse_sent += 1;
            self.events_sent += 1;
        } else {
            self.failures += 1;
            self.last_error = Some("XTestFakeInput(motion) failed".to_string());
        }
    }

    pub fn press_vk(&mut self, vk: u32) {
        if vk != 0 {
            self.send_key(vk, false, false);
        }
    }

    pub fn release_vk(&mut self, vk: u32) {
        // Forced, as on Windows: never leave a key down.
        if vk != 0 {
            self.send_key(vk, true, true);
        }
    }

    pub fn tap_vk(&mut self, vk: u32, seconds: f32) {
        if vk == 0 {
            return;
        }
        self.transient_vks.insert(vk);
        self.send_key(vk, false, false);
        std::thread::sleep(std::time::Duration::from_secs_f32(seconds.max(0.0)));
        self.send_key(vk, true, false);
        self.transient_vks.remove(&vk);
    }

    pub fn mouse_down(&mut self, button: &str) {
        if let Some(code) = button_code(button) {
            self.send_button(code, false, false);
        }
    }

    pub fn mouse_up(&mut self, button: &str) {
        if let Some(code) = button_code(button) {
            self.send_button(code, true, true);
        }
    }

    pub fn click(&mut self, button: &str, seconds: f32) {
        let marker = match button {
            "left" => Some(0x01),
            "middle" => Some(0x02),
            "right" => Some(0x03),
            _ => None,
        };
        if let Some(marker) = marker {
            self.transient_vks.insert(marker);
        }
        self.mouse_down(button);
        std::thread::sleep(std::time::Duration::from_secs_f32(seconds.max(0.0)));
        self.mouse_up(button);
        if let Some(marker) = marker {
            self.transient_vks.remove(&marker);
        }
    }

    pub fn release_all(&mut self, held_vks: &[u32]) {
        let held: Vec<u32> = held_vks.to_vec();
        for vk in held {
            self.release_vk(vk);
        }
    }
}

fn button_code(button: &str) -> Option<u8> {
    match button {
        "left" => Some(1),
        "middle" => Some(2),
        "right" => Some(3),
        _ => None,
    }
}

// =============================================================================
// Global hotkeys
// =============================================================================

/// Modifier masks, as X11 numbers them.
const X_SHIFT: u16 = 1 << 0;
const X_LOCK: u16 = 1 << 1;
const X_CONTROL: u16 = 1 << 2;
const X_MOD1: u16 = 1 << 3; // Alt
const X_MOD2: u16 = 1 << 4; // NumLock, usually
const X_MOD4: u16 = 1 << 6; // Super

/// The bot's modifier flags, translated to this platform's.
///
/// `parse_hotkey` speaks the Windows vocabulary, because that is the one the
/// user's configuration and the keymap file are written in. What a modifier *is*
/// differs per platform, so the translation lives here rather than in the parser.
pub fn x11_modifiers(parsed: u32) -> u16 {
    let mut mask = 0u16;
    if parsed & crate::keys::MOD_SHIFT != 0 {
        mask |= X_SHIFT;
    }
    if parsed & crate::keys::MOD_CONTROL != 0 {
        mask |= X_CONTROL;
    }
    if parsed & crate::keys::MOD_ALT != 0 {
        mask |= X_MOD1;
    }
    if parsed & crate::keys::MOD_WIN != 0 {
        mask |= X_MOD4;
    }
    mask
}

pub struct Hotkeys {
    specs: Vec<(super::HotkeyAction, String)>,
    actions: std::sync::Arc<std::sync::Mutex<Vec<super::HotkeyAction>>>,
    messages: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    registered: std::sync::Arc<std::sync::Mutex<Vec<(super::HotkeyAction, String)>>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Hotkeys {
    pub fn new(pause: &str, save: &str, quit: &str) -> Self {
        Self {
            specs: vec![
                (super::HotkeyAction::Pause, pause.to_string()),
                (super::HotkeyAction::Save, save.to_string()),
                (super::HotkeyAction::Quit, quit.to_string()),
            ],
            actions: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            messages: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            registered: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            thread: None,
        }
    }

    pub fn start(&mut self) -> bool {
        let specs = self.specs.clone();
        let actions = std::sync::Arc::clone(&self.actions);
        let messages = std::sync::Arc::clone(&self.messages);
        let registered = std::sync::Arc::clone(&self.registered);
        let stop = std::sync::Arc::clone(&self.stop);
        let (ready_sender, ready_receiver) = std::sync::mpsc::channel::<()>();
        match std::thread::Builder::new()
            .name("hotkeys".to_string())
            .spawn(move || hotkey_loop(specs, actions, messages, registered, stop, ready_sender))
        {
            Ok(handle) => self.thread = Some(handle),
            Err(error) => {
                self.messages
                    .lock()
                    .map(|mut queue| queue.push(format!("Hotkeys unavailable: {error}")))
                    .ok();
                return false;
            }
        }
        let _ = ready_receiver.recv_timeout(std::time::Duration::from_secs(3));
        self.available()
    }

    pub fn available(&self) -> bool {
        self.registered
            .lock()
            .map(|list| !list.is_empty())
            .unwrap_or(false)
    }

    pub fn poll(&mut self) -> Vec<super::HotkeyAction> {
        self.actions
            .lock()
            .map(|mut queue| std::mem::take(&mut *queue))
            .unwrap_or_default()
    }

    pub fn drain_messages(&mut self) -> Vec<String> {
        self.messages
            .lock()
            .map(|mut queue| std::mem::take(&mut *queue))
            .unwrap_or_default()
    }

    pub fn describe(&self) -> String {
        let registered = self
            .registered
            .lock()
            .map(|list| list.clone())
            .unwrap_or_default();
        if registered.is_empty() {
            return "none".to_string();
        }
        ["pause", "save", "quit"]
            .iter()
            .filter_map(|name| {
                registered
                    .iter()
                    .find(|(action, _)| action.name() == *name)
                    .map(|(action, spec)| {
                        format!("{}={}", action.name().to_uppercase(), spec.to_uppercase())
                    })
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn stop(&mut self) {
        self.stop
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Hotkeys {
    fn drop(&mut self) {
        self.stop();
    }
}

fn hotkey_loop(
    specs: Vec<(super::HotkeyAction, String)>,
    actions: std::sync::Arc<std::sync::Mutex<Vec<super::HotkeyAction>>>,
    messages: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    registered: std::sync::Arc<std::sync::Mutex<Vec<(super::HotkeyAction, String)>>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ready: std::sync::mpsc::Sender<()>,
) {
    let Ok((connection, screen_num)) = connect() else {
        messages
            .lock()
            .map(|mut queue| queue.push("no X11 display for hotkeys".to_string()))
            .ok();
        let _ = ready.send(());
        return;
    };
    let root = connection.setup().roots[screen_num].root;
    let mut grabs: Vec<(u16, u8)> = Vec::new();
    let mut bindings: Vec<(u8, u16, super::HotkeyAction)> = Vec::new();

    for (action, spec) in &specs {
        if spec.is_empty() {
            continue;
        }
        let (parsed_modifiers, code) = match crate::keys::parse_hotkey(spec) {
            Ok(parsed) => parsed,
            Err(message) => {
                messages.lock().map(|mut queue| queue.push(message)).ok();
                continue;
            }
        };
        let Some(keysym) = keysym_from_code(code) else {
            messages
                .lock()
                .map(|mut queue| queue.push(format!("'{spec}' has no X11 keysym")))
                .ok();
            continue;
        };
        let Some(keycode) = keycode_for(&connection, keysym) else {
            messages
                .lock()
                .map(|mut queue| {
                    queue.push(format!("no key produces {spec} on this keyboard layout"))
                })
                .ok();
            continue;
        };
        let base = x11_modifiers(parsed_modifiers);
        let mut any = false;
        // NumLock and CapsLock change the modifier state, so a grab that names
        // only the modifiers the user pressed stops firing the moment either is
        // on. All four combinations are grabbed for that reason.
        for lock in [0u16, X_LOCK, X_MOD2, X_LOCK | X_MOD2] {
            let modifiers = base | lock;
            let grabbed = connection
                .grab_key(
                    false,
                    root,
                    ModMask::from(modifiers),
                    keycode,
                    GrabMode::ASYNC,
                    GrabMode::ASYNC,
                )
                .map(|cookie| cookie.check().is_ok())
                .unwrap_or(false);
            if grabbed {
                grabs.push((modifiers, keycode));
                any = true;
            }
        }
        if any {
            bindings.push((keycode, base, *action));
            registered
                .lock()
                .map(|mut list| list.push((*action, spec.clone())))
                .ok();
        } else {
            messages
                .lock()
                .map(|mut queue| {
                    queue.push(format!(
                        "Could not register {} for '{}' - another program may already use it.",
                        spec.to_uppercase(),
                        action.name()
                    ))
                })
                .ok();
        }
    }
    let _ = connection.flush();
    let _ = ready.send(());

    while !stop.load(std::sync::atomic::Ordering::SeqCst) && !bindings.is_empty() {
        match connection.poll_for_event() {
            Ok(Some(Event::KeyPress(event))) => {
                // The lock modifiers are masked out: which of the four grabs
                // fired is not what the user pressed.
                let state = u16::from(event.state) & !(X_LOCK | X_MOD2);
                if let Some((_, _, action)) = bindings
                    .iter()
                    .find(|(keycode, modifiers, _)| *keycode == event.detail && *modifiers == state)
                    && let Ok(mut queue) = actions.lock()
                {
                    queue.push(*action);
                }
            }
            Ok(Some(_)) => {}
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(_) => break,
        }
    }

    for (modifiers, keycode) in grabs {
        let _ = connection
            .ungrab_key(keycode, root, ModMask::from(modifiers))
            .map(|cookie| cookie.check());
    }
    let _ = connection.flush();
    registered.lock().map(|mut list| list.clear()).ok();
}

// =============================================================================
// The calibration recorder
// =============================================================================

/// The recorder, not built yet for X11.
///
/// What it has to be: a *passive* global key listener, which on X11 means either
/// the RECORD extension or XInput2 raw events selected on the root window. What
/// it must not be is `XGrabKeyboard`, which would take the keyboard away from the
/// game - the exact opposite of what calibration is for. Until it lands this
/// refuses rather than half-working, and says which route it will take.
pub struct KeyRecorder {
    announced: bool,
}

impl KeyRecorder {
    pub fn start(_toggle_vk: u32, _target: Handle, _only_when_focused: bool) -> Self {
        Self { announced: false }
    }

    pub fn installed(&mut self) -> bool {
        if !self.announced {
            self.announced = true;
            println!(
                "[Calibrate] X11 key recording is not built yet in this port. The recorder \
                 needs the RECORD extension (or XInput2 raw events) to listen to keys without \
                 taking the keyboard from the game; until then, edit keymap.json by hand and \
                 the bot will use it."
            );
        }
        false
    }

    pub fn pump(&mut self) {}

    pub fn recording(&self) -> bool {
        false
    }

    pub fn toggles(&self) -> u64 {
        0
    }

    pub fn events(&self) -> u64 {
        0
    }

    pub fn stop(&mut self) {}

    pub fn snapshot(&self, _target: Option<&WindowInfo>) -> crate::keys::Recording {
        crate::keys::Recording::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_own_process_chain_includes_this_process() {
        let own = own_process_ids();
        assert!(own.contains(&std::process::id()));
    }

    #[test]
    fn process_names_resolve_from_proc() {
        let name = get_process_name(std::process::id());
        assert!(!name.is_empty(), "the test binary has an /proc entry");
    }

    #[test]
    fn keysyms_follow_the_x11_conventions() {
        // A letter's unshifted keysym is its lowercase code: that is the key the
        // user means by "W".
        assert_eq!(keysym_for("W"), Some(0x77));
        assert_eq!(keysym_for("w"), Some(0x77));
        assert_eq!(keysym_for("7"), Some(0x37));
        assert_eq!(keysym_for("SPACE"), Some(0x0020));
        assert_eq!(keysym_for("F8"), Some(0xFFC5));
        assert_eq!(keysym_for("UP"), Some(0xFF52));
        // A mouse button is not a keysym, and is refused rather than guessed at.
        assert_eq!(keysym_for("MOUSE_LEFT"), None);
    }

    #[test]
    fn the_virtual_key_codes_of_the_config_become_keysyms() {
        // "F8" as a virtual key code comes back as the X11 function-key keysym.
        assert_eq!(keysym_from_code(0x77), Some(0xFFC5));
        assert_eq!(keysym_from_code(0x57), Some(0x77));
        assert_eq!(keysym_from_code(0x37), Some(0x37));
        assert_eq!(keysym_from_code(0x20), Some(0x20));
        // Something with no X11 equivalent is refused rather than guessed.
        assert_eq!(keysym_from_code(0x01), None);
    }

    #[test]
    fn the_bots_modifier_vocabulary_translates_to_x11_masks() {
        // The parser speaks Windows flags; X11 numbers its modifiers
        // differently, and getting this wrong registers a hotkey that never
        // fires.
        assert_eq!(x11_modifiers(crate::keys::MOD_SHIFT), X_SHIFT);
        assert_eq!(x11_modifiers(crate::keys::MOD_CONTROL), X_CONTROL);
        assert_eq!(x11_modifiers(crate::keys::MOD_ALT), X_MOD1);
        assert_eq!(x11_modifiers(crate::keys::MOD_WIN), X_MOD4);
        assert_eq!(
            x11_modifiers(crate::keys::MOD_CONTROL | crate::keys::MOD_SHIFT),
            X_CONTROL | X_SHIFT
        );
        assert_eq!(x11_modifiers(0), 0);
    }
}
