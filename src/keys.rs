//! What the bot is allowed to press, and how it presses it. Ported from
//! `botcore/keys.py`.
//!
//! Two ideas carry this module:
//!
//! **The whitelist.** The bot may only ever press keys that have been
//! explicitly allowed, and that list comes from a keymap file which
//! `--calibrate` writes by watching the user play. A key that is not in the
//! list cannot be pressed and cannot be learned, so there is no path by which
//! the bot reaches into a game's menus or its own dev overlays.
//!
//! **The factored action space.** One action per combination of held keys, look
//! direction, turn distance and click explodes into hundreds of mostly-nonsense
//! actions. Instead the action is four independent decisions - which keys to
//! hold (capped), which way to swing, how far, and one momentary button - each
//! with its own policy head, sampled together and credited together.
//!
//! Splitting "which way" from "how far" is what lets the bot own its mouse
//! speed: the distance is a multiplier on a base step the model carries as its
//! own state, not a list of hand-tuned pixel deltas.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde_json::{Map, Value};

/// The four ways a view can swing, plus "do not turn".
///
/// A direction, not a distance: how far the mouse actually moves is the speed
/// decision, so a direction is all the head has to choose.
pub const DIRECTIONS: [(&str, (i32, i32)); 5] = [
    ("no turn", (0, 0)),
    ("left", (-1, 0)),
    ("right", (1, 0)),
    ("up", (0, -1)),
    ("down", (0, 1)),
];

/// Mouse buttons, by the name a keymap uses.
const MOUSE_BUTTON_VK: [(&str, u32); 3] =
    [("left", 0x01), ("right", 0x02), ("middle", 0x04)];

const MOUSE_VK_NAME: [(u32, &str); 3] =
    [(0x01, "MOUSE_LEFT"), (0x02, "MOUSE_RIGHT"), (0x04, "MOUSE_MIDDLE")];

/// Punctuation keys, which have no `chr()` of their own.
const EXTRA_NAMES: [(u32, &str); 11] = [
    (0xBA, ";"),
    (0xBB, "="),
    (0xBC, ","),
    (0xBD, "-"),
    (0xBE, "."),
    (0xBF, "/"),
    (0xC0, "`"),
    (0xDB, "["),
    (0xDC, "\\"),
    (0xDD, "]"),
    (0xDE, "'"),
];

/// The keys a bot needs most, used to order the action space so the useful
/// entries come first.
const KEY_PRIORITY: [&str; 11] = [
    "W",
    "A",
    "S",
    "D",
    "SPACE",
    "LSHIFT",
    "LCTRL",
    "MOUSE_LEFT",
    "MOUSE_RIGHT",
    "E",
    "Q",
];

/// Falls back to this when no keymap file exists: a conventional FPS layout.
pub const DEFAULT_HOLD_KEYS: [&str; 7] = ["W", "A", "S", "D", "SPACE", "LSHIFT", "LCTRL"];
pub const DEFAULT_TAP_KEYS: [&str; 6] = ["E", "Q", "1", "2", "3", "4"];
pub const DEFAULT_MOUSE_BUTTONS: [&str; 2] = ["MOUSE_LEFT", "MOUSE_RIGHT"];

/// Canonical name -> virtual key code.
///
/// Sided modifiers matter: games bind `VK_LSHIFT`/`VK_LCONTROL` (0xA0/0xA2), and
/// the generic `VK_SHIFT` (0x10) is a different key that many games ignore.
pub fn key_code(name: &str) -> Option<u32> {
    let upper = name.trim().to_uppercase();
    let fixed: [(&str, u32); 33] = [
        ("SPACE", 0x20),
        ("ESC", 0x1B),
        ("TAB", 0x09),
        ("ENTER", 0x0D),
        ("BACKSPACE", 0x08),
        ("LSHIFT", 0xA0),
        ("RSHIFT", 0xA1),
        ("SHIFT", 0xA0),
        ("LCTRL", 0xA2),
        ("RCTRL", 0xA3),
        ("CTRL", 0xA2),
        ("LALT", 0xA4),
        ("RALT", 0xA5),
        ("ALT", 0xA4),
        ("UP", 0x26),
        ("DOWN", 0x28),
        ("LEFT", 0x25),
        ("RIGHT", 0x27),
        ("INSERT", 0x2D),
        ("DELETE", 0x2E),
        ("HOME", 0x24),
        ("END", 0x23),
        ("PAGEUP", 0x21),
        ("PAGEDOWN", 0x22),
        ("MOUSE_LEFT", 0x01),
        ("MOUSE_RIGHT", 0x02),
        ("MOUSE_MIDDLE", 0x04),
        (";", 0xBA),
        ("=", 0xBB),
        (",", 0xBC),
        ("-", 0xBD),
        (".", 0xBE),
        ("/", 0xBF),
    ];
    if let Some((_, code)) = fixed.iter().find(|(key, _)| *key == upper) {
        return Some(*code);
    }
    if let Some((code, _)) = EXTRA_NAMES.iter().find(|(_, key)| *key == upper) {
        return Some(*code);
    }
    let bytes = upper.as_bytes();
    // A-Z and 0-9 are their own codes.
    if bytes.len() == 1 {
        let byte = bytes[0];
        if byte.is_ascii_uppercase() || byte.is_ascii_digit() {
            return Some(byte as u32);
        }
        if byte == b'`' {
            return Some(0xC0);
        }
        if byte == b'[' {
            return Some(0xDB);
        }
        if byte == b'\\' {
            return Some(0xDC);
        }
        if byte == b']' {
            return Some(0xDD);
        }
        if byte == b'\'' {
            return Some(0xDE);
        }
    }
    // NUMPAD0-9 and F1-F24.
    if let Some(rest) = upper.strip_prefix("NUMPAD")
        && let Ok(digit) = rest.parse::<u32>()
        && digit <= 9
    {
        return Some(0x60 + digit);
    }
    if let Some(rest) = upper.strip_prefix('F')
        && let Ok(number) = rest.parse::<u32>()
        && (1..=24).contains(&number)
    {
        return Some(0x6F + number);
    }
    None
}

/// Canonical name for a virtual-key code.
///
/// The Python's version names the letters, digits, numpad and function keys and
/// falls back to `VK_XX` for everything else, so it renders the space bar as
/// `VK_20`. This names the named keys too, because a name that does not round
/// trip through [`key_code`] is a name that shows up as a hex code in the status
/// line the user is reading. The keymap file is unaffected either way: it stores
/// the virtual-key code alongside the name.
pub fn key_name(vk: u32, extended: bool) -> String {
    if (0x41..=0x5A).contains(&vk) || (0x30..=0x39).contains(&vk) {
        return char::from_u32(vk).unwrap_or('?').to_string();
    }
    if (0x60..=0x69).contains(&vk) {
        return format!("NUMPAD{}", vk - 0x60);
    }
    if (0x70..=0x87).contains(&vk) {
        return format!("F{}", vk - 0x6F);
    }
    match vk {
        0x10 => return if extended { "RSHIFT" } else { "LSHIFT" }.to_string(),
        0x11 => return if extended { "RCTRL" } else { "LCTRL" }.to_string(),
        0x12 => return if extended { "RALT" } else { "LALT" }.to_string(),
        0x20 => return "SPACE".to_string(),
        0x1B => return "ESC".to_string(),
        0x09 => return "TAB".to_string(),
        0x0D => return "ENTER".to_string(),
        0x08 => return "BACKSPACE".to_string(),
        0x26 => return "UP".to_string(),
        0x28 => return "DOWN".to_string(),
        0x25 => return "LEFT".to_string(),
        0x27 => return "RIGHT".to_string(),
        0x2D => return "INSERT".to_string(),
        0x2E => return "DELETE".to_string(),
        0x24 => return "HOME".to_string(),
        0x23 => return "END".to_string(),
        0x21 => return "PAGEUP".to_string(),
        0x22 => return "PAGEDOWN".to_string(),
        0xA0 => return "LSHIFT".to_string(),
        0xA1 => return "RSHIFT".to_string(),
        0xA2 => return "LCTRL".to_string(),
        0xA3 => return "RCTRL".to_string(),
        0xA4 => return "LALT".to_string(),
        0xA5 => return "RALT".to_string(),
        _ => {}
    }
    if let Some((_, name)) = MOUSE_VK_NAME.iter().find(|(code, _)| *code == vk) {
        return name.to_string();
    }
    if let Some((_, name)) = EXTRA_NAMES.iter().find(|(code, _)| *code == vk) {
        return name.to_string();
    }
    format!("VK_{vk:02X}")
}

pub fn is_mouse_vk(vk: u32) -> bool {
    matches!(vk, 0x01 | 0x02 | 0x04)
}

fn sort_key(name: &str) -> (usize, String) {
    match KEY_PRIORITY.iter().position(|key| *key == name) {
        Some(index) => (index, name.to_string()),
        None => (KEY_PRIORITY.len(), name.to_string()),
    }
}

// =============================================================================
// Keymap
// =============================================================================

/// What `--calibrate` measured about this game's mouse look.
///
/// Kept in the keymap because it is the evidence behind `default_mouse_turn`:
/// when that number is a twitch because the game locks the cursor, this is where
/// the user can see that the recorder knew it was measuring a flick.
#[derive(Debug, Clone, PartialEq)]
pub struct MouseSensitivity {
    pub samples: u64,
    pub median_pixels_per_step: f64,
    pub max_pixels_per_step: f64,
}

/// The bot's whitelist: which keys exist, which are held vs tapped, and how far
/// a mouse turn step is.
#[derive(Debug, Clone)]
pub struct Keymap {
    pub path: Option<String>,
    pub origin: String,
    pub game: String,
    pub game_hwnd: Option<usize>,
    pub game_resolution: Vec<i32>,
    pub samples: u32,
    /// Pixels one 1x mouse step moves, as measured by `--calibrate` or set by
    /// `--mouse-turn`.
    pub default_mouse_turn: i32,
    pub mouse_sensitivity: Option<MouseSensitivity>,
    pub holds: Vec<String>,
    pub taps: Vec<String>,
    pub vks: HashMap<String, u32>,
    pub mouse_buttons: Vec<String>,
}

impl Default for Keymap {
    fn default() -> Self {
        Self::from_json(&Value::Null)
    }
}

impl Keymap {
    /// Read a keymap out of parsed JSON, tolerating everything the Python does.
    pub fn from_json(raw: &Value) -> Self {
        let object = raw.as_object().cloned().unwrap_or_default();
        let text = |key: &str, fallback: &str| -> String {
            object
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or(fallback)
                .to_string()
        };
        let number = |key: &str, fallback: i64| -> i64 {
            object.get(key).and_then(Value::as_i64).unwrap_or(fallback)
        };
        let mut keymap = Self {
            path: None,
            origin: text("origin", "default"),
            game: text("game", ""),
            game_hwnd: object
                .get("game_hwnd")
                .and_then(Value::as_u64)
                .map(|v| v as usize),
            game_resolution: object
                .get("game_resolution")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_i64)
                        .map(|v| v as i32)
                        .collect()
                })
                .unwrap_or_default(),
            samples: number("samples", 0) as u32,
            default_mouse_turn: number("default_mouse_turn", 20) as i32,
            mouse_sensitivity: object.get("mouse_sensitivity").and_then(|value| {
                let fields = value.as_object()?;
                Some(MouseSensitivity {
                    samples: fields.get("samples").and_then(Value::as_u64).unwrap_or(0),
                    median_pixels_per_step: fields
                        .get("median_pixels_per_step")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0),
                    max_pixels_per_step: fields
                        .get("max_pixels_per_step")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0),
                })
            }),
            holds: Vec::new(),
            taps: Vec::new(),
            vks: HashMap::new(),
            mouse_buttons: object
                .get("mouse_buttons")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(|s| s.to_uppercase())
                        .collect()
                })
                .unwrap_or_default(),
        };
        if let Some(keys) = object.get("keys").and_then(Value::as_object) {
            for (name, entry) in keys {
                let upper = name.to_uppercase();
                let (code, hold) = match entry {
                    Value::Number(value) => (value.as_u64().unwrap_or(0) as u32, "hold"),
                    Value::Object(fields) => (
                        fields.get("vk").and_then(Value::as_u64).unwrap_or(0) as u32,
                        fields.get("hold").and_then(Value::as_str).unwrap_or("hold"),
                    ),
                    _ => (0, "hold"),
                };
                let code = if code != 0 {
                    code
                } else {
                    key_code(&upper).unwrap_or(0)
                };
                if code == 0 {
                    continue;
                }
                keymap.vks.insert(upper.clone(), code);
                if hold == "hold" {
                    keymap.holds.push(upper);
                } else {
                    keymap.taps.push(upper);
                }
            }
        }
        keymap
    }

    pub fn load(path: impl AsRef<Path>) -> Option<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return None;
        }
        let text = std::fs::read_to_string(path).ok()?;
        let raw: Value = serde_json::from_str(&text).ok()?;
        if !raw.is_object() {
            println!(
                "[Keymap] '{}' is not a keymap object; ignoring it.",
                path.display()
            );
            return None;
        }
        let mut keymap = Self::from_json(&raw);
        keymap.path = Some(path.to_string_lossy().to_string());
        Some(keymap)
    }

    /// The built-in whitelist, used before any `--calibrate` run.
    pub fn default_layout() -> Self {
        let mut keymap = Self {
            origin: "default".to_string(),
            game: "generic defaults (FPS-style layout)".to_string(),
            default_mouse_turn: 20,
            ..Self::empty()
        };
        for name in DEFAULT_HOLD_KEYS {
            keymap.add(name, true, None);
        }
        for name in DEFAULT_TAP_KEYS {
            keymap.add(name, false, None);
        }
        for name in DEFAULT_MOUSE_BUTTONS {
            keymap.add(name, true, None);
        }
        keymap
    }

    fn empty() -> Self {
        Self {
            path: None,
            origin: "default".to_string(),
            game: String::new(),
            game_hwnd: None,
            game_resolution: Vec::new(),
            samples: 0,
            default_mouse_turn: 20,
            mouse_sensitivity: None,
            holds: Vec::new(),
            taps: Vec::new(),
            vks: HashMap::new(),
            mouse_buttons: Vec::new(),
        }
    }

    /// Add a key. Returns true when it was not already present.
    pub fn add(&mut self, name: &str, hold: bool, vk: Option<u32>) -> bool {
        let upper = name.to_uppercase();
        let code = match vk.or_else(|| key_code(&upper)) {
            Some(code) => code,
            None => return false,
        };
        self.vks.insert(upper.clone(), code);
        let (bucket, other) = if hold {
            (&mut self.holds, &mut self.taps)
        } else {
            (&mut self.taps, &mut self.holds)
        };
        other.retain(|existing| *existing != upper);
        if !bucket.contains(&upper) {
            bucket.push(upper);
            return true;
        }
        false
    }

    pub fn sort_keys(&mut self) {
        self.holds.sort_by(|a, b| sort_key(a).cmp(&sort_key(b)));
        self.taps.sort_by(|a, b| sort_key(a).cmp(&sort_key(b)));
        self.mouse_buttons
            .sort_by(|a, b| sort_key(a).cmp(&sort_key(b)));
    }

    /// Override the measured mouse step. Zero keeps the measurement.
    pub fn set_mouse_turn(&mut self, pixels: Option<i32>) -> bool {
        let value = pixels.unwrap_or(0);
        if value <= 0 || value == self.default_mouse_turn {
            return false;
        }
        println!(
            "[Keymap] Mouse turn step: {} px (measured) -> {} px (--mouse-turn).",
            self.default_mouse_turn, value
        );
        self.default_mouse_turn = value;
        true
    }

    pub fn describe(&self) -> String {
        let join = |items: &[String]| {
            if items.is_empty() {
                "-".to_string()
            } else {
                items.join(", ")
            }
        };
        format!(
            "hold=[{}]  tap=[{}]  mouse=[{}]",
            join(&self.holds),
            join(&self.taps),
            join(&self.mouse_buttons)
        )
    }

    pub fn to_json(&self) -> Value {
        let mut keys = Map::new();
        for name in &self.holds {
            let mut entry = Map::new();
            entry.insert("vk".into(), self.vks.get(name).copied().unwrap_or(0).into());
            entry.insert("hold".into(), "hold".into());
            keys.insert(name.clone(), Value::Object(entry));
        }
        for name in &self.taps {
            let mut entry = Map::new();
            entry.insert("vk".into(), self.vks.get(name).copied().unwrap_or(0).into());
            entry.insert("hold".into(), "tap".into());
            keys.insert(name.clone(), Value::Object(entry));
        }
        let mut out = Map::new();
        out.insert("version".into(), 2.into());
        out.insert("origin".into(), self.origin.clone().into());
        out.insert("game".into(), self.game.clone().into());
        if let Some(handle) = self.game_hwnd {
            out.insert("game_hwnd".into(), (handle as u64).into());
        }
        out.insert(
            "game_resolution".into(),
            Value::Array(
                self.game_resolution
                    .iter()
                    .map(|value| Value::from(*value))
                    .collect(),
            ),
        );
        out.insert(
            "default_mouse_turn".into(),
            self.default_mouse_turn.into(),
        );
        out.insert("samples".into(), self.samples.into());
        if let Some(sensitivity) = &self.mouse_sensitivity {
            let mut fields = Map::new();
            fields.insert("samples".into(), (sensitivity.samples as u64).into());
            fields.insert(
                "median_pixels_per_step".into(),
                sensitivity.median_pixels_per_step.into(),
            );
            fields.insert(
                "max_pixels_per_step".into(),
                sensitivity.max_pixels_per_step.into(),
            );
            out.insert("mouse_sensitivity".into(), Value::Object(fields));
        }
        out.insert("keys".into(), Value::Object(keys));
        out.insert(
            "mouse_buttons".into(),
            Value::Array(self.mouse_buttons.iter().cloned().map(Value::from).collect()),
        );
        Value::Object(out)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<String> {
        let target = path.as_ref().to_string_lossy().to_string();
        if let Some(parent) = path.as_ref().parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(&self.to_json())?;
        let temporary = format!("{target}.tmp");
        std::fs::write(&temporary, text)?;
        std::fs::rename(&temporary, &target)?;
        Ok(target)
    }
}

// =============================================================================
// The factored action space
// =============================================================================

/// One momentary button the bot may press.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tap {
    pub label: String,
    pub vk: u32,
}

#[derive(Debug, Clone)]
pub struct ActionSpace {
    pub hold_names: Vec<String>,
    pub hold_vks: Vec<u32>,
    pub directions: Vec<(String, (i32, i32))>,
    pub speed_levels: Vec<f32>,
    pub taps: Vec<Tap>,
    pub max_held: usize,
    pub mouse_enabled: bool,
}

impl ActionSpace {
    /// Derive the action space from a keymap.
    ///
    /// The keymap supplies the keys and the base mouse step; the speed levels
    /// are multipliers on that base, and the bot owns the base.
    pub fn build(
        keymap: &Keymap,
        max_held: usize,
        speed_levels: &[f32],
        mouse_enabled: bool,
        include_mouse_taps: bool,
    ) -> Self {
        let mut hold_names = keymap.holds.clone();
        hold_names.sort_by(|a, b| sort_key(a).cmp(&sort_key(b)));
        let hold_vks: Vec<u32> = hold_names
            .iter()
            .filter_map(|name| keymap.vks.get(name).copied())
            .filter(|code| *code != 0)
            .collect();

        let directions: Vec<(String, (i32, i32))> = if mouse_enabled {
            DIRECTIONS
                .iter()
                .map(|(name, delta)| (name.to_string(), *delta))
                .collect()
        } else {
            vec![(DIRECTIONS[0].0.to_string(), DIRECTIONS[0].1)]
        };
        let levels: Vec<f32> = if speed_levels.is_empty() {
            vec![1.0]
        } else {
            speed_levels.to_vec()
        };

        let mut taps = vec![Tap {
            label: "nothing".to_string(),
            vk: 0,
        }];
        let mut tap_names = keymap.taps.clone();
        tap_names.sort_by(|a, b| sort_key(a).cmp(&sort_key(b)));
        for name in tap_names {
            if let Some(code) = keymap.vks.get(&name).copied()
                && code != 0
            {
                taps.push(Tap {
                    label: format!("tap:{name}"),
                    vk: code,
                });
            }
        }
        if include_mouse_taps {
            for (button, code) in MOUSE_BUTTON_VK {
                if !hold_vks.contains(&code) && !taps.iter().any(|tap| tap.vk == code) {
                    taps.push(Tap {
                        label: format!("click:{button}"),
                        vk: code,
                    });
                }
            }
        }

        Self {
            hold_names,
            hold_vks,
            directions,
            speed_levels: levels,
            taps,
            max_held: max_held.max(1),
            mouse_enabled,
        }
    }

    /// One toggle choice per key, plus "toggle nothing".
    pub fn n_hold(&self) -> usize {
        self.hold_vks.len() + 1
    }

    pub fn n_turn(&self) -> usize {
        self.directions.len().max(1)
    }

    pub fn n_speed(&self) -> usize {
        self.speed_levels.len().max(1)
    }

    pub fn n_tap(&self) -> usize {
        self.taps.len().max(1)
    }

    pub fn head_sizes(&self) -> [usize; 4] {
        [self.n_hold(), self.n_turn(), self.n_speed(), self.n_tap()]
    }

    /// Width of the one-hot action description fed to the reward.
    pub fn action_vector_size(&self) -> usize {
        self.n_hold() + self.n_turn() + self.n_speed() + self.n_tap()
    }

    /// Flat one-hot description of a decision.
    pub fn action_vector(
        &self,
        held_index: usize,
        turn_index: usize,
        speed_index: usize,
        tap_index: usize,
    ) -> Vec<f32> {
        let mut vector = vec![0.0f32; self.action_vector_size()];
        let clamp = |value: usize, limit: usize| value.min(limit.saturating_sub(1));
        vector[clamp(held_index, self.n_hold())] = 1.0;
        let mut offset = self.n_hold();
        vector[offset + clamp(turn_index, self.n_turn())] = 1.0;
        offset += self.n_turn();
        vector[offset + clamp(speed_index, self.n_speed())] = 1.0;
        offset += self.n_speed();
        vector[offset + clamp(tap_index, self.n_tap())] = 1.0;
        vector
    }

    /// Flat action description for the reward.
    ///
    /// The held mask is summarised by its lowest set key plus a count bit rather
    /// than by the whole mask: the reward only needs to tell decisions apart, and
    /// the difference between "hold W" and "hold W and shift" is one bit.
    pub fn describe_action(
        &self,
        held: &[f32],
        turn_index: usize,
        speed_index: usize,
        tap_index: usize,
    ) -> Vec<f32> {
        let first = held
            .iter()
            .position(|bit| *bit > 0.5)
            .map(|index| index + 1)
            .unwrap_or(0);
        self.action_vector(first, turn_index, speed_index, tap_index)
    }

    pub fn direction(&self, index: usize) -> (&str, (i32, i32)) {
        match self.directions.get(index) {
            Some((name, delta)) => (name.as_str(), *delta),
            None => ("no turn", (0, 0)),
        }
    }

    pub fn speed_level(&self, index: usize) -> f32 {
        self.speed_levels.get(index).copied().unwrap_or(1.0)
    }

    /// The mouse delta for a `(direction, speed)` decision at this base step.
    ///
    /// `base_pixels` comes from the model, not from a constant here: the bot
    /// moves its own base step, and this is only the multiplication that turns a
    /// decision into pixels.
    pub fn turn_delta(
        &self,
        direction_index: usize,
        speed_index: usize,
        base_pixels: i32,
    ) -> (i32, i32) {
        let (_name, (dx, dy)) = self.direction(direction_index);
        if dx == 0 && dy == 0 {
            return (0, 0);
        }
        let level = self.speed_level(speed_index);
        let step = ((base_pixels.max(1) as f32) * level).round() as i32;
        (dx * step.max(1), dy * step.max(1))
    }

    pub fn is_noop(
        &self,
        held_index: usize,
        turn_index: usize,
        speed_index: usize,
        tap_index: usize,
    ) -> bool {
        let (_name, delta) = self.direction(turn_index);
        let _ = speed_index;
        held_index == 0 && delta == (0, 0) && tap_index == 0
    }

    /// Readable name for a decision, coarse enough to be a useful histogram.
    pub fn label(
        &self,
        held_index: usize,
        turn_index: usize,
        speed_index: usize,
        tap_index: usize,
    ) -> String {
        let mut parts: Vec<String> = Vec::new();
        if held_index >= 1
            && held_index <= self.hold_names.len()
            && let Some(name) = self.hold_names.get(held_index - 1)
        {
            parts.push(name.clone());
        }
        let (name, delta) = self.direction(turn_index);
        if delta != (0, 0) {
            parts.push(format!("turn {name} x{}", self.speed_level(speed_index)));
        }
        if tap_index >= 1
            && let Some(tap) = self.taps.get(tap_index)
        {
            parts.push(tap.label.clone());
        }
        if parts.is_empty() {
            "noop".to_string()
        } else {
            parts.join("+")
        }
    }

    pub fn describe(&self) -> String {
        let speeds = self
            .speed_levels
            .iter()
            .map(|s| format!("{s}"))
            .collect::<Vec<_>>()
            .join("/");
        format!(
            "{} hold key(s) (<= {} at once), {} turn direction(s) x {} speed(s) \
             x[{}], {} tap/click choice(s)  ->  {} combinations (the bot's own \
             base step sets the pixels)",
            self.hold_vks.len(),
            self.max_held,
            self.n_turn(),
            self.n_speed(),
            speeds,
            self.n_tap(),
            self.n_hold() * self.n_turn() * self.n_speed() * self.n_tap()
        )
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert(
            "hold_names".into(),
            Value::Array(self.hold_names.iter().cloned().map(Value::from).collect()),
        );
        out.insert(
            "hold_vks".into(),
            Value::Array(self.hold_vks.iter().map(|v| Value::from(*v)).collect()),
        );
        out.insert(
            "directions".into(),
            Value::Array(
                self.directions
                    .iter()
                    .map(|(name, delta)| {
                        Value::Array(vec![
                            Value::from(name.clone()),
                            Value::Array(vec![Value::from(delta.0), Value::from(delta.1)]),
                        ])
                    })
                    .collect(),
            ),
        );
        out.insert(
            "speed_levels".into(),
            Value::Array(
                self.speed_levels
                    .iter()
                    .map(|level| Value::from(*level as f64))
                    .collect(),
            ),
        );
        out.insert(
            "taps".into(),
            Value::Array(
                self.taps
                    .iter()
                    .map(|tap| {
                        let mut entry = Map::new();
                        entry.insert("label".into(), tap.label.clone().into());
                        entry.insert("vk".into(), tap.vk.into());
                        Value::Object(entry)
                    })
                    .collect(),
            ),
        );
        out.insert("max_held".into(), self.max_held.into());
        out.insert("mouse_enabled".into(), self.mouse_enabled.into());
        Value::Object(out)
    }
}

// =============================================================================
// Hotkey specifications
// =============================================================================

pub const MOD_ALT: u32 = 0x0001;
pub const MOD_CONTROL: u32 = 0x0002;
pub const MOD_SHIFT: u32 = 0x0004;
pub const MOD_WIN: u32 = 0x0008;
/// Windows-only: do not repeat while the key is held down.
pub const MOD_NOREPEAT: u32 = 0x4000;

/// The keys a hotkey specification may name.
///
/// Deliberately narrower than [`key_code`]: a global hotkey cannot be a mouse
/// button, and saying so here is better than registering something that silently
/// never fires.
fn hotkey_key(name: &str) -> Option<u32> {
    let upper = name.trim().to_uppercase();
    let named: [(&str, u32); 14] = [
        ("SPACE", 0x20),
        ("ESC", 0x1B),
        ("TAB", 0x09),
        ("ENTER", 0x0D),
        ("INSERT", 0x2D),
        ("DELETE", 0x2E),
        ("HOME", 0x24),
        ("END", 0x23),
        ("PAGEUP", 0x21),
        ("PAGEDOWN", 0x22),
        ("UP", 0x26),
        ("DOWN", 0x28),
        ("LEFT", 0x25),
        ("RIGHT", 0x27),
    ];
    if let Some((_, code)) = named.iter().find(|(key, _)| *key == upper) {
        return Some(*code);
    }
    let bytes = upper.as_bytes();
    if bytes.len() == 1 {
        let byte = bytes[0];
        if byte.is_ascii_uppercase() || byte.is_ascii_digit() {
            return Some(byte as u32);
        }
    }
    if let Some(rest) = upper.strip_prefix('F')
        && let Ok(number) = rest.parse::<u32>()
        && (1..=24).contains(&number)
    {
        return Some(0x6F + number);
    }
    None
}

/// `"ctrl+shift+f9"` -> `(modifiers, virtual key)`.
///
/// Case-insensitive, whitespace-tolerant, and empty segments are ignored, so
/// `"ctrl++f8"` and `"+f8"` both work.
pub fn parse_hotkey(spec: &str) -> Result<(u32, u32), String> {
    let parts: Vec<String> = spec
        .split('+')
        .map(|part| part.trim().to_lowercase())
        .filter(|part| !part.is_empty())
        .collect();
    if parts.is_empty() {
        return Err("empty hotkey".to_string());
    }
    let key = parts.last().cloned().unwrap_or_default();
    let mut modifiers = 0u32;
    for part in &parts[..parts.len() - 1] {
        modifiers |= match part.as_str() {
            "ctrl" | "control" => MOD_CONTROL,
            "alt" => MOD_ALT,
            "shift" => MOD_SHIFT,
            "win" | "super" => MOD_WIN,
            _ => return Err(format!("unknown modifier '{part}' in '{spec}'")),
        };
    }
    match hotkey_key(&key) {
        Some(code) => Ok((modifiers, code)),
        None => Err(format!("unknown key '{key}' in '{spec}'")),
    }
}

// =============================================================================
// Turning a recording into a keymap
// =============================================================================

/// What the recorder saw of one key.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct KeyStats {
    pub count: u64,
    pub total_held: f64,
    pub longest_hold: f64,
}

/// Everything a calibration session recorded.
#[derive(Debug, Clone, Default)]
pub struct Recording {
    pub stats: BTreeMap<String, KeyStats>,
    /// Euclidean length of every non-zero mouse move, per hook event.
    pub mouse_pixels: Vec<f64>,
    pub events: u64,
    /// The window the recording was aimed at, for the keymap's own record.
    pub target_handle: Option<usize>,
    pub target_title: String,
    pub target_client: (i32, i32),
}

impl Recording {
    /// Is this a key the user *holds*, rather than taps?
    ///
    /// The mean hold, never the median or the longest: a key held for half a
    /// second on every press is a movement key, and one that is tapped is a
    /// menu key, and the average is what separates them.
    pub fn is_hold(&self, name: &str, min_hold: f64) -> bool {
        match self.stats.get(name) {
            Some(stats) if stats.count > 0 => stats.total_held / stats.count as f64 >= min_hold,
            _ => false,
        }
    }

    /// Turn the recording into a keymap.
    pub fn build_keymap(&self, existing: Option<&Keymap>, min_hold: f64) -> Keymap {
        let mut keymap = match existing {
            Some(keymap) => keymap.clone(),
            None => Keymap {
                origin: "default".to_string(),
                default_mouse_turn: 20,
                ..Keymap::empty()
            },
        };
        keymap.origin = "calibrated".to_string();

        // Alphabetical, as the Python sorts the stats dictionary.
        for (name, _stats) in &self.stats {
            if name.starts_with("MOUSE_") {
                if !keymap.mouse_buttons.contains(name) {
                    keymap.mouse_buttons.push(name.clone());
                }
                continue;
            }
            if key_code(name).is_none() {
                continue;
            }
            let hold = self.is_hold(name, min_hold);
            keymap.add(name, hold, None);
        }
        keymap.mouse_buttons.sort();
        keymap.sort_keys();

        if !self.mouse_pixels.is_empty() {
            let mut values = self.mouse_pixels.clone();
            values.sort_by(|a, b| a.total_cmp(b));
            // The upper median: element `n / 2`, not the mean of the two
            // middles. Faithful, and it matters at small sample counts.
            let typical = values[values.len() / 2];
            keymap.default_mouse_turn = (round_half_to_even(typical) as i32).max(1);
            keymap.mouse_sensitivity = Some(MouseSensitivity {
                samples: values.len() as u64,
                median_pixels_per_step: typical,
                max_pixels_per_step: values[values.len() - 1],
            });
        }
        keymap.samples = self.events as u32;
        if let Some(handle) = self.target_handle {
            keymap.game_hwnd = Some(handle);
            keymap.game.clone_from(&self.target_title);
            keymap.game_resolution = vec![self.target_client.0, self.target_client.1];
        }
        keymap
    }

    /// The table a user reads after a recording.
    pub fn report(&self, min_hold: f64) -> String {
        if self.stats.is_empty() && self.mouse_pixels.is_empty() {
            return "  (nothing recorded)".to_string();
        }
        let mut rows: Vec<(&String, &KeyStats)> = self.stats.iter().collect();
        rows.sort_by(|a, b| b.1.count.cmp(&a.1.count));
        let mut lines: Vec<String> = rows
            .iter()
            .map(|(name, stats)| {
                let mean = stats.total_held / stats.count.max(1) as f64;
                format!(
                    "  {name:<12} {}  pressed {:4}x  mean {:6.0} ms  longest {:6.0} ms",
                    if self.is_hold(name, min_hold) {
                        "hold"
                    } else {
                        "tap "
                    },
                    stats.count,
                    mean * 1000.0,
                    stats.longest_hold * 1000.0
                )
            })
            .collect();
        if !self.mouse_pixels.is_empty() {
            let mut values = self.mouse_pixels.clone();
            values.sort_by(|a, b| a.total_cmp(b));
            lines.push(format!(
                "  {:<12}       {:4} events  median {:5.1} px  max {:5.1} px",
                "MOUSE MOVE",
                values.len(),
                values[values.len() / 2],
                values[values.len() - 1]
            ));
        }
        lines.join("\n")
    }
}

/// Python's `round`: half to even, which is not what `f64::round` does.
fn round_half_to_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if (fraction - 0.5).abs() < f64::EPSILON {
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        value.round()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hotkeys_parse_the_way_the_python_parses_them() {
        assert_eq!(parse_hotkey("f8").unwrap(), (0, 0x77));
        assert_eq!(parse_hotkey("F8").unwrap(), (0, 0x77));
        assert_eq!(parse_hotkey(" ctrl + shift + F9 ").unwrap(), (0x0006, 0x78));
        assert_eq!(parse_hotkey("+f8").unwrap(), (0, 0x77));
        assert_eq!(parse_hotkey("ctrl++f8").unwrap(), (0x0002, 0x77));
        // The message uses the original spec, not the trimmed parts.
        assert!(parse_hotkey("ctrl+nope").unwrap_err().contains("nope"));
        assert!(parse_hotkey("hyper+f8").unwrap_err().contains("hyper"));
        assert!(parse_hotkey("").is_err());
        // A hotkey cannot be a mouse button, even though a keymap can.
        assert!(parse_hotkey("mouse_left").is_err());
    }

    #[test]
    fn a_recording_becomes_a_keymap_the_way_the_python_builds_one() {
        let mut recording = Recording {
            events: 12,
            target_handle: Some(4242),
            target_title: "Some Game".to_string(),
            target_client: (1280, 720),
            ..Default::default()
        };
        // A key held for half a second on every press, and one tapped.
        recording.stats.insert(
            "W".to_string(),
            KeyStats {
                count: 4,
                total_held: 2.0,
                longest_hold: 0.7,
            },
        );
        recording.stats.insert(
            "E".to_string(),
            KeyStats {
                count: 3,
                total_held: 0.12,
                longest_hold: 0.05,
            },
        );
        recording.stats.insert(
            "MOUSE_LEFT".to_string(),
            KeyStats {
                count: 5,
                total_held: 1.0,
                longest_hold: 0.3,
            },
        );
        recording.mouse_pixels = vec![1.0, 2.0, 30.0];

        let keymap = recording.build_keymap(None, 0.12);
        assert_eq!(keymap.origin, "calibrated");
        assert!(keymap.holds.contains(&"W".to_string()));
        assert!(keymap.taps.contains(&"E".to_string()));
        assert!(keymap.mouse_buttons.contains(&"MOUSE_LEFT".to_string()));
        assert_eq!(keymap.samples, 12);
        assert_eq!(keymap.game_hwnd, Some(4242));
        assert_eq!(keymap.game, "Some Game");
        assert_eq!(keymap.game_resolution, vec![1280, 720]);
        // Upper median of three sorted values is the middle one.
        assert_eq!(keymap.default_mouse_turn, 2);
        let sensitivity = keymap.mouse_sensitivity.expect("the measurement is kept");
        assert_eq!(sensitivity.samples, 3);
        assert!((sensitivity.max_pixels_per_step - 30.0).abs() < 1e-9);
    }

    #[test]
    fn the_median_is_the_upper_one_and_rounds_half_to_even() {
        let mut recording = Recording::default();
        // Even count: the upper middle, not the average of the two.
        recording.mouse_pixels = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(recording.build_keymap(None, 0.12).default_mouse_turn, 3);
        // Half to even: 2.5 -> 2, 3.5 -> 4.
        recording.mouse_pixels = vec![2.5];
        assert_eq!(recording.build_keymap(None, 0.12).default_mouse_turn, 2);
        recording.mouse_pixels = vec![3.5];
        assert_eq!(recording.build_keymap(None, 0.12).default_mouse_turn, 4);
        // And never below one pixel.
        recording.mouse_pixels = vec![0.2];
        assert_eq!(recording.build_keymap(None, 0.12).default_mouse_turn, 1);
    }

    #[test]
    fn an_empty_recording_produces_no_holds() {
        let recording = Recording::default();
        let keymap = recording.build_keymap(None, 0.12);
        assert!(keymap.holds.is_empty());
        assert!(keymap.vks.is_empty());
        assert_eq!(recording.report(0.12), "  (nothing recorded)");
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn the_default_layout_matches_the_python() {
        let keymap = Keymap::default_layout();
        assert_eq!(keymap.default_mouse_turn, 20);
        assert_eq!(keymap.holds.len(), 9, "{:?}", keymap.holds);
        assert_eq!(keymap.taps.len(), 6);
        // Ordered by the priority list, not alphabetically: W A S D first.
        assert_eq!(&keymap.holds[..4], &["W", "A", "S", "D"]);
    }

    #[test]
    fn action_space_shape_follows_the_keymap() {
        let keymap = Keymap::default_layout();
        let space = ActionSpace::build(&keymap, 4, &[0.25, 0.5, 1.0, 2.0, 4.0], true, true);
        assert_eq!(space.hold_vks.len(), 9);
        assert_eq!(space.head_sizes(), [10, 5, 5, 8]);
        // "nothing" plus the six tap keys, plus the one mouse button that is not
        // already a hold key: the default layout holds both main buttons, so
        // only the middle click is added.
        assert_eq!(space.n_tap(), 8);
        assert_eq!(space.action_vector_size(), 10 + 5 + 5 + 8);
        assert_eq!(space.turn_delta(1, 2, 20), (-20, 0));
        assert_eq!(space.turn_delta(0, 4, 20), (0, 0));
        assert_eq!(space.turn_delta(3, 0, 20), (0, -5));
    }

    #[test]
    fn key_names_round_trip() {
        for name in ["W", "NUMPAD7", "F8", "SPACE", "MOUSE_LEFT", "1"] {
            let code = key_code(name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(key_name(code, false), name);
        }
    }

    #[test]
    fn a_keymap_round_trips_through_json() {
        let mut keymap = Keymap::default_layout();
        keymap.add("F", true, None);
        keymap.sort_keys();
        let json = keymap.to_json();
        let again = Keymap::from_json(&json);
        assert_eq!(again.holds, keymap.holds);
        assert_eq!(again.taps, keymap.taps);
        assert_eq!(again.vks, keymap.vks);
    }
}
