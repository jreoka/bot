//! Checkpoints: what a run writes so that stopping is not losing. Ported from
//! `botcore/runtime.py`'s `CheckpointManager`.
//!
//! A checkpoint is two files with one stem: the weights as a safetensors file,
//! and everything that is not a tensor in that file's own `__metadata__` block.
//! One file rather than two for the state, because a checkpoint that has the
//! weights but not the step counter it belongs to is worse than no checkpoint:
//! it resumes a trained model at step zero.
//!
//! Naming is the Python's, because the two should be able to sit in the same
//! directory and be recognisable to each other:
//!
//! ```text
//! checkpoint_000000001_step000001024_20260920_030611.safetensors
//! last.safetensors        the rolling copy, updated after every save
//! index.json              what was kept, and why each one was written
//! ```
//!
//! Every write goes through a temporary file and a rename, so an interrupted
//! save cannot destroy a good checkpoint.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Bumped whenever a checkpoint stops being loadable, so an incompatible file is
/// refused rather than half-loaded.
pub const CHECKPOINT_VERSION: u32 = 1;

/// Everything about a run that is not a tensor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionMeta {
    pub version: u32,
    pub arch: String,
    pub reward_version: u32,
    pub step: u64,
    pub episodes: u64,
    pub seconds: f64,
    /// The bot's own mouse step: part of what it has learned, since the session
    /// moves it while it plays.
    pub mouse_step: f32,
    pub head_sizes: Vec<usize>,
    pub observation_shape: Vec<usize>,
    /// The intrinsic reward's running estimates, which the novelty term is
    /// measured against. Restarting them from scratch after a resume would make
    /// the first minutes of the new run a different reward.
    pub error_mean: f32,
    pub error_var: f32,
    pub error_count: u64,
    pub ambient: f32,
    pub reward_steps: u64,
    pub seed: u64,
}

impl SessionMeta {
    /// Serialise into the metadata map a safetensors header carries.
    pub fn to_metadata(&self) -> anyhow::Result<HashMap<String, String>> {
        let mut map = HashMap::new();
        map.insert("bot1".to_string(), serde_json::to_string(self)?);
        Ok(map)
    }

    pub fn from_metadata(metadata: &HashMap<String, String>) -> anyhow::Result<Self> {
        let text = metadata
            .get("bot1")
            .ok_or_else(|| anyhow::anyhow!("no bot1 metadata in this file"))?;
        Ok(serde_json::from_str(text)?)
    }

    /// Refuse a checkpoint that does not belong to this build, or to this action
    /// space, rather than loading half of it.
    pub fn validate(
        &self,
        arch: &str,
        reward_version: u32,
        head_sizes: &[usize],
        observation_shape: &[usize],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.version == CHECKPOINT_VERSION,
            "checkpoint is version {}, this build writes {CHECKPOINT_VERSION}",
            self.version
        );
        anyhow::ensure!(
            self.arch == arch,
            "checkpoint architecture '{}' does not match this build's '{arch}'",
            self.arch
        );
        anyhow::ensure!(
            self.reward_version == reward_version,
            "checkpoint was trained against reward version {}, this build uses \
             {reward_version} - resuming it would continue a policy optimised for \
             something else",
            self.reward_version
        );
        anyhow::ensure!(
            self.head_sizes == head_sizes,
            "checkpoint action space {:?} does not match the current {:?}",
            self.head_sizes,
            head_sizes
        );
        anyhow::ensure!(
            self.observation_shape == observation_shape,
            "checkpoint observation {:?} does not match the current {:?}",
            self.observation_shape,
            observation_shape
        );
        Ok(())
    }
}

/// One checkpoint on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub path: String,
    pub seq: u64,
    pub step: u64,
    pub mtime: f64,
    pub reason: String,
}

/// A checkpoint whose bytes have not been written yet: where they go, and what
/// has to be recorded once they are there.
///
/// The names are decided on the control loop (they need the sequence counter),
/// the bytes are written on another thread, and the renames happen back on the
/// control loop. Between the two, nothing outside points at the temporary files.
#[derive(Debug, Clone)]
pub struct CheckpointPlan {
    pub seq: u64,
    pub step: u64,
    pub reason: String,
    /// The final path, not yet existing.
    pub path: PathBuf,
    /// Where the numbered checkpoint's bytes go first.
    pub temporary: PathBuf,
    /// Where the rolling `last.safetensors` copy's bytes go first.
    pub last_temporary: PathBuf,
}

/// Owns a checkpoint directory: what is in it, what to write next, and what to
/// delete.
pub struct CheckpointManager {
    pub directory: PathBuf,
    pub keep: usize,
    entries: Vec<Entry>,
    seq: u64,
}

impl CheckpointManager {
    pub fn new(directory: impl AsRef<Path>, keep: usize) -> Self {
        let directory = directory.as_ref().to_path_buf();
        let _ = std::fs::create_dir_all(&directory);
        let entries = load_index(&directory);
        let seq = highest_sequence(&directory);
        Self {
            directory,
            keep: keep.max(1),
            entries,
            seq,
        }
    }

    pub fn last_path(&self) -> PathBuf {
        self.directory.join("last.safetensors")
    }

    fn index_path(&self) -> PathBuf {
        self.directory.join("index.json")
    }

    /// Write a checkpoint. Returns the path written.
    pub fn save(
        &mut self,
        tensors: &[(String, Vec<usize>, Vec<f32>)],
        meta: &SessionMeta,
        reason: &str,
    ) -> anyhow::Result<PathBuf> {
        let plan = self.plan_write(meta, reason);
        Self::stage_write(&plan, tensors, meta)?;
        Ok(self.finish_write(plan))
    }

    /// Name the next checkpoint: the sequence number, the path, and the two
    /// temporary files the bytes go into.
    ///
    /// Cheap and cheap on purpose - a counter and a timestamp - so it is safe to
    /// call from the control loop. [`CheckpointManager::stage_write`] is the part
    /// that belongs on a thread of its own.
    pub fn plan_write(&mut self, meta: &SessionMeta, reason: &str) -> CheckpointPlan {
        self.seq += 1;
        let stamp = timestamp();
        let path = self.directory.join(format!(
            "checkpoint_{:09}_step{:09}_{stamp}.safetensors",
            self.seq, meta.step
        ));
        CheckpointPlan {
            seq: self.seq,
            step: meta.step,
            reason: reason.to_string(),
            temporary: path.with_extension("safetensors.tmp"),
            last_temporary: self.directory.join("last.safetensors.tmp"),
            path,
        }
    }

    /// Write the checkpoint's bytes to the plan's temporary files.
    ///
    /// This is the expensive half - every weight serialised, twice - and it
    /// touches nothing on the manager: the same plan can be staged on another
    /// thread while the session carries on playing. Nothing is visible to
    /// `--resume` until [`CheckpointManager::finish_write`] renames it, so an
    /// interrupted stage cannot destroy a good checkpoint.
    pub fn stage_write(
        plan: &CheckpointPlan,
        tensors: &[(String, Vec<usize>, Vec<f32>)],
        meta: &SessionMeta,
    ) -> anyhow::Result<()> {
        let metadata = meta.to_metadata()?;
        crate::weights::write_safetensors_with_metadata(&plan.temporary, tensors, &metadata)?;
        // The rolling copy, for `--resume` to find quickly. A failure here is
        // reported and never fatal: the numbered checkpoint is still staged.
        match crate::weights::write_safetensors_with_metadata(
            &plan.last_temporary,
            tensors,
            &metadata,
        ) {
            Ok(()) => {}
            Err(error) => {
                println!("[Checkpoint] Rolling last.safetensors copy failed: {error}");
                let _ = std::fs::remove_file(&plan.last_temporary);
            }
        }
        Ok(())
    }

    /// The cheap half: make the staged files the real ones, and record what was
    /// written.
    pub fn finish_write(&mut self, plan: CheckpointPlan) -> PathBuf {
        let _ = std::fs::rename(&plan.temporary, &plan.path);
        if plan.last_temporary.exists() {
            let _ = std::fs::rename(&plan.last_temporary, self.last_path());
        }

        self.entries.push(Entry {
            path: plan.path.to_string_lossy().to_string(),
            seq: plan.seq,
            step: plan.step,
            mtime: now_seconds(),
            reason: plan.reason,
        });
        self.prune();
        self.save_index();
        plan.path
    }

    /// Keep the newest `keep`, delete the rest.
    fn prune(&mut self) {
        let mut unique: HashMap<String, Entry> = HashMap::new();
        for entry in self.entries.drain(..) {
            unique.insert(entry.path.clone(), entry);
        }
        let mut entries: Vec<Entry> = unique.into_values().collect();
        entries.sort_by(|a, b| {
            (a.seq, a.mtime)
                .partial_cmp(&(b.seq, b.mtime))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let doomed = entries.len().saturating_sub(self.keep);
        for entry in entries.iter().take(doomed) {
            if Path::new(&entry.path)
                .file_name()
                .map(|name| name != "last.safetensors")
                .unwrap_or(false)
                && Path::new(&entry.path).exists()
            {
                let _ = std::fs::remove_file(&entry.path);
            }
        }
        self.entries = entries.split_off(doomed);
    }

    fn save_index(&self) {
        let path = self.index_path();
        let temporary = path.with_extension("json.tmp");
        match serde_json::to_string_pretty(&self.entries) {
            Ok(text) => {
                if std::fs::write(&temporary, text).is_ok() {
                    let _ = std::fs::rename(&temporary, &path);
                }
            }
            Err(error) => println!("[Checkpoint] Could not write index: {error}"),
        }
    }

    /// The checkpoints still on disk, oldest first.
    pub fn list_kept(&self) -> Vec<Entry> {
        let mut kept: Vec<Entry> = self
            .entries
            .iter()
            .filter(|entry| Path::new(&entry.path).exists())
            .cloned()
            .collect();
        kept.sort_by_key(|entry| entry.seq);
        kept
    }

    /// What `--resume` should load: the rolling copy when it is there, else the
    /// newest numbered checkpoint.
    pub fn newest(&self) -> Option<PathBuf> {
        if self.last_path().exists() {
            return Some(self.last_path());
        }
        self.list_kept()
            .last()
            .map(|entry| PathBuf::from(&entry.path))
    }
}

fn highest_sequence(directory: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return 0;
    };
    let mut highest = 0u64;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("checkpoint_") || !name.ends_with(".safetensors") {
            continue;
        }
        if let Some(sequence) = name
            .split('_')
            .nth(1)
            .and_then(|value| value.parse::<u64>().ok())
        {
            highest = highest.max(sequence);
        }
    }
    highest
}

fn load_index(directory: &Path) -> Vec<Entry> {
    let Ok(text) = std::fs::read_to_string(directory.join("index.json")) else {
        return Vec::new();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

fn now_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

/// `YYYYMMDD_HHMMSS` in local time, as the Python's filename stamp is.
fn timestamp() -> String {
    // No calendar library is pulled in for this: the seconds since the epoch are
    // converted by hand, in UTC, and the stamp is only ever read by a human
    // sorting filenames.
    let seconds = now_seconds() as i64;
    let days = seconds.div_euclid(86_400);
    let time_of_day = seconds.rem_euclid(86_400);
    let (hour, minute, second) = (
        time_of_day / 3600,
        (time_of_day % 3600) / 60,
        time_of_day % 60,
    );
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}{month:02}{day:02}_{hour:02}{minute:02}{second:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(step: u64) -> SessionMeta {
        SessionMeta {
            version: CHECKPOINT_VERSION,
            arch: crate::config::ARCH_NAME.to_string(),
            reward_version: crate::config::REWARD_VERSION,
            step,
            episodes: 3,
            seconds: 12.5,
            mouse_step: 20.0,
            head_sizes: vec![10, 5, 5, 8],
            observation_shape: vec![5, 96, 96],
            error_mean: 0.5,
            error_var: 0.25,
            error_count: 40,
            ambient: 0.01,
            reward_steps: 900,
            seed: 0,
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bot1-checkpoint-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_checkpoint_round_trips_its_metadata() {
        let dir = temp_dir("meta");
        let mut manager = CheckpointManager::new(&dir, 3);
        let tensors = vec![("w".to_string(), vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])];
        let path = manager.save(&tensors, &meta(1024), "periodic").unwrap();
        assert!(path.exists());

        let weights = crate::weights::Weights::open(&path).unwrap();
        let read = SessionMeta::from_metadata(&weights.metadata()).unwrap();
        assert_eq!(read, meta(1024));
        assert_eq!(weights.f32("w").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn only_the_newest_checkpoints_are_kept() {
        let dir = temp_dir("prune");
        let mut manager = CheckpointManager::new(&dir, 2);
        let tensors = vec![("w".to_string(), vec![1], vec![1.0])];
        for step in [100u64, 200, 300, 400] {
            manager.save(&tensors, &meta(step), "periodic").unwrap();
        }
        let kept = manager.list_kept();
        assert_eq!(kept.len(), 2, "{kept:?}");
        assert_eq!(kept[0].step, 300);
        assert_eq!(kept[1].step, 400);
        // The rolling copy survives every prune, because `--resume` uses it.
        assert!(manager.last_path().exists());
    }

    #[test]
    fn resuming_finds_the_rolling_copy_first() {
        let dir = temp_dir("newest");
        let mut manager = CheckpointManager::new(&dir, 3);
        assert_eq!(manager.newest(), None);
        let tensors = vec![("w".to_string(), vec![1], vec![1.0])];
        let numbered = manager.save(&tensors, &meta(64), "periodic").unwrap();
        assert_eq!(manager.newest(), Some(manager.last_path()));
        std::fs::remove_file(manager.last_path()).unwrap();
        assert_eq!(manager.newest(), Some(numbered));
    }

    #[test]
    fn a_sequence_number_survives_a_restart() {
        let dir = temp_dir("seq");
        let tensors = vec![("w".to_string(), vec![1], vec![1.0])];
        {
            let mut manager = CheckpointManager::new(&dir, 3);
            manager.save(&tensors, &meta(10), "periodic").unwrap();
            manager.save(&tensors, &meta(20), "periodic").unwrap();
        }
        // A new manager over the same directory continues the numbering rather
        // than overwriting what is there.
        let mut manager = CheckpointManager::new(&dir, 3);
        let path = manager.save(&tensors, &meta(30), "periodic").unwrap();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with("checkpoint_000000003_"), "{name}");
    }

    #[test]
    fn an_incompatible_checkpoint_is_refused_with_a_reason() {
        let good = meta(1);
        good.validate(
            crate::config::ARCH_NAME,
            crate::config::REWARD_VERSION,
            &[10, 5, 5, 8],
            &[5, 96, 96],
        )
        .unwrap();

        // A different action space: the heads would not fit.
        let error = good
            .validate(
                crate::config::ARCH_NAME,
                crate::config::REWARD_VERSION,
                &[10, 5, 5, 4],
                &[5, 96, 96],
            )
            .unwrap_err();
        assert!(error.to_string().contains("action space"), "{error}");

        // A different reward: resuming would continue a policy optimised for
        // something else, which is exactly how a run ends up babysitting a no-op.
        let error = good
            .validate(crate::config::ARCH_NAME, 99, &[10, 5, 5, 8], &[5, 96, 96])
            .unwrap_err();
        assert!(error.to_string().contains("reward version"), "{error}");

        // A different observation size.
        let error = good
            .validate(
                crate::config::ARCH_NAME,
                crate::config::REWARD_VERSION,
                &[10, 5, 5, 8],
                &[5, 48, 48],
            )
            .unwrap_err();
        assert!(error.to_string().contains("observation"), "{error}");
    }

    /// The split the async path relies on: the bytes are staged on their own
    /// thread, and the control loop only does the renames.
    #[test]
    fn a_checkpoint_can_be_written_in_two_halves() {
        let dir = temp_dir("split");
        let mut manager = CheckpointManager::new(&dir, 3);
        let tensors = vec![("w".to_string(), vec![2], vec![1.0, 2.0])];

        // Half one: name it, then stage the bytes - as another thread would.
        let plan = manager.plan_write(&meta(512), "periodic");
        assert!(!plan.path.exists(), "the numbered file is not there yet");
        CheckpointManager::stage_write(&plan, &tensors, &meta(512)).unwrap();
        assert!(plan.temporary.exists());
        assert!(plan.last_temporary.exists());
        assert!(
            manager.list_kept().is_empty(),
            "nothing is kept until it is finished"
        );
        // And what was staged is already a complete, loadable checkpoint.
        let staged = crate::weights::Weights::open(&plan.temporary).unwrap();
        assert_eq!(staged.f32("w").unwrap(), vec![1.0, 2.0]);
        assert_eq!(
            SessionMeta::from_metadata(&staged.metadata()).unwrap().step,
            512
        );

        // Half two: renames and the index, nothing else.
        let finished = manager.finish_write(plan);
        assert!(finished.exists());
        assert_eq!(manager.list_kept().len(), 1);
        assert!(manager.last_path().exists());
        assert_eq!(
            SessionMeta::from_metadata(
                &crate::weights::Weights::open(&finished).unwrap().metadata()
            )
            .unwrap()
            .step,
            512
        );
    }

    /// The sequence number is allocated at planning time, so a checkpoint staged
    /// but never finished cannot make the next one reuse its name.
    #[test]
    fn a_staged_checkpoint_does_not_hand_its_number_to_the_next_one() {
        let dir = temp_dir("stage-seq");
        let mut manager = CheckpointManager::new(&dir, 3);
        let abandoned = manager.plan_write(&meta(1), "periodic");
        let next = manager.plan_write(&meta(2), "periodic");
        assert_ne!(abandoned.seq, next.seq);
        assert_ne!(abandoned.path, next.path);
    }
}
