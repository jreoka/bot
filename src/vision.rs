//! What the bot sees, and the coarse fingerprint it reduces that to.
//! Ported from `capture.py`'s `gray_signature` and `novelty.py`'s
//! `_signature_key`.
//!
//! The fingerprint is the single most consequential choice in the whole reward:
//! it decides what counts as "a new situation". 256 cells at 16 levels is fine
//! enough that walking from one end of a corridor to the other is a hundred new
//! states rather than one, and coarse enough that a flickering texture is not.

/// One frame stack: `(channels, height, width)`, row-major, in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl Observation {
    pub fn zeros(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
            data: vec![0.0; channels * height * width],
        }
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.channels, self.height, self.width)
    }

    pub fn channel(&self, index: usize) -> &[f32] {
        let plane = self.height * self.width;
        &self.data[index * plane..(index + 1) * plane]
    }

    pub fn channel_mut(&mut self, index: usize) -> &mut [f32] {
        let plane = self.height * self.width;
        &mut self.data[index * plane..(index + 1) * plane]
    }

    /// Mean of the first `count` channels, as one `(height, width)` plane.
    ///
    /// The session fingerprints this rather than a derived channel, so the
    /// reward sees the picture and not the motion channel it computed from it.
    pub fn mean_of_channels(&self, count: usize) -> Vec<f32> {
        let count = count.max(1).min(self.channels);
        let plane = self.height * self.width;
        let mut out = vec![0.0f32; plane];
        for channel in 0..count {
            let source = self.channel(channel);
            for (target, value) in out.iter_mut().zip(source) {
                *target += value;
            }
        }
        let scale = 1.0 / count as f32;
        for value in out.iter_mut() {
            *value *= scale;
        }
        out
    }
}

/// Number of cells in the coarse fingerprint.
pub const GRID: usize = 256;
/// Levels each cell is quantised into.
pub const LEVELS: usize = 16;

/// Tiny fingerprint of a frame, used by novelty and reset detection.
///
/// Deliberately blunt: a sharp fingerprint makes every frame look like a new
/// situation, which turns count-based novelty into a flat per-step bonus and
/// makes reset detection fire constantly.
///
/// The Python resizes with `cv2.INTER_AREA`; this is the same area average
/// written out. `cv2` does it in fixed-point arithmetic, so the two can differ
/// by one part in 255 on a boundary pixel - which is not nothing for a
/// quantised fingerprint, but it changes only which side of a cell boundary a
/// frame lands on, never whether the fingerprint is meaningful. A run trains
/// from its own fingerprints either way.
pub fn gray_signature(frame: &[f32], height: usize, width: usize, size: usize) -> Vec<f32> {
    // The Python detects the range rather than assuming it: a fingerprint of
    // all zeros makes every frame look identical, which silently removes the
    // entire episodic reward.
    let scale = if frame.iter().copied().fold(0.0f32, f32::max) > 1.5 {
        255.0
    } else {
        1.0
    };
    let resized = area_resize(frame, height, width, size, size);
    resized.into_iter().map(|value| value / scale).collect()
}

/// Area-average resize, which is what `cv2.INTER_AREA` means when shrinking.
pub fn area_resize(
    frame: &[f32],
    height: usize,
    width: usize,
    out_height: usize,
    out_width: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_height * out_width];
    let row_scale = height as f32 / out_height as f32;
    let column_scale = width as f32 / out_width as f32;
    for out_row in 0..out_height {
        let top = out_row as f32 * row_scale;
        let bottom = (out_row + 1) as f32 * row_scale;
        let first_row = top.floor() as usize;
        let last_row = (bottom.ceil() as usize).min(height);
        for out_column in 0..out_width {
            let left = out_column as f32 * column_scale;
            let right = (out_column + 1) as f32 * column_scale;
            let first_column = left.floor() as usize;
            let last_column = (right.ceil() as usize).min(width);
            let mut total = 0.0f32;
            let mut weight = 0.0f32;
            for row in first_row..last_row {
                let row_overlap = (bottom.min(row as f32 + 1.0) - top.max(row as f32)).max(0.0);
                if row_overlap <= 0.0 {
                    continue;
                }
                for column in first_column..last_column {
                    let column_overlap =
                        (right.min(column as f32 + 1.0) - left.max(column as f32)).max(0.0);
                    if column_overlap <= 0.0 {
                        continue;
                    }
                    let area = row_overlap * column_overlap;
                    total += frame[row * width + column] * area;
                    weight += area;
                }
            }
            out[out_row * out_width + out_column] = if weight > 0.0 {
                total / weight
            } else {
                0.0
            };
        }
    }
    out
}

/// Hash a frame fingerprint to an integer.
///
/// The Python uses the built-in `hash` of the packed bits, which is salted per
/// process, so its keys are not stable across runs either - only equality
/// within a run matters, and this is the same thing with a fixed hash so a
/// Rust run is reproducible.
pub fn signature_key(signature: &[f32]) -> u64 {
    let columns = (signature.len() / GRID).max(1);
    let mut cells = [0u8; GRID];
    for (cell, slot) in cells.iter_mut().enumerate() {
        let start = cell * columns;
        let end = ((cell + 1) * columns).min(signature.len());
        if start >= end {
            continue;
        }
        let mean: f32 = signature[start..end].iter().sum::<f32>() / (end - start) as f32;
        let quantised = ((mean * LEVELS as f32) as i64).clamp(0, LEVELS as i64 - 1) as u8;
        *slot = quantised;
    }
    // Four bits per cell, most significant first, packed the way `np.packbits`
    // packs them - so the bytes hashed here are the bytes the Python hashes.
    let mut packed = [0u8; GRID / 2];
    for (index, cell) in cells.iter().enumerate() {
        let byte = index / 2;
        if index % 2 == 0 {
            packed[byte] |= cell << 4;
        } else {
            packed[byte] |= cell;
        }
    }
    fnv1a(&packed)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    // A final mix, so that fingerprints differing in the last byte do not land
    // in adjacent buckets of any table that indexes by the low bits.
    hash ^= hash >> 32;
    hash = hash.wrapping_mul(0xd6e8feb86659fd93);
    hash ^ (hash >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_resize_averages_the_block() {
        // 4x4 all ones, halved: every output pixel is exactly 1.
        let frame = vec![1.0f32; 16];
        let out = area_resize(&frame, 4, 4, 2, 2);
        assert_eq!(out, vec![1.0, 1.0, 1.0, 1.0]);
        // A single bright pixel spreads over the cells that cover it.
        let mut frame = vec![0.0f32; 16];
        frame[0] = 4.0;
        let out = area_resize(&frame, 4, 4, 2, 2);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn identical_frames_hash_the_same_and_different_frames_do_not() {
        let frame = vec![0.5f32; 48 * 48];
        let first = gray_signature(&frame, 48, 48, 32);
        let second = gray_signature(&frame, 48, 48, 32);
        assert_eq!(signature_key(&first), signature_key(&second));

        let mut other = frame.clone();
        for value in other.iter_mut().take(48 * 48 / 2) {
            *value = 0.0;
        }
        let different = gray_signature(&other, 48, 48, 32);
        assert_ne!(signature_key(&first), signature_key(&different));
    }

    #[test]
    fn a_fingerprint_is_the_size_the_quantiser_expects() {
        let frame = vec![0.25f32; 32 * 32];
        let signature = gray_signature(&frame, 32, 32, 32);
        assert_eq!(signature.len(), 1024);
        // Two cells per byte, so 256 cells at four bits each.
        assert_eq!(GRID / 2, 128);
    }
}
