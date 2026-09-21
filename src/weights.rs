//! Reading and writing flat named-tensor files.
//!
//! The port has to prove it computes what the Python computes, so it has to be
//! able to read the Python's weights. This is a minimal safetensors reader
//! (hand-rolled rather than pulled in, so nothing about the format or its error
//! messages is out of our hands): an 8-byte little-endian header length, a JSON
//! header of `name -> {dtype, shape, data_offsets}`, then the buffer.
//!
//! Two layouts meet here and the difference is a real trap. PyTorch stores a
//! `Linear` weight as `(out_features, in_features)`; Burn stores it as
//! `(in_features, out_features)`. Every `Linear` weight read through
//! [`Weights::param_linear`] is transposed on the way in; embeddings, biases,
//! convolution weights and layer norms go through [`Weights::param`] unchanged.
//! The parity test is what keeps that honest.

use std::collections::HashMap;
use std::path::Path;

use burn::module::Param;
use burn::prelude::Backend;
use burn::tensor::{Int, Shape, Tensor, TensorData};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// A parsed named-tensor file, holding each tensor's bytes.
#[derive(Debug, Default)]
pub struct Weights {
    entries: HashMap<String, Entry>,
    buffer: Vec<u8>,
    metadata: HashMap<String, String>,
}

impl Weights {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        Self::from_bytes(&bytes).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
    }

    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        anyhow::ensure!(bytes.len() >= 8, "file is shorter than a header length");
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        anyhow::ensure!(
            bytes.len() >= 8 + header_len,
            "file is shorter than its own header says ({header_len} bytes)"
        );
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..8 + header_len])?;
        let metadata = header
            .get("__metadata__")
            .and_then(|value| value.as_object())
            .map(|fields| {
                fields
                    .iter()
                    .filter_map(|(key, value)| {
                        value.as_str().map(|text| (key.clone(), text.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let entries = header
            .into_iter()
            .filter(|(name, _)| name != "__metadata__")
            .map(|(name, value)| {
                let entry: Entry = serde_json::from_value(value)?;
                Ok((name, entry))
            })
            .collect::<anyhow::Result<HashMap<String, Entry>>>()?;
        Ok(Self {
            entries,
            buffer: bytes[8 + header_len..].to_vec(),
            metadata,
        })
    }

    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.entries.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// The format's `__metadata__` map, which a checkpoint uses for its
    /// counters and configuration.
    pub fn metadata(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }

    pub fn metadata_value(&self, key: &str) -> Option<String> {
        self.metadata.get(key).cloned()
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> anyhow::Result<&[usize]> {
        Ok(&self.entry(name)?.shape)
    }

    fn entry(&self, name: &str) -> anyhow::Result<&Entry> {
        self.entries
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("no tensor named '{name}'"))
    }

    fn raw(&self, name: &str) -> anyhow::Result<(&Entry, &[u8])> {
        let entry = self.entry(name)?;
        let [begin, end] = entry.data_offsets;
        anyhow::ensure!(
            end <= self.buffer.len(),
            "'{name}' claims bytes {begin}..{end} of a {}-byte buffer",
            self.buffer.len()
        );
        Ok((entry, &self.buffer[begin..end]))
    }

    /// Elements as `f32`, whatever they were stored as.
    pub fn f32(&self, name: &str) -> anyhow::Result<Vec<f32>> {
        let (entry, bytes) = self.raw(name)?;
        let count: usize = entry.shape.iter().product();
        let out = match entry.dtype.as_str() {
            "F32" => {
                anyhow::ensure!(bytes.len() == count * 4, "'{name}': byte count mismatch");
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            }
            "F64" => {
                anyhow::ensure!(bytes.len() == count * 8, "'{name}': byte count mismatch");
                bytes
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect()
            }
            "I64" => {
                anyhow::ensure!(bytes.len() == count * 8, "'{name}': byte count mismatch");
                bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect()
            }
            "U8" | "BOOL" => {
                anyhow::ensure!(bytes.len() == count, "'{name}': byte count mismatch");
                bytes.iter().map(|b| *b as f32).collect()
            }
            other => anyhow::bail!("'{name}' has unsupported dtype {other}"),
        };
        Ok(out)
    }

    /// Elements as `i64`, whatever they were stored as.
    pub fn i64(&self, name: &str) -> anyhow::Result<Vec<i64>> {
        Ok(self.f32(name)?.into_iter().map(|v| v as i64).collect())
    }

    /// A tensor in the file's own layout, checked against the shape it must have.
    pub fn tensor<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> anyhow::Result<Tensor<B, D>> {
        let shape = self.shape(name)?.to_vec();
        anyhow::ensure!(
            shape.len() == D,
            "'{name}' has {} dimensions, expected {D}",
            shape.len()
        );
        Ok(self.tensor_shaped(name, device, &shape)?)
    }

    /// A tensor in the file's own layout, checked against an expected shape.
    pub fn tensor_shaped<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
        expect: &[usize],
    ) -> anyhow::Result<Tensor<B, D>> {
        let shape = self.shape(name)?.to_vec();
        anyhow::ensure!(
            shape == expect,
            "'{name}' has shape {shape:?}, expected {expect:?}"
        );
        let values = self.f32(name)?;
        Ok(Tensor::from_data(
            TensorData::new(values, expect.to_vec()),
            device,
        ))
    }

    /// An integer tensor in the file's own layout.
    pub fn tensor_int<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> anyhow::Result<Tensor<B, D, Int>> {
        let shape = self.shape(name)?.to_vec();
        anyhow::ensure!(
            shape.len() == D,
            "'{name}' has {} dimensions, expected {D}",
            shape.len()
        );
        let values = self.i64(name)?;
        Ok(Tensor::from_data(TensorData::new(values, shape), device))
    }

    /// A parameter in the file's own layout.
    pub fn param<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
        expect: &[usize],
    ) -> anyhow::Result<Param<Tensor<B, D>>> {
        Ok(Param::from_tensor(
            self.tensor_shaped::<B, D>(name, device, expect)?,
        ))
    }

    /// A 2-D parameter stored PyTorch-style, i.e. `(out, in)`, transposed into
    /// Burn's `(in, out)`. Used for every `Linear` weight in the model.
    pub fn param_linear<B: Backend>(
        &self,
        name: &str,
        device: &B::Device,
        expect: &[usize],
    ) -> anyhow::Result<Param<Tensor<B, 2>>> {
        let [rows, cols] = self.shape(name)? else {
            anyhow::bail!("'{name}' is not a matrix");
        };
        let (rows, cols) = (*rows, *cols);
        anyhow::ensure!(
            [cols, rows] == *expect,
            "'{name}' is {rows}x{cols}, which transposes to {}x{}, expected {:?}",
            cols,
            rows,
            expect
        );
        let values = self.f32(name)?;
        let mut transposed = vec![0.0f32; values.len()];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = values[r * cols + c];
            }
        }
        Ok(Param::from_tensor(Tensor::from_data(
            TensorData::new(transposed, Shape::new([cols, rows])),
            device,
        )))
    }

    /// A 2-D parameter whose trailing dimensions are flattened into the second
    /// axis, with no reordering.
    ///
    /// The patch-embedding convolution is stored by PyTorch as
    /// `(width, channels, patch, patch)`; this port runs it as one matmul over
    /// unrolled patches, so it wants `(width, channels * patch * patch)`. Since
    /// the file is row-major, that is a reinterpretation, not a copy — but the
    /// element counts have to agree, and this checks that they do.
    pub fn param_flattened<B: Backend>(
        &self,
        name: &str,
        device: &B::Device,
        expect: &[usize],
    ) -> anyhow::Result<Param<Tensor<B, 2>>> {
        let shape = self.shape(name)?.to_vec();
        anyhow::ensure!(
            shape.len() >= 2 && shape[0] == expect[0],
            "'{name}' has shape {shape:?}, expected a leading dimension of {}",
            expect[0]
        );
        let product: usize = shape[1..].iter().product();
        anyhow::ensure!(
            product == expect[1],
            "'{name}' has {product} trailing elements, expected {}",
            expect[1]
        );
        let values = self.f32(name)?;
        Ok(Param::from_tensor(Tensor::from_data(
            TensorData::new(values, expect.to_vec()),
            device,
        )))
    }

    /// A 2-D parameter whose trailing dimensions are flattened into the first
    /// axis, with the leading dimension becoming the second.
    ///
    /// The patch-embedding convolution is stored by PyTorch as
    /// `(width, channels, patch, patch)`. The port runs it as
    /// `patches (tokens, channels * patch * patch) @ weight`, so the weight has
    /// to arrive as `(channels * patch * patch, width)` — flattened *and*
    /// transposed. Doing it at load time costs one pass over 160 kB and saves a
    /// transpose on every control step, which is the trade this port wants.
    pub fn param_flattened_transposed<B: Backend>(
        &self,
        name: &str,
        device: &B::Device,
        expect: &[usize],
    ) -> anyhow::Result<Param<Tensor<B, 2>>> {
        let shape = self.shape(name)?.to_vec();
        anyhow::ensure!(
            shape.len() >= 2 && shape[0] == expect[1],
            "'{name}' has shape {shape:?}, expected a leading dimension of {}",
            expect[1]
        );
        let product: usize = shape[1..].iter().product();
        anyhow::ensure!(
            product == expect[0],
            "'{name}' has {product} trailing elements, expected {}",
            expect[0]
        );
        let values = self.f32(name)?;
        let leading = shape[0];
        let mut transposed = vec![0.0f32; values.len()];
        for row in 0..leading {
            for column in 0..product {
                transposed[column * leading + row] = values[row * product + column];
            }
        }
        Ok(Param::from_tensor(Tensor::from_data(
            TensorData::new(transposed, expect.to_vec()),
            device,
        )))
    }
}

/// Build the safetensors bytes for a set of named 1-D/2-D `f32` arrays.
///
/// Used by tests and by checkpoints; the Python writer produces the same layout
/// independently.
pub fn write_safetensors(
    path: impl AsRef<Path>,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
) -> anyhow::Result<()> {
    write_safetensors_with_metadata(path, tensors, &HashMap::new())
}

/// The same, with the format's `__metadata__` map carrying whatever strings the
/// caller wants alongside the tensors.
///
/// A checkpoint puts its counters and configuration in there rather than in a
/// second file, so a checkpoint cannot be half-copied: if the weights are there,
/// so is the state that says what they belong to.
pub fn write_safetensors_with_metadata(
    path: impl AsRef<Path>,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
    metadata: &HashMap<String, String>,
) -> anyhow::Result<()> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        let mut fields = serde_json::Map::new();
        for (key, value) in metadata {
            fields.insert(key.clone(), serde_json::Value::String(value.clone()));
        }
        header.insert("__metadata__".into(), serde_json::Value::Object(fields));
    }
    let mut buffer: Vec<u8> = Vec::new();
    for (name, shape, values) in tensors {
        anyhow::ensure!(
            shape.iter().product::<usize>() == values.len(),
            "'{name}': shape {shape:?} does not match {} values",
            values.len()
        );
        let begin = buffer.len();
        for v in values {
            buffer.extend_from_slice(&v.to_le_bytes());
        }
        let mut entry = serde_json::Map::new();
        entry.insert("dtype".into(), "F32".into());
        entry.insert(
            "shape".into(),
            serde_json::Value::Array(shape.iter().map(|d| (*d as u64).into()).collect()),
        );
        entry.insert(
            "data_offsets".into(),
            serde_json::Value::Array(vec![(begin as u64).into(), (buffer.len() as u64).into()]),
        );
        header.insert(name.clone(), serde_json::Value::Object(entry));
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header))?;
    let mut out = Vec::with_capacity(8 + header.len() + buffer.len());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&buffer);
    std::fs::write(path.as_ref(), out)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", path.as_ref().display()))?;
    Ok(())
}

/// The integer code `_mask_codes` in `model.py` assigns to a key-mask.
///
/// Python builds the row as `[(mask >> i) & 1 for i in range(n_keys)]` and then
/// encodes it with `1 << arange(n_keys - 1, -1, -1)`, so the code is the mask
/// with its bits reversed. Only the *matching* between a category and a target
/// action depends on this, but the two sides must use the same function, so it
/// lives in exactly one place: here.
pub fn mask_code(mask: &[f32]) -> i64 {
    let n_keys = mask.len();
    let mut code = 0i64;
    for (j, bit) in mask.iter().enumerate() {
        if *bit > 0.5 {
            code |= 1 << (n_keys - 1 - j);
        }
    }
    code
}

/// Enumerate the held-key categories: every key-mask with at most `max_held`
/// bits set, in the order `range(1 << n_keys)` produces, with each row's code.
///
/// Written once here so the Python and the Rust cannot drift: this is the
/// distribution PPO scores against, and a silent difference between the sampled
/// and the scored distribution is exactly the bug the Python's comments warn
/// about.
pub fn held_categories(n_keys: usize, max_held: usize) -> (Vec<Vec<f32>>, Vec<i64>) {
    let mut masks = Vec::new();
    let mut codes = Vec::new();
    for mask in 0..(1u64 << n_keys) {
        if mask.count_ones() as usize > max_held {
            continue;
        }
        let row: Vec<f32> = (0..n_keys).map(|i| ((mask >> i) & 1) as f32).collect();
        codes.push(mask_code(&row));
        masks.push(row);
    }
    (masks, codes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_categories_match_the_python_enumeration() {
        // n_keys = 3, max_held = 1: the empty set plus each singleton, in the
        // order `range(1 << n_keys)` produces, with `_mask_codes`' bit-reversed
        // encoding.
        let (masks, codes) = held_categories(3, 1);
        assert_eq!(codes, vec![0, 4, 2, 1]);
        assert_eq!(masks[0], vec![0.0, 0.0, 0.0]);
        assert_eq!(masks[1], vec![1.0, 0.0, 0.0]);
        assert_eq!(masks[2], vec![0.0, 1.0, 0.0]);
        assert_eq!(masks[3], vec![0.0, 0.0, 1.0]);
        // Every row's code is what `mask_code` says it is, by definition.
        for (row, code) in masks.iter().zip(&codes) {
            assert_eq!(mask_code(row), *code);
        }
    }

    #[test]
    fn held_categories_count_what_the_cap_allows() {
        // The default action space: 9 keys, at most 4 at once.
        let (masks, _) = held_categories(9, 4);
        let expected: usize = (0..=4)
            .map(|k| {
                let mut c = 1usize;
                for i in 0..k {
                    c = c * (9 - i) / (i + 1);
                }
                c
            })
            .sum();
        assert_eq!(masks.len(), expected);
    }

    #[test]
    fn a_round_trip_preserves_values_and_shapes() {
        let dir = std::env::temp_dir().join("bot1-weights-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("round-trip.safetensors");
        let tensors = vec![
            ("a".to_string(), vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("b".to_string(), vec![4], vec![-1.0, 0.5, 0.25, 0.0]),
        ];
        write_safetensors(&path, &tensors).unwrap();
        let read = Weights::open(&path).unwrap();
        assert_eq!(read.shape("a").unwrap(), &[2, 3]);
        assert_eq!(read.f32("a").unwrap(), tensors[0].2);
        assert_eq!(read.f32("b").unwrap(), tensors[1].2);
        assert!(read.f32("missing").is_err());
    }
}
