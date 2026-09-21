//! The model, ported from `botcore/model.py`.
//!
//! There is exactly one network, and everything the bot does comes out of it:
//!
//! ```text
//! frame stack (C, S, S) -> patch embedding -> tokens (N, E)
//!                                               |
//!                         memory window (M, E) -+-> causal transformer
//!                                                    (RoPE attention + SwiGLU)
//!                                                           |
//!                                     +---------------------+------------------+
//!                                     |            |             |            |
//!                                  held keys    turn dir     turn speed      tap
//!                                     |            |             |            |
//!                                 value head   next-frame prediction (curiosity)
//! ```
//!
//! The shape of the port, and why it looks like this:
//!
//! * **RoPE angles are computed on the host and uploaded.** They depend only on
//!   `head_dim` and the token count, and building them out of Burn scalar ops
//!   would put a second, gratuitously different implementation of `powf` between
//!   this and the Python. Host-side `f32` arithmetic matches `torch`'s `f32`
//!   arithmetic to the last bit almost everywhere, and the parity test measures
//!   the rest.
//! * **Attention is written out** rather than taken from `burn-nn`. The Python
//!   applies RoPE to `q` and `k` *before* splitting heads apart for the matmul,
//!   uses one fused `qkv` projection, and masks with a boolean lower triangle;
//!   reproducing that exactly is worth more than the lines it saves.
//! * **The memory window is stored, not recomputed.** Same as the Python: the
//!   window that was in force when an action was taken is what PPO scores that
//!   action against.
//!
//! Layers are laid out as a plain parameter container (`Net`) plus a wrapper
//! (`ActorCritic`) that holds the non-trainable state — the rope tables, the
//! held-key category set, the bot's own mouse step. Burn's `Module` derive only
//! sees things that are parameters, which keeps the optimiser's view of the
//! model exactly the set of tensors the Python's `policy.parameters()` yields.

use burn::module::{Module, Param};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::ops::PadMode;
use burn::tensor::{Int, TensorData, activation};

use crate::config::Config;
use crate::weights::{held_categories, mask_code, Weights};

/// `torch.nn.LayerNorm`'s default epsilon. Burn's default differs, and a
/// silent difference here would show up as a small, permanent offset in every
/// activation.
const LAYER_NORM_EPS: f64 = 1e-5;

/// `-inf` for a masked attention position, in a form that survives `exp`.
///
/// `torch`'s boolean mask makes the fused kernel use `-inf`; `exp(-1e30 - max)`
/// underflows to exactly zero in `f32`, so this is the same thing without ever
/// producing a `NaN` from `-inf * 0`.
const MASKED_ATTENTION: f32 = -1e30;

/// A decision: the held-key mask plus the three categorical choices.
///
/// The held mask is a `n_keys`-long 0/1 vector rather than an index, because
/// the policy is a set of independent Bernoulli bits under a top-k cap; see
/// [`ActorCritic::held_log_prob`].
#[derive(Debug, Clone, PartialEq)]
pub struct Action {
    pub held: Vec<f32>,
    pub direction: usize,
    pub speed: usize,
    pub tap: usize,
}

impl Action {
    /// The index of the lowest key being held, plus one, or 0 for none.
    ///
    /// This is how the action description the novelty code reads summarises a
    /// key-set: the difference between "hold W" and "hold W and shift" is one
    /// bit, and the rest of the mask would only make its input wider.
    pub fn first_held(&self) -> usize {
        self.held
            .iter()
            .position(|bit| *bit > 0.5)
            .map(|i| i + 1)
            .unwrap_or(0)
    }
}

/// Shapes the model needs at run time that are not parameters.
#[derive(Debug, Clone)]
pub struct Dims {
    pub embed_dim: usize,
    pub mem_tokens: usize,
    pub patch: usize,
    pub grid: usize,
    pub frame_tokens: usize,
    pub in_channels: usize,
    pub patch_width: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    pub transition_hidden: usize,
    pub n_keys: usize,
    pub n_turn: usize,
    pub n_speed: usize,
    pub n_tap: usize,
    pub max_held: usize,
}

/// The rotary tables, and the parameters they are built from.
#[derive(Debug, Clone)]
pub struct Rope {
    pub head_dim: usize,
    pub base: f32,
    /// `(tokens, half)` cosine and sine tables per token count, in the order
    /// the file's largest forward pass needs.
    pub cache: std::cell::RefCell<Vec<(usize, Vec<f32>, Vec<f32>)>>,
}

impl Rope {
    pub fn new(head_dim: usize, base: f32) -> Self {
        let head_dim = if head_dim % 2 == 1 { head_dim - 1 } else { head_dim };
        Self {
            head_dim: head_dim.max(2),
            base,
            cache: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// `(tokens, half)` cosine and sine tables.
    ///
    /// This is `RotaryEmbedding._angles` with the same `inv_freq`:
    /// `1 / base ** (i / half)`, where `half = head_dim / 2`. The Python
    /// divides by `half`, not by `head_dim`, which is unusual enough to be
    /// worth stating: the tables here have to be the same ones or nothing
    /// downstream lines up.
    fn tables(&self, tokens: usize) -> (Vec<f32>, Vec<f32>) {
        if let Some((_, cos, sin)) = self
            .cache
            .borrow()
            .iter()
            .find(|(t, _, _)| *t == tokens)
        {
            return (cos.clone(), sin.clone());
        }
        let half = self.head_dim / 2;
        let mut cos = vec![0.0f32; tokens * half];
        let mut sin = vec![0.0f32; tokens * half];
        for position in 0..tokens {
            for i in 0..half {
                let inv_freq = 1.0 / self.base.powf(i as f32 / half.max(1) as f32);
                let angle = position as f32 * inv_freq;
                cos[position * half + i] = angle.cos();
                sin[position * half + i] = angle.sin();
            }
        }
        let mut cache = self.cache.borrow_mut();
        // A handful of token counts in practice, so the cache is bounded rather
        // than growing with the run - the same reasoning as the Python's.
        if cache.len() > 8 {
            cache.clear();
        }
        cache.push((tokens, cos.clone(), sin.clone()));
        (cos, sin)
    }

    /// `x` is `(rows, tokens, heads, head_dim)`; the result is the same shape,
    /// rotated by an angle that depends only on the token position.
    pub fn apply<B: Backend>(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [rows, tokens, heads, head_dim] = x.dims();
        let half = head_dim / 2;
        let (cos, sin) = self.tables(tokens);
        let device = x.device();
        let cos = Tensor::<B, 5>::from_data(
            TensorData::new(cos, [1, tokens, 1, half, 1]),
            &device,
        );
        let sin = Tensor::<B, 5>::from_data(
            TensorData::new(sin, [1, tokens, 1, half, 1]),
            &device,
        );
        let pairs = x.reshape([rows, tokens, heads, half, 2]);
        let take = |index: usize| {
            pairs
                .clone()
                .slice([0..rows, 0..tokens, 0..heads, 0..half, index..index + 1])
                .reshape([rows, tokens, heads, half, 1])
        };
        // `_rotate`: a quarter turn, `(first, second) -> (-second, first)`.
        let rotated = Tensor::cat(vec![take(1).neg(), take(0)], 4);
        (pairs * cos + rotated * sin).reshape([rows, tokens, heads, head_dim])
    }
}

/// Every key-mask with at most `max_held` bits set.
///
/// This is the support of the held-key distribution. It is a constant, so it is
/// built once and kept - including the `±1` sign matrix the log-probability
/// needs, which turns what would be a `where` into a multiply. See
/// [`ActorCritic::held_logits_of_masks`].
#[derive(Debug, Clone)]
pub struct HeldCats {
    pub masks: Vec<Vec<f32>>,
    pub codes: Vec<i64>,
    /// `(n_cat, n_keys)`, `+1` where the bit is unset and `-1` where it is set.
    pub sign: Vec<f32>,
    pub n_keys: usize,
    pub n_cat: usize,
}

impl HeldCats {
    pub fn new(n_keys: usize, max_held: usize) -> Self {
        let (masks, codes) = held_categories(n_keys, max_held);
        let sign = masks
            .iter()
            .flat_map(|row| row.iter().map(|bit| 1.0 - 2.0 * bit))
            .collect();
        let n_cat = masks.len();
        Self {
            masks,
            codes,
            sign,
            n_keys,
            n_cat,
        }
    }

    /// The category index for a held-key mask, or `None` if it is outside the
    /// set the cap allows.
    pub fn index_of(&self, held: &[f32]) -> Option<usize> {
        let code = mask_code(held);
        self.codes.iter().position(|c| *c == code)
    }
}

// =============================================================================
// Building blocks
// =============================================================================

/// `SwiGLU(x) = W2( SiLU(W1 x) * W3 x )`, the gated feed-forward every block
/// uses. Two projections where a plain MLP has one, which is why the hidden
/// width stays a small multiple of the embedding.
#[derive(Module, Debug)]
pub struct SwiGlu<B: Backend> {
    pub gate: Linear<B>,
    pub up: Linear<B>,
    pub down: Linear<B>,
}

impl<B: Backend> SwiGlu<B> {
    pub fn new(dim: usize, hidden: usize, device: &B::Device) -> Self {
        Self {
            gate: LinearConfig::new(dim, hidden).with_bias(false).init(device),
            up: LinearConfig::new(dim, hidden).with_bias(false).init(device),
            down: LinearConfig::new(hidden, dim).with_bias(false).init(device),
        }
    }

    /// `W2( SiLU(W1 x) * W3 x )`, on a `(rows, tokens, dim)` tensor.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gated = activation::silu(self.gate.forward(x.clone())) * self.up.forward(x);
        self.down.forward(gated)
    }
}

/// Pre-norm attention + SwiGLU, both residual.
#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub attn_norm: LayerNorm<B>,
    pub qkv: Linear<B>,
    pub attn_out: Linear<B>,
    pub ffn_norm: LayerNorm<B>,
    pub ffn: SwiGlu<B>,
}

/// `(B, C, S, S) -> (B, tokens, embed_dim)`.
///
/// One strided convolution and nothing else: the transformer does the sequence
/// work. The convolution is run as an explicit matmul over unrolled patches
/// rather than through a convolution op, because with `stride == kernel` it *is*
/// a matmul, and doing it this way makes the token order and the weight layout
/// visible instead of conventional.
#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    /// `(channels * patch * patch, width)`: the convolution's weight with its
    /// patch dimensions flattened and the output width moved last, so the patch
    /// matmul reads straight off it. PyTorch stores the transpose of this.
    pub proj_w: Param<Tensor<B, 2>>,
    pub proj_b: Param<Tensor<B, 1>>,
    pub norm: LayerNorm<B>,
    pub out: Linear<B>,
}

// =============================================================================
// The whole bot: one transformer, every head
// =============================================================================

/// The curiosity head, as a module of its own.
///
/// It is `nn.Sequential(LayerNorm, Linear, SiLU, Linear)` in the Python, and it
/// is kept separate here for one reason that matters: it has its own optimiser,
/// and an optimiser in Burn carries state keyed by parameter *id*. Handing the
/// transition optimiser the whole network would create state for every policy
/// parameter and leave it at zero; handing it this module updates exactly the
/// tensors the Python's `transition_optimizer` owns.
#[derive(Module, Debug)]
pub struct Transition<B: Backend> {
    pub norm: LayerNorm<B>,
    pub input: Linear<B>,
    pub output: Linear<B>,
}

impl<B: Backend> Transition<B> {
    pub fn new(embed_dim: usize, hidden: usize, device: &B::Device) -> Self {
        Self {
            norm: LayerNormConfig::new(embed_dim)
                .with_epsilon(LAYER_NORM_EPS)
                .init(device),
            input: LinearConfig::new(embed_dim, hidden).init(device),
            output: LinearConfig::new(hidden, embed_dim).init(device),
        }
    }

    /// What the next frame's representation should be.
    ///
    /// The input is detached, so this cannot bend the policy's features to make
    /// the reward look better - the same reasoning as the Python's.
    pub fn forward(&self, context: Tensor<B, 2>) -> Tensor<B, 2> {
        let normed = self.norm.forward(context.detach());
        let hidden = activation::silu(self.input.forward(normed));
        self.output.forward(hidden)
    }
}

/// The parameters, and only the parameters.
///
/// This is the exact set of tensors the Python's `policy.parameters()` yields,
/// which is what the optimiser is allowed to move.
#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    pub patch_embed: PatchEmbed<B>,
    /// The memory slot for a step that has not happened yet. Learned, so the
    /// model can decide for itself what "the start of a run" looks like.
    pub bos: Param<Tensor<B, 3>>,
    pub blocks: Vec<Block<B>>,
    pub final_norm: LayerNorm<B>,
    pub hold_head: Linear<B>,
    pub turn_head: Linear<B>,
    pub speed_head: Linear<B>,
    pub tap_head: Linear<B>,
    pub value_head: Linear<B>,
    pub transition: Transition<B>,
}

/// What one control step produces.
pub struct StepOutput<B: Backend> {
    pub action: Action,
    /// Log-probability of the action under the capped held distribution and the
    /// three categorical heads, summed - the number PPO compares.
    pub log_prob: Tensor<B, 1>,
    pub value: Tensor<B, 1>,
    /// The window for the next step: this frame's tokens pushed in.
    pub new_hidden: Tensor<B, 3>,
    /// The pooled representation of the window, reused by the curiosity head.
    pub context: Tensor<B, 2>,
    /// This frame's token summary, which is what the next step's prediction
    /// error is measured against.
    pub summary: Tensor<B, 2>,
}

/// What scoring a recorded sequence produces.
pub struct SequenceOutput<B: Backend> {
    pub log_prob: Tensor<B, 1>,
    pub entropy: Tensor<B, 1>,
    pub value: Tensor<B, 1>,
    pub context: Tensor<B, 2>,
}

/// The whole bot.
pub struct ActorCritic<B: Backend> {
    pub net: Net<B>,
    pub dims: Dims,
    pub rope: Rope,
    pub held: HeldCats,
    /// The bot's own mouse speed: pixels per 1x turn step.
    ///
    /// A buffer rather than a parameter in the Python, and a plain field here,
    /// because it lives outside the differentiable graph: the policy says "turn
    /// left at speed 2x" and this says what 1x is worth. No policy gradient can
    /// reach it, which is why the session measures and corrects it instead.
    pub mouse_step: f32,
    pub mouse_step_min: f32,
    pub mouse_step_max: f32,
}

impl<B: Backend> ActorCritic<B> {
    /// Build a model around an already-built parameter tree, taking every
    /// non-parameter field from a template.
    ///
    /// This is how the collection path gets its own copy of the weights on the
    /// non-autodiff backend: `Net::valid()` converts the parameters once after
    /// each update, and the per-step forward pass then runs without recording a
    /// graph it will never differentiate.
    pub fn from_net<B2: Backend>(net: Net<B>, template: &ActorCritic<B2>) -> Self {
        Self {
            net,
            dims: template.dims.clone(),
            rope: template.rope.clone(),
            held: template.held.clone(),
            mouse_step: template.mouse_step,
            mouse_step_min: template.mouse_step_min,
            mouse_step_max: template.mouse_step_max,
        }
    }

    /// Build the model, with the shapes but *not* the weights: call
    /// [`ActorCritic::load`] to put real numbers in it, or [`ActorCritic::init`]
    /// for the Python's own initialisation.
    pub fn new(
        cfg: &Config,
        head_sizes: [usize; 4],
        observation_channels: usize,
        device: &B::Device,
    ) -> Self {
        let embed_dim = cfg.embed_dim;
        let mem_tokens = cfg.mem_tokens.max(1);
        let patch = cfg.patch_size.max(2);
        let grid = cfg.frame_size.div_ceil(patch).max(1);
        let frame_tokens = grid * grid;
        let heads = cfg.attention_heads.max(1);
        let head_dim = (embed_dim / heads).max(2);
        let inner = head_dim * heads;
        let [n_hold, n_turn, n_speed, n_tap] = head_sizes;
        let n_keys = n_hold.saturating_sub(1).max(1);
        let max_held = cfg.max_held_keys.min(n_keys).max(1);

        let patch_embed = PatchEmbed {
            proj_w: Param::from_tensor(Tensor::zeros(
                [observation_channels * patch * patch, cfg.patch_width],
                device,
            )),
            proj_b: Param::from_tensor(Tensor::zeros([cfg.patch_width], device)),
            norm: LayerNormConfig::new(cfg.patch_width)
                .with_epsilon(LAYER_NORM_EPS)
                .init(device),
            out: LinearConfig::new(cfg.patch_width, embed_dim)
                .with_bias(false)
                .init(device),
        };
        let blocks = (0..cfg.transformer_layers.max(1))
            .map(|_| Block {
                attn_norm: LayerNormConfig::new(embed_dim)
                    .with_epsilon(LAYER_NORM_EPS)
                    .init(device),
                qkv: LinearConfig::new(embed_dim, 3 * inner)
                    .with_bias(false)
                    .init(device),
                attn_out: LinearConfig::new(inner, embed_dim)
                    .with_bias(false)
                    .init(device),
                ffn_norm: LayerNormConfig::new(embed_dim)
                    .with_epsilon(LAYER_NORM_EPS)
                    .init(device),
                ffn: SwiGlu::new(embed_dim, cfg.ffn_hidden, device),
            })
            .collect();
        let net = Net {
            patch_embed,
            bos: Param::from_tensor(Tensor::zeros([1, 1, embed_dim], device)),
            blocks,
            final_norm: LayerNormConfig::new(embed_dim)
                .with_epsilon(LAYER_NORM_EPS)
                .init(device),
            hold_head: LinearConfig::new(embed_dim, n_keys).init(device),
            turn_head: LinearConfig::new(embed_dim, n_turn.max(1)).init(device),
            speed_head: LinearConfig::new(embed_dim, n_speed.max(1)).init(device),
            tap_head: LinearConfig::new(embed_dim, n_tap.max(1)).init(device),
            value_head: LinearConfig::new(embed_dim, 1).init(device),
            transition: Transition::new(embed_dim, cfg.transition_hidden, device),
        };

        Self {
            net,
            dims: Dims {
                embed_dim,
                mem_tokens,
                patch,
                grid,
                frame_tokens,
                in_channels: observation_channels,
                patch_width: cfg.patch_width,
                layers: cfg.transformer_layers.max(1),
                heads,
                head_dim,
                ffn_hidden: cfg.ffn_hidden,
                transition_hidden: cfg.transition_hidden,
                n_keys,
                n_turn: n_turn.max(1),
                n_speed: n_speed.max(1),
                n_tap: n_tap.max(1),
                max_held,
            },
            rope: Rope::new(head_dim, 10_000.0),
            held: HeldCats::new(n_keys, max_held),
            mouse_step: cfg.mouse_turn_start.max(1) as f32,
            mouse_step_min: cfg.mouse_turn_min.max(1) as f32,
            mouse_step_max: cfg.mouse_turn_max.max(cfg.mouse_turn_min).max(1) as f32,
        }
    }

    // ---- the bot's own mouse speed ----

    /// Pixels per 1x turn step, as an integer, for the input layer.
    pub fn mouse_step_value(&self) -> i32 {
        (self.mouse_step.round() as i32).max(1)
    }

    pub fn set_mouse_step(&mut self, pixels: Option<f32>) {
        if let Some(value) = pixels
            && value > 0.0
        {
            self.mouse_step = value.clamp(self.mouse_step_min, self.mouse_step_max);
        }
    }

    /// Move the base turn step, staying inside the configured bounds.
    ///
    /// Used by the session's speed correction: when turns repeatedly fail to
    /// move the picture, this is the knob that closes the gap. Bounded on both
    /// ends so a bad estimate cannot turn a small correction into a spin.
    pub fn nudge_mouse_step(&mut self, factor: f32) {
        self.mouse_step =
            (self.mouse_step * factor).clamp(self.mouse_step_min, self.mouse_step_max);
    }

    // ---- the held-key decision ----
    //
    // The held-key head is a set of independent Bernoulli bits, conditioned on
    // at most `max_held` of them being set. `sample` and `sequence_log_prob`
    // share every distribution below, because PPO compares the two: a sample
    // modified after its log-probability was recorded makes the ratio - and so
    // the gradient - meaningless.

    /// Unnormalised log-probability of every category, `(rows, n_cat)`.
    ///
    /// The per-key term is the Bernoulli log-probability of that bit, written
    /// the numerically stable way round: `-softplus(-z)` for a set bit and
    /// `-softplus(z)` for an unset one.
    pub fn held_logits_of_masks(&self, logits: Tensor<B, 2>) -> Tensor<B, 2> {
        let [rows, n_keys] = logits.dims();
        let n_cat = self.held.n_cat;
        let sign = Tensor::<B, 2>::from_data(
            TensorData::new(self.held.sign.clone(), [n_cat, n_keys]),
            &logits.device(),
        );
        let wide = logits.unsqueeze_dim::<3>(1).expand([rows, n_cat, n_keys]);
        // `sign` is `+1` where the bit is unset and `-1` where it is set, so
        // `wide * sign` is the `where(masks > 0.5, -wide, wide)` the Python
        // writes - and the negation outside the softplus is what turns it into
        // `log sigmoid(z)` for a set bit and `log(1 - sigmoid(z))` for an unset
        // one. Getting the sign the wrong way round is a silent, plausible
        // wrong answer, which is why the parity test compares this number.
        let signed = wide * sign.unsqueeze::<3>();
        softplus_stable(signed)
            .neg()
            .sum_dim(2)
            .reshape([rows, n_cat])
    }

    /// `logsumexp` over the category dimension, kept as a `(rows, 1)` tensor so
    /// it broadcasts against the scores.
    fn category_log_norm(&self, scores: Tensor<B, 2>) -> Tensor<B, 2> {
        let max = scores.clone().max_dim(1);
        let shifted = scores - max.clone();
        shifted.exp().sum_dim(1).log() + max
    }

    /// Log-probability of a key-set under the capped distribution, `(rows,)`.
    ///
    /// Shared by the sampler and the update, which is the whole point: the
    /// importance ratio PPO computes is only meaningful when the behaviour
    /// distribution and the scored distribution are the same one.
    pub fn held_log_prob(&self, logits: Tensor<B, 2>, held: Tensor<B, 2>) -> Tensor<B, 1> {
        let [rows, n_keys] = logits.dims();
        let n_cat = self.held.n_cat;
        let scores = self.held_logits_of_masks(logits);
        let log_norm = self.category_log_norm(scores.clone()).reshape([rows]);

        // The target's one-hot position, built on the host: the actions are
        // constants in the graph and the category table is a Rust-side constant,
        // so this is a table lookup, not a tensor operation.
        let held_host = held.into_data().to_vec::<f32>().expect("held is f32");
        let mut position = vec![0.0f32; rows * n_cat];
        let mut in_set = vec![0.0f32; rows];
        for row in 0..rows {
            let bits = &held_host[row * n_keys..(row + 1) * n_keys];
            if let Some(index) = self.held.index_of(bits) {
                position[row * n_cat + index] = 1.0;
                in_set[row] = 1.0;
            }
        }
        let device = scores.device();
        let position =
            Tensor::<B, 2>::from_data(TensorData::new(position, [rows, n_cat]), &device);
        let in_set = Tensor::<B, 1>::from_data(TensorData::new(in_set, [rows]), &device);
        let selected = (scores * position).sum_dim(1).reshape([rows]);
        // A mask outside the category set cannot come out of `sample`, but if one
        // ever did, its probability is zero rather than a silently wrong number.
        let inside = selected - log_norm;
        let outside = Tensor::<B, 1>::full([rows], -1e9f32, &device);
        inside * in_set.clone() + outside * (in_set.neg() + 1.0)
    }

    /// Entropy of the capped distribution, in nats, `(rows,)`.
    pub fn held_entropy(&self, logits: Tensor<B, 2>) -> Tensor<B, 1> {
        let [rows, _n_keys] = logits.dims();
        let scores = self.held_logits_of_masks(logits);
        let log_probs = scores.clone() - self.category_log_norm(scores);
        let probabilities = log_probs.clone().exp();
        (probabilities * log_probs)
            .sum_dim(1)
            .reshape([rows])
            .neg()
    }

    /// Draw a key-set from the capped distribution, and score it.
    ///
    /// Sampling happens on the host, because the distribution is a categorical
    /// over up to a few hundred categories and moving it to the host is both
    /// simpler and cheaper than a device-side sampler. The recorded
    /// log-probability then comes from [`Self::held_log_prob`] rather than being
    /// recomputed inline, so the sampler and the update cannot drift apart.
    pub fn sample_held(
        &self,
        logits: Tensor<B, 2>,
        rng: &mut impl rand::Rng,
    ) -> (Vec<f32>, Tensor<B, 1>) {
        let [rows, n_keys] = logits.dims();
        debug_assert_eq!(rows, 1, "sampling is per control step");
        let scores = self
            .held_logits_of_masks(logits.clone())
            .into_data()
            .to_vec::<f32>()
            .expect("scores are f32");
        let index = sample_categorical(&scores, rng);
        let bits = self.held.masks[index].clone();
        let held = Tensor::<B, 2>::from_data(
            TensorData::new(bits.clone(), [rows, n_keys]),
            &logits.device(),
        );
        let log_prob = self.held_log_prob(logits, held);
        (bits, log_prob)
    }

    // ---- forward paths ----

    /// Frame stack -> patch tokens, `(rows, C, S, S) -> (rows, tokens, E)`.
    pub fn encode(&self, frames: Tensor<B, 4>) -> Tensor<B, 3> {
        let [rows, channels, height, width] = frames.dims();
        let patch = self.dims.patch;
        // Pad once at the edge rather than re-deriving the grid: the token count
        // downstream must stay fixed for a fixed frame_size. PyTorch pads the
        // right and bottom edges with zeros, which is what this reproduces.
        let frames = if height % patch != 0 || width % patch != 0 {
            let pad_h = (patch - height % patch) % patch;
            let pad_w = (patch - width % patch) % patch;
            frames.pad(
                [(0, 0), (0, 0), (0, pad_h), (0, pad_w)],
                PadMode::Constant(0.0),
            )
        } else {
            frames
        };
        let [_rows, _channels, height, width] = frames.dims();
        let grid_h = height / patch;
        let grid_w = width / patch;
        let unrolled = channels * patch * patch;
        let tokens = grid_h * grid_w;
        // (rows, channels, gh, p, gw, p) -> (rows, gh, gw, channels, p, p):
        // token index is gh * grid_w + gw, which is the order `flatten(2)` gives.
        let patches = frames
            .reshape([rows, channels, grid_h, patch, grid_w, patch])
            .permute([0, 2, 4, 1, 3, 5])
            .reshape([rows * tokens, unrolled]);
        let projected = patches.matmul(self.net.patch_embed.proj_w.val())
            + self.net.patch_embed.proj_b.val().unsqueeze::<2>();
        let projected = projected.reshape([rows, tokens, self.dims.patch_width]);
        self.net.patch_embed.out.forward(activation::silu(
            self.net.patch_embed.norm.forward(projected),
        ))
    }

    /// The memory window at the start of a run: every slot is the BOS token.
    pub fn initial_hidden(&self, batch: usize, device: &B::Device) -> Tensor<B, 3> {
        self.net
            .bos
            .val()
            .detach()
            .expand([batch, self.dims.mem_tokens, self.dims.embed_dim])
            .to_device(device)
    }

    /// Push one step's frame tokens into the window.
    ///
    /// The window is appended to and rolled; the new step's representation is
    /// the mean of its own tokens, which is what the next step will attend to.
    /// Rolling rather than growing is what keeps the per-step cost flat.
    pub fn advance(&self, hidden: Tensor<B, 3>, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let [rows, _mem, embed] = hidden.dims();
        let window = self.dims.mem_tokens;
        let summary = tokens.mean_dim(1);
        Tensor::cat(vec![hidden, summary], 1).slice([
            0..rows,
            1..window + 1,
            0..embed,
        ])
    }

    /// The transformer over the memory window plus this step's tokens.
    ///
    /// `hidden` is `(rows, M, E)` and `tokens` is `(rows, N, E)`. Returns the
    /// window for the next step, the pooled context (value estimate and
    /// curiosity head), and the last position's output (the decision heads).
    ///
    /// The leading batch dimensions the Python's version folds together are
    /// written out as `rows` here: flattening `(T, B)` into one attention batch
    /// is the caller's job, and burn's shapes make that explicit rather than
    /// implicit.
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        tokens: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>) {
        let window = self.advance(hidden, tokens.clone());
        let sequence = Tensor::cat(vec![window.clone(), tokens], 1);
        let [rows, total, embed] = sequence.dims();
        let device = sequence.device();
        let mask = causal_mask::<B>(total, &device);
        let mut x = sequence;
        for block in &self.net.blocks {
            x = self.attention_block(block, x, mask.clone());
        }
        let x = self.net.final_norm.forward(x);
        let mem = self.dims.mem_tokens;
        let context = x
            .clone()
            .slice([0..rows, 0..mem, 0..embed])
            .mean_dim(1)
            .reshape([rows, embed]);
        let features = x
            .slice([0..rows, total - 1..total, 0..embed])
            .reshape([rows, embed]);
        (window, context, features)
    }

    /// One pre-norm block: RoPE causal self-attention, then SwiGLU, both added
    /// back onto their input.
    fn attention_block(
        &self,
        block: &Block<B>,
        x: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let [rows, tokens, _embed] = x.dims();
        let heads = self.dims.heads;
        let head_dim = self.dims.head_dim;
        let inner = heads * head_dim;
        let qkv = block
            .qkv
            .forward(block.attn_norm.forward(x.clone()))
            .reshape([rows, tokens, 3, heads, head_dim]);
        let part = |index: usize| {
            qkv.clone()
                .slice([
                    0..rows,
                    0..tokens,
                    index..index + 1,
                    0..heads,
                    0..head_dim,
                ])
                .reshape([rows, tokens, heads, head_dim])
        };
        let query = self.rope.apply(part(0));
        let key = self.rope.apply(part(1));
        let value = part(2);
        // (rows, heads, tokens, head_dim) so the attention weights read
        // naturally as a tokens x tokens square.
        let query = query.swap_dims(1, 2);
        let key = key.swap_dims(1, 2).swap_dims(2, 3);
        let value = value.swap_dims(1, 2);
        let batched = rows * heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scores = query
            .reshape([batched, tokens, head_dim])
            .matmul(key.reshape([batched, head_dim, tokens]))
            * scale
            + mask.unsqueeze::<3>();
        let probabilities = activation::softmax(scores, 2);
        let attended = probabilities
            .matmul(value.reshape([batched, tokens, head_dim]))
            .reshape([rows, heads, tokens, head_dim])
            .swap_dims(1, 2)
            .reshape([rows, tokens, inner]);
        let x = x + block.attn_out.forward(attended);
        x.clone() + block.ffn.forward(block.ffn_norm.forward(x))
    }

    /// Every decision head, plus the value, from one forward pass.
    pub fn heads(
        &self,
        features: Tensor<B, 2>,
        context: Tensor<B, 2>,
    ) -> (
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 1>,
    ) {
        let [rows, _] = features.dims();
        (
            self.net.hold_head.forward(features.clone()),
            self.net.turn_head.forward(features.clone()),
            self.net.speed_head.forward(features.clone()),
            self.net.tap_head.forward(features),
            self.net.value_head.forward(context).reshape([rows]),
        )
    }

    /// The curiosity head: what the next frame's representation should be.
    ///
    /// The input is detached inside [`Transition::forward`], exactly as
    /// `predict_next` does in the Python.
    pub fn predict_next(&self, context: Tensor<B, 2>) -> Tensor<B, 2> {
        self.net.transition.forward(context)
    }

    /// One control step: `(1, C, S, S)` and the memory window in.
    pub fn step(
        &self,
        obs: Tensor<B, 4>,
        hidden: Tensor<B, 3>,
        rng: &mut impl rand::Rng,
    ) -> StepOutput<B> {
        let tokens = self.encode(obs);
        let (new_hidden, context, features) = self.forward(hidden, tokens.clone());
        let (hold_logits, turn_logits, speed_logits, tap_logits, value) =
            self.heads(features, context.clone());
        let (held, hold_log_prob) = self.sample_held(hold_logits, rng);
        let direction = sample_categorical(&logits_to_host(&turn_logits), rng);
        let speed = sample_categorical(&logits_to_host(&speed_logits), rng);
        let tap = sample_categorical(&logits_to_host(&tap_logits), rng);
        let rest = categorical_log_prob(&turn_logits, direction)
            + categorical_log_prob(&speed_logits, speed)
            + categorical_log_prob(&tap_logits, tap);
        let summary = tokens.mean_dim(1).reshape([1, self.dims.embed_dim]);
        StepOutput {
            action: Action {
                held,
                direction,
                speed,
                tap,
            },
            log_prob: hold_log_prob + rest,
            value,
            new_hidden,
            context,
            summary,
        }
    }

    /// Value of an observation given the window - the GAE bootstrap.
    pub fn value_only(
        &self,
        obs: Tensor<B, 4>,
        hidden: Tensor<B, 3>,
    ) -> (Tensor<B, 1>, Tensor<B, 3>) {
        let tokens = self.encode(obs);
        let (new_hidden, context, features) = self.forward(hidden, tokens);
        let (_h, _t, _s, _a, value) = self.heads(features, context);
        (value, new_hidden)
    }

    /// Score a recorded sequence under the current policy.
    ///
    /// `hidden` is `(rows, M, E)` - the memory window *before* each step, exactly
    /// as it was during collection - and `summary` is `(rows, E)`, the
    /// representation of the frame that step saw. The frame is rebuilt from its
    /// stored summary rather than re-encoded, so the transformer reads the same
    /// input in both paths and the same computation is scored as was sampled.
    /// The caller flattens `(T, B)` into `rows`.
    pub fn sequence_log_prob(
        &self,
        hidden: Tensor<B, 3>,
        summary: Tensor<B, 2>,
        held: Tensor<B, 2>,
        direction: Tensor<B, 1, Int>,
        speed: Tensor<B, 1, Int>,
        tap: Tensor<B, 1, Int>,
    ) -> SequenceOutput<B> {
        let [rows, embed] = summary.dims();
        let tokens = summary.reshape([rows, 1, embed]);
        let (_window, context, features) = self.forward(hidden, tokens);
        let (hold_logits, turn_logits, speed_logits, tap_logits, value) =
            self.heads(features, context.clone());
        let log_prob = self.held_log_prob(hold_logits.clone(), held)
            + categorical_log_prob_indexed(&turn_logits, direction)
            + categorical_log_prob_indexed(&speed_logits, speed)
            + categorical_log_prob_indexed(&tap_logits, tap);
        let entropy = self.held_entropy(hold_logits)
            + categorical_entropy(&turn_logits)
            + categorical_entropy(&speed_logits)
            + categorical_entropy(&tap_logits);
        SequenceOutput {
            log_prob,
            entropy,
            value,
            context,
        }
    }

    /// The curiosity head's prediction for a batch of contexts.
    pub fn predict_next_batch(&self, context: Tensor<B, 2>) -> Tensor<B, 2> {
        self.predict_next(context)
    }
    // ---- loading ----

    // ---- saving ----

    /// Every weight, in the same names and layout [`Self::load`] reads.
    ///
    /// This is the inverse of `load`, and it exists so a checkpoint is written
    /// in exactly the format the parity suite already proved this model can
    /// read back: Burn's own recorders would be less code, but they would also
    /// be a second, untested path in and out of the weights.
    pub fn to_weights(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let d = self.dims.clone();
        let embed = d.embed_dim;
        let inner = d.heads * d.head_dim;
        let mut out: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

        // A `Linear` weight, transposed back to PyTorch's (out, in).
        let linear = |linear: &Linear<B>| -> (Vec<usize>, Vec<f32>) {
            let shape = linear.weight.val().dims();
            let (rows, cols) = (shape[0], shape[1]);
            let values = linear
                .weight
                .val()
                .into_data()
                .to_vec::<f32>()
                .expect("weights are f32");
            let mut transposed = vec![0.0f32; values.len()];
            for row in 0..rows {
                for column in 0..cols {
                    transposed[column * rows + row] = values[row * cols + column];
                }
            }
            (vec![cols, rows], transposed)
        };
        let one = |param: &Param<Tensor<B, 1>>| -> (Vec<usize>, Vec<f32>) {
            let values = param.val().into_data().to_vec::<f32>().expect("f32");
            (vec![values.len()], values)
        };
        let norm = |norm: &LayerNorm<B>| -> Vec<(Vec<usize>, Vec<f32>)> {
            let mut fields = vec![one(&norm.gamma)];
            if let Some(beta) = &norm.beta {
                fields.push(one(beta));
            }
            fields
        };
        let push = |out: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
                    name: &str,
                    (shape, values): (Vec<usize>, Vec<f32>)| {
            out.push((name.to_string(), shape, values));
        };

        // The patch embedding's convolution, back to (width, channels, patch, patch).
        {
            let shape = self.net.patch_embed.proj_w.val().dims();
            let (rows, cols) = (shape[0], shape[1]);
            let values = self
                .net
                .patch_embed
                .proj_w
                .val()
                .into_data()
                .to_vec::<f32>()
                .expect("f32");
            let mut transposed = vec![0.0f32; values.len()];
            for row in 0..rows {
                for column in 0..cols {
                    transposed[column * rows + row] = values[row * cols + column];
                }
            }
            push(
                &mut out,
                "patch_embed.proj.weight",
                (
                    vec![d.patch_width, d.in_channels, d.patch, d.patch],
                    transposed,
                ),
            );
            push(&mut out, "patch_embed.proj.bias", one(&self.net.patch_embed.proj_b));
            for (index, field) in norm(&self.net.patch_embed.norm).into_iter().enumerate() {
                push(
                    &mut out,
                    if index == 0 {
                        "patch_embed.norm.weight"
                    } else {
                        "patch_embed.norm.bias"
                    },
                    field,
                );
            }
            push(&mut out, "patch_embed.out.weight", linear(&self.net.patch_embed.out));
        }

        push(&mut out, "bos", {
            let values = self.net.bos.val().into_data().to_vec::<f32>().expect("f32");
            (vec![1, 1, embed], values)
        });

        for (index, block) in self.net.blocks.iter().enumerate() {
            let prefix = format!("blocks.{index}");
            for (field, name) in norm(&block.attn_norm)
                .into_iter()
                .zip(["attn_norm.weight", "attn_norm.bias"])
            {
                push(&mut out, &format!("{prefix}.{name}"), field);
            }
            push(&mut out, &format!("{prefix}.attn.qkv.weight"), linear(&block.qkv));
            push(&mut out, &format!("{prefix}.attn.out.weight"), linear(&block.attn_out));
            for (field, name) in norm(&block.ffn_norm)
                .into_iter()
                .zip(["ffn_norm.weight", "ffn_norm.bias"])
            {
                push(&mut out, &format!("{prefix}.{name}"), field);
            }
            push(&mut out, &format!("{prefix}.ffn.gate.weight"), linear(&block.ffn.gate));
            push(&mut out, &format!("{prefix}.ffn.up.weight"), linear(&block.ffn.up));
            push(&mut out, &format!("{prefix}.ffn.down.weight"), linear(&block.ffn.down));
        }

        for (field, name) in norm(&self.net.final_norm)
            .into_iter()
            .zip(["final_norm.weight", "final_norm.bias"])
        {
            push(&mut out, name, field);
        }

        for (head, name) in [
            (&self.net.hold_head, "hold_head"),
            (&self.net.turn_head, "turn_head"),
            (&self.net.speed_head, "speed_head"),
            (&self.net.tap_head, "tap_head"),
            (&self.net.value_head, "value_head"),
        ] {
            push(&mut out, &format!("{name}.weight"), linear(head));
            if let Some(bias) = &head.bias {
                push(&mut out, &format!("{name}.bias"), one(bias));
            }
        }

        for (field, name) in norm(&self.net.transition.norm)
            .into_iter()
            .zip(["transition_norm.weight", "transition_norm.bias"])
        {
            push(&mut out, name, field);
        }
        push(&mut out, "transition.0.weight", linear(&self.net.transition.input));
        if let Some(bias) = &self.net.transition.input.bias {
            push(&mut out, "transition.0.bias", one(bias));
        }
        push(&mut out, "transition.2.weight", linear(&self.net.transition.output));
        if let Some(bias) = &self.net.transition.output.bias {
            push(&mut out, "transition.2.bias", one(bias));
        }
        let _ = inner;
        out
    }

    /// Load every weight from a file the Python wrote.
    ///
    /// Every tensor must be consumed: a weight that no longer has a home is a
    /// silent behaviour change, and this makes it a loud one instead.
    pub fn load(&mut self, weights: &Weights, device: &B::Device) -> anyhow::Result<()> {
        let d = self.dims.clone();
        let embed = d.embed_dim;
        let inner = d.heads * d.head_dim;
        let mut used: Vec<String> = Vec::new();

        // ---- patch embedding ----
        let flattened = d.in_channels * d.patch * d.patch;
        self.net.patch_embed.proj_w = weights.param_flattened_transposed(
            "patch_embed.proj.weight",
            device,
            &[flattened, d.patch_width],
        )?;
        used.push("patch_embed.proj.weight".into());
        self.net.patch_embed.proj_b =
            weights.param("patch_embed.proj.bias", device, &[d.patch_width])?;
        used.push("patch_embed.proj.bias".into());
        self.net.patch_embed.norm = with_norm(
            weights,
            "patch_embed.norm",
            device,
            d.patch_width,
        )?;
        used.push("patch_embed.norm.weight".into());
        used.push("patch_embed.norm.bias".into());
        self.net.patch_embed.out.weight =
            weights.param_linear("patch_embed.out.weight", device, &[d.patch_width, embed])?;
        used.push("patch_embed.out.weight".into());

        // ---- the memory slot for a step that has not happened yet ----
        self.net.bos = weights.param("bos", device, &[1, 1, embed])?;
        used.push("bos".into());

        // ---- the transformer ----
        for layer in 0..d.layers {
            let prefix = format!("blocks.{layer}");
            let block = &mut self.net.blocks[layer];
            block.attn_norm = with_norm(weights, &format!("{prefix}.attn_norm"), device, embed)?;
            used.push(format!("{prefix}.attn_norm.weight"));
            used.push(format!("{prefix}.attn_norm.bias"));
            block.qkv.weight =
                weights.param_linear(&format!("{prefix}.attn.qkv.weight"), device, &[embed, 3 * inner])?;
            used.push(format!("{prefix}.attn.qkv.weight"));
            block.attn_out.weight = weights.param_linear(
                &format!("{prefix}.attn.out.weight"),
                device,
                &[inner, embed],
            )?;
            used.push(format!("{prefix}.attn.out.weight"));
            block.ffn_norm = with_norm(weights, &format!("{prefix}.ffn_norm"), device, embed)?;
            used.push(format!("{prefix}.ffn_norm.weight"));
            used.push(format!("{prefix}.ffn_norm.bias"));
            block.ffn.gate.weight = weights.param_linear(
                &format!("{prefix}.ffn.gate.weight"),
                device,
                &[embed, d.ffn_hidden],
            )?;
            used.push(format!("{prefix}.ffn.gate.weight"));
            block.ffn.up.weight = weights.param_linear(
                &format!("{prefix}.ffn.up.weight"),
                device,
                &[embed, d.ffn_hidden],
            )?;
            used.push(format!("{prefix}.ffn.up.weight"));
            block.ffn.down.weight = weights.param_linear(
                &format!("{prefix}.ffn.down.weight"),
                device,
                &[d.ffn_hidden, embed],
            )?;
            used.push(format!("{prefix}.ffn.down.weight"));
        }

        self.net.final_norm = with_norm(weights, "final_norm", device, embed)?;
        used.push("final_norm.weight".into());
        used.push("final_norm.bias".into());

        // ---- the decision heads ----
        self.net.hold_head.weight =
            weights.param_linear("hold_head.weight", device, &[embed, d.n_keys])?;
        self.net.hold_head.bias = Some(weights.param("hold_head.bias", device, &[d.n_keys])?);
        used.push("hold_head.weight".into());
        used.push("hold_head.bias".into());
        let head = |weights: &Weights,
                        device: &B::Device,
                        name: &str,
                        size: usize,
                        used: &mut Vec<String>|
         -> anyhow::Result<(Param<Tensor<B, 2>>, Param<Tensor<B, 1>>)> {
            let weight = weights.param_linear(&format!("{name}.weight"), device, &[embed, size])?;
            let bias = weights.param(&format!("{name}.bias"), device, &[size])?;
            used.push(format!("{name}.weight"));
            used.push(format!("{name}.bias"));
            Ok((weight, bias))
        };
        let (w, b) = head(weights, device, "turn_head", d.n_turn, &mut used)?;
        self.net.turn_head.weight = w;
        self.net.turn_head.bias = Some(b);
        let (w, b) = head(weights, device, "speed_head", d.n_speed, &mut used)?;
        self.net.speed_head.weight = w;
        self.net.speed_head.bias = Some(b);
        let (w, b) = head(weights, device, "tap_head", d.n_tap, &mut used)?;
        self.net.tap_head.weight = w;
        self.net.tap_head.bias = Some(b);
        let (w, b) = head(weights, device, "value_head", 1, &mut used)?;
        self.net.value_head.weight = w;
        self.net.value_head.bias = Some(b);

        // ---- the curiosity head ----
        self.net.transition.norm = with_norm(weights, "transition_norm", device, embed)?;
        used.push("transition_norm.weight".into());
        used.push("transition_norm.bias".into());
        let hidden = d.transition_hidden;
        self.net.transition.input.weight =
            weights.param_linear("transition.0.weight", device, &[embed, hidden])?;
        self.net.transition.input.bias =
            Some(weights.param("transition.0.bias", device, &[hidden])?);
        used.push("transition.0.weight".into());
        used.push("transition.0.bias".into());
        self.net.transition.output.weight =
            weights.param_linear("transition.2.weight", device, &[hidden, embed])?;
        self.net.transition.output.bias =
            Some(weights.param("transition.2.bias", device, &[embed])?);
        used.push("transition.2.weight".into());
        used.push("transition.2.bias".into());

        let mut missing: Vec<&str> = weights
            .names()
            .into_iter()
            .filter(|name| !used.iter().any(|u| u == name))
            .collect();
        missing.sort_unstable();
        anyhow::ensure!(
            missing.is_empty(),
            "weights the model has no home for: {missing:?}"
        );
        Ok(())
    }
}

// =============================================================================
// Initialisation
// =============================================================================

/// A standard normal draw, by Box-Muller.
///
/// `torch.nn.init` draws from its own generator and this draws from `rand`'s, so
/// the *values* differ; what has to match is the distribution, because the
/// initialisation is what decides whether a run can learn at all - and two
/// choices in particular carry real weight: the output projections start small
/// so a fresh block is close to the identity, and the held-key bias starts
/// negative so pressing keys has to be earned.
fn standard_normal(rng: &mut impl rand::Rng) -> f32 {
    let first: f32 = rng.random::<f32>().max(1e-7);
    let second: f32 = rng.random::<f32>();
    (-2.0 * first.ln()).sqrt() * (2.0 * std::f32::consts::PI * second).cos()
}

fn normal_values(count: usize, std: f32, rng: &mut impl rand::Rng) -> Vec<f32> {
    (0..count).map(|_| standard_normal(rng) * std).collect()
}

/// `torch.nn.init.orthogonal_`, written out: a matrix with orthonormal columns
/// (or rows, when it is wider than it is tall), scaled by `gain`.
///
/// The random matrix is orthogonalised by modified Gram-Schmidt. `torch` uses a
/// QR decomposition and then flips each column by the sign of `R`'s diagonal;
/// Gram-Schmidt leaves `R`'s diagonal positive by construction, so the two
/// conventions agree.
///
/// The gain is applied *after* the whole orthogonalisation, not inside it: the
/// projection a column is reduced against has to be a unit vector, or the
/// subtraction removes only `gain` of the component it is meant to remove.
fn orthogonal_values(rows: usize, cols: usize, gain: f32, rng: &mut impl rand::Rng) -> Vec<f32> {
    // `torch` orthogonalises the smaller dimension, working on the transpose
    // when the matrix is short and wide.
    let (tall, wide, transposed) = if rows < cols {
        (cols, rows, true)
    } else {
        (rows, cols, false)
    };
    let mut values = normal_values(tall * wide, 1.0, rng);
    for column in 0..wide {
        for previous in 0..column {
            let dot: f32 = (0..tall)
                .map(|row| values[row * wide + column] * values[row * wide + previous])
                .sum();
            for row in 0..tall {
                values[row * wide + column] -= dot * values[row * wide + previous];
            }
        }
        let norm: f32 = (0..tall)
            .map(|row| values[row * wide + column] * values[row * wide + column])
            .sum::<f32>()
            .sqrt()
            .max(1e-8);
        for row in 0..tall {
            values[row * wide + column] /= norm;
        }
    }
    for value in values.iter_mut() {
        *value *= gain;
    }
    if !transposed {
        return values;
    }
    // Back to (rows, cols).
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..tall {
        for column in 0..wide {
            out[column * tall + row] = values[row * wide + column];
        }
    }
    out
}

/// `torch.nn.init.kaiming_normal_` with `nonlinearity="relu"`: `N(0, sqrt(2/fan_in))`.
fn kaiming_values(count: usize, fan_in: usize, rng: &mut impl rand::Rng) -> Vec<f32> {
    let std = (2.0 / fan_in.max(1) as f32).sqrt();
    normal_values(count, std, rng)
}

impl<B: Backend> ActorCritic<B> {
    /// Initialise every weight the way the Python's constructor does.
    pub fn init(&mut self, rng: &mut impl rand::Rng, device: &B::Device) {
        let d = self.dims.clone();
        let embed = d.embed_dim;
        let inner = d.heads * d.head_dim;

        let tensor2 = |values: Vec<f32>, shape: [usize; 2]| -> Tensor<B, 2> {
            Tensor::from_data(TensorData::new(values, Shape::new(shape)), device)
        };
        let tensor1 = |values: Vec<f32>, shape: [usize; 1]| -> Tensor<B, 1> {
            Tensor::from_data(TensorData::new(values, Shape::new(shape)), device)
        };
        // `torch` stores a linear weight as (out, in) and this port stores it as
        // (in, out), so every weight below is drawn in the Python's orientation
        // and transposed on the way in.
        let transpose = |values: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; values.len()];
            for row in 0..rows {
                for column in 0..cols {
                    out[column * rows + row] = values[row * cols + column];
                }
            }
            out
        };

        // ---- patch embedding: one convolution ----
        let flattened = d.in_channels * d.patch * d.patch;
        let conv = kaiming_values(d.patch_width * flattened, flattened, rng);
        self.net.patch_embed.proj_w =
            Param::from_tensor(tensor2(transpose(&conv, d.patch_width, flattened), [flattened, d.patch_width]));
        self.net.patch_embed.proj_b =
            Param::from_tensor(tensor1(vec![0.0; d.patch_width], [d.patch_width]));
        let out_weight = orthogonal_values(embed, d.patch_width, 1.0, rng);
        self.net.patch_embed.out.weight =
            Param::from_tensor(tensor2(transpose(&out_weight, embed, d.patch_width), [d.patch_width, embed]));

        // ---- the memory slot for a step that has not happened yet ----
        self.net.bos = Param::from_tensor(Tensor::from_data(
            TensorData::new(normal_values(embed, 0.02, rng), Shape::new([1, 1, embed])),
            device,
        ));

        // ---- the transformer ----
        for block in self.net.blocks.iter_mut() {
            let qkv = orthogonal_values(3 * inner, embed, 1.0, rng);
            block.qkv.weight =
                Param::from_tensor(tensor2(transpose(&qkv, 3 * inner, embed), [embed, 3 * inner]));
            // The output projection starts small so a freshly initialised block
            // is close to the identity: a transformer whose residual branches
            // shout from the first step is one the critic never catches up with.
            let attention_out = orthogonal_values(embed, inner, 0.5, rng);
            block.attn_out.weight = Param::from_tensor(tensor2(
                transpose(&attention_out, embed, inner),
                [inner, embed],
            ));
            let gate = orthogonal_values(d.ffn_hidden, embed, 1.0, rng);
            block.ffn.gate.weight = Param::from_tensor(tensor2(
                transpose(&gate, d.ffn_hidden, embed),
                [embed, d.ffn_hidden],
            ));
            let up = orthogonal_values(d.ffn_hidden, embed, 1.0, rng);
            block.ffn.up.weight = Param::from_tensor(tensor2(
                transpose(&up, d.ffn_hidden, embed),
                [embed, d.ffn_hidden],
            ));
            let down = orthogonal_values(embed, d.ffn_hidden, 0.5, rng);
            block.ffn.down.weight = Param::from_tensor(tensor2(
                transpose(&down, embed, d.ffn_hidden),
                [d.ffn_hidden, embed],
            ));
        }

        // ---- the decision heads ----
        //
        // A near-uniform start for the decisions: a policy that begins already
        // certain of one action never explores its way out of it.
        let (weight, bias) = head_parameters::<B>(d.n_keys, embed, 0.01, -1.5, rng, device);
        self.net.hold_head.weight = weight;
        // The held-key head starts biased *against* pressing, so the keys a
        // policy does press have to be earned rather than being the default.
        // Without this, every key drifts up together under the top-k rule and
        // the bot ends up holding every key it owns, which moves it nowhere.
        self.net.hold_head.bias = Some(bias);
        let (weight, bias) = head_parameters::<B>(d.n_turn, embed, 0.01, 0.0, rng, device);
        self.net.turn_head.weight = weight;
        self.net.turn_head.bias = Some(bias);
        let (weight, bias) = head_parameters::<B>(d.n_speed, embed, 0.01, 0.0, rng, device);
        self.net.speed_head.weight = weight;
        self.net.speed_head.bias = Some(bias);
        let (weight, bias) = head_parameters::<B>(d.n_tap, embed, 0.01, 0.0, rng, device);
        self.net.tap_head.weight = weight;
        self.net.tap_head.bias = Some(bias);
        let (weight, bias) = head_parameters::<B>(1, embed, 1.0, 0.0, rng, device);
        self.net.value_head.weight = weight;
        self.net.value_head.bias = Some(bias);

        // ---- the curiosity head ----
        let input = orthogonal_values(d.transition_hidden, embed, 1.0, rng);
        self.net.transition.input.weight = Param::from_tensor(tensor2(
            transpose(&input, d.transition_hidden, embed),
            [embed, d.transition_hidden],
        ));
        self.net.transition.input.bias = Some(Param::from_tensor(tensor1(
            vec![0.0; d.transition_hidden],
            [d.transition_hidden],
        )));
        let output = orthogonal_values(embed, d.transition_hidden, 0.5, rng);
        self.net.transition.output.weight = Param::from_tensor(tensor2(
            transpose(&output, embed, d.transition_hidden),
            [d.transition_hidden, embed],
        ));
        self.net.transition.output.bias =
            Some(Param::from_tensor(tensor1(vec![0.0; embed], [embed])));
    }
}


/// Draw one decision head's weight and bias.
///
/// `orthogonal_` with a small gain is what keeps a fresh policy near-uniform.
/// The bias is built from host values rather than by arithmetic on a parameter:
/// `Param::from_tensor` marks a tensor as trainable, and Burn refuses that for a
/// tensor that is already an interior node of a graph.
fn head_parameters<B: Backend>(
    out: usize,
    embed: usize,
    gain: f32,
    bias_fill: f32,
    rng: &mut impl rand::Rng,
    device: &B::Device,
) -> (Param<Tensor<B, 2>>, Param<Tensor<B, 1>>) {
    let weight = orthogonal_values(out, embed, gain, rng);
    let mut transposed = vec![0.0f32; weight.len()];
    for row in 0..out {
        for column in 0..embed {
            transposed[column * out + row] = weight[row * embed + column];
        }
    }
    (
        Param::from_tensor(Tensor::from_data(
            TensorData::new(transposed, Shape::new([embed, out])),
            device,
        )),
        Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![bias_fill; out], Shape::new([out])),
            device,
        )),
    )
}

fn with_norm<B: Backend>(
    weights: &Weights,
    name: &str,
    device: &B::Device,
    size: usize,
) -> anyhow::Result<LayerNorm<B>> {
    let mut norm = LayerNormConfig::new(size)
        .with_epsilon(LAYER_NORM_EPS)
        .init(device);
    norm.gamma = weights.param(&format!("{name}.weight"), device, &[size])?;
    norm.beta = Some(weights.param(&format!("{name}.bias"), device, &[size])?);
    Ok(norm)
}

/// The lower-triangular attention mask: `0` where a position may attend, and a
/// large negative number where it may not.
fn causal_mask<B: Backend>(tokens: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut values = vec![0.0f32; tokens * tokens];
    for row in 0..tokens {
        for column in 0..tokens {
            if column > row {
                values[row * tokens + column] = MASKED_ATTENTION;
            }
        }
    }
    Tensor::from_data(TensorData::new(values, [tokens, tokens]), device)
}

/// `log(1 + exp(x))`, written the stable way round.
///
/// `torch.nn.functional.softplus` switches to the identity above a threshold of
/// 20 for stability; `max(x, 0) + log(1 + exp(-|x|))` is the same function and
/// is stable everywhere, so there is no threshold to get wrong. Burn's own
/// `softplus` has no threshold and overflows for large positive inputs, which
/// would turn a very negative logit into an infinity rather than into `-z`.
pub fn softplus_stable<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let log_term = x.clone().abs().neg().exp().add_scalar(1.0).log();
    x.clamp_min(0.0) + log_term
}

/// `log softmax` over one dimension, as a tensor.
fn log_softmax<B: Backend, const D: usize>(x: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let max = x.clone().max_dim(dim);
    let shifted = x - max.clone();
    let log_norm = shifted.clone().exp().sum_dim(dim).log();
    shifted - log_norm
}

/// Log-probability of one category, as a `(1,)` tensor.
fn categorical_log_prob<B: Backend>(
    logits: &Tensor<B, 2>,
    index: usize,
) -> Tensor<B, 1> {
    let [rows, classes] = logits.dims();
    debug_assert_eq!(rows, 1, "categorical scoring here is per control step");
    let values = logits
        .clone()
        .into_data()
        .to_vec::<f32>()
        .expect("logits are f32");
    let lp = log_softmax_host(&values[..classes])[index];
    Tensor::<B, 1>::from_data(TensorData::new(vec![lp], [1]), &logits.device())
}

/// Log-probability of the recorded category, for every row: `(rows,)`.
fn categorical_log_prob_indexed<B: Backend>(
    logits: &Tensor<B, 2>,
    index: Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    let [rows, _classes] = logits.dims();
    let log_probs = log_softmax(logits.clone(), 1);
    log_probs
        .gather(1, index.reshape([rows, 1]))
        .reshape([rows])
}

/// Entropy of a categorical head, `(rows,)`.
fn categorical_entropy<B: Backend>(logits: &Tensor<B, 2>) -> Tensor<B, 1> {
    let [rows, _classes] = logits.dims();
    let log_probs = log_softmax(logits.clone(), 1);
    let probabilities = log_probs.clone().exp();
    (probabilities * log_probs)
        .sum_dim(1)
        .reshape([rows])
        .neg()
}

/// A tensor's values, for the cases where a decision is made on the host.
fn logits_to_host<B: Backend>(logits: &Tensor<B, 2>) -> Vec<f32> {
    logits
        .clone()
        .into_data()
        .to_vec::<f32>()
        .expect("logits are f32")
}

/// Sample one index from `scores`, treating them as unnormalised logits.
fn sample_categorical(scores: &[f32], rng: &mut impl rand::Rng) -> usize {
    let probabilities = softmax_host(scores);
    let draw: f32 = rng.random();
    let mut cumulative = 0.0f32;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if draw < cumulative {
            return index;
        }
    }
    probabilities.len() - 1
}

fn softmax_host(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    for value in exps.iter_mut() {
        *value /= total;
    }
    exps
}

fn log_softmax_host(scores: &[f32]) -> Vec<f32> {
    let probabilities = softmax_host(scores);
    probabilities.iter().map(|p| p.max(1e-45).ln()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;

    type TestBackend = Flex;

    fn test_model(cfg: &Config) -> ActorCritic<TestBackend> {
        let device = Default::default();
        let mut model =
            ActorCritic::<TestBackend>::new(cfg, [10, 5, 5, 4], cfg.channels(), &device);
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(7);
        model.init(&mut rng, &device);
        model
    }

    /// `W W^T` or `W^T W` over the smaller dimension: an orthogonal matrix has
    /// the identity there, scaled by the gain squared.
    fn gram_error(values: &[f32], rows: usize, columns: usize, gain: f32) -> f32 {
        let mut worst = 0.0f32;
        if rows <= columns {
            for i in 0..rows {
                for j in 0..rows {
                    let dot: f32 = (0..columns)
                        .map(|k| values[i * columns + k] * values[j * columns + k])
                        .sum();
                    let expected = if i == j { gain * gain } else { 0.0 };
                    worst = worst.max((dot - expected).abs());
                }
            }
        } else {
            for i in 0..columns {
                for j in 0..columns {
                    let dot: f32 = (0..rows)
                        .map(|k| values[k * columns + i] * values[k * columns + j])
                        .sum();
                    let expected = if i == j { gain * gain } else { 0.0 };
                    worst = worst.max((dot - expected).abs());
                }
            }
        }
        worst
    }

    #[test]
    fn orthogonal_values_makes_orthonormal_columns_or_rows() {
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(3);
        // Tall: columns are orthonormal.
        let tall = orthogonal_values(192, 96, 1.0, &mut rng);
        assert!(
            gram_error(&tall, 192, 96, 1.0) < 1e-4,
            "tall error {}",
            gram_error(&tall, 192, 96, 1.0)
        );
        // Wide: rows are orthonormal, because the smaller dimension is what
        // gets orthogonalised.
        let wide = orthogonal_values(96, 192, 1.0, &mut rng);
        assert!(
            gram_error(&wide, 96, 192, 1.0) < 1e-4,
            "wide error {}",
            gram_error(&wide, 96, 192, 1.0)
        );
        // Square, and with a gain.
        let square = orthogonal_values(96, 96, 0.5, &mut rng);
        assert!(
            gram_error(&square, 96, 96, 0.5) < 1e-4,
            "square error {}",
            gram_error(&square, 96, 96, 0.5)
        );
    }

    #[test]
    fn the_initialisation_is_orthogonal_where_the_python_says_it_is() {
        let cfg = Config::default();
        let model = test_model(&cfg);
        let block = &model.net.blocks[0];

        // The Python's gains: 1.0 for the feed-forward's two input projections,
        // 0.5 for every output projection.
        let gate = block
            .ffn
            .gate
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(
            gram_error(&gate, cfg.embed_dim, cfg.ffn_hidden, 1.0) < 1e-4,
            "the SwiGLU gate should be orthogonal with gain 1.0"
        );
        let down = block
            .ffn
            .down
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(
            gram_error(&down, cfg.ffn_hidden, cfg.embed_dim, 0.5) < 1e-4,
            "the SwiGLU down projection should be orthogonal with gain 0.5 (error {})",
            gram_error(&down, cfg.ffn_hidden, cfg.embed_dim, 0.5)
        );
        // `qkv` is the one projection whose two axes differ in meaning: the
        // stored shape is (embed, 3 * inner), and it is the 96 rows that carry
        // the orthonormality.
        let inner = cfg.attention_heads * (cfg.embed_dim / cfg.attention_heads);
        let qkv = block.qkv.weight.val().into_data().to_vec::<f32>().unwrap();
        assert_eq!(qkv.len(), cfg.embed_dim * 3 * inner);
        assert!(
            gram_error(&qkv, cfg.embed_dim, 3 * inner, 1.0) < 1e-4,
            "the qkv projection should be orthogonal with gain 1.0 (error {})",
            gram_error(&qkv, cfg.embed_dim, 3 * inner, 1.0)
        );
    }

    #[test]
    fn the_held_key_head_starts_biased_against_pressing() {
        let cfg = Config::default();
        let model = test_model(&cfg);
        let bias = model
            .net
            .hold_head
            .bias
            .as_ref()
            .expect("the held-key head has a bias")
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(
            bias.iter().all(|value| (value + 1.5).abs() < 1e-6),
            "{bias:?}"
        );

        // With the bias at -1.5 an undecided policy presses nothing, which is
        // what stops four keys being held from the very first step.
        let probabilities: Vec<f32> = bias
            .iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
            .collect();
        assert!(
            probabilities.iter().all(|p| *p < 0.2),
            "a fresh policy should not want to hold keys: {probabilities:?}"
        );
    }

    #[test]
    fn the_mouse_step_stays_inside_its_bounds() {
        let cfg = Config::default();
        let mut model = test_model(&cfg);
        assert_eq!(model.mouse_step_value(), cfg.mouse_turn_start);
        for _ in 0..200 {
            model.nudge_mouse_step(1.5);
        }
        assert_eq!(model.mouse_step_value(), cfg.mouse_turn_max);
        for _ in 0..200 {
            model.nudge_mouse_step(0.5);
        }
        assert_eq!(model.mouse_step_value(), cfg.mouse_turn_min);
        model.set_mouse_step(Some(-5.0));
        assert_eq!(model.mouse_step_value(), cfg.mouse_turn_min);
    }
}

