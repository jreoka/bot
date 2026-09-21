//! Turning a stream of screenshots into the tensor the model sees, and holding
//! the loop to a target rate. Ported from `capture.py`'s `FrameStack` and
//! `Pacer`.
//!
//! Nothing here touches an OS: a frame arrives as raw pixels from whichever
//! platform backend grabbed it, and what comes out is the observation the model
//! was built for. That is deliberate - the observation pipeline is the same on
//! all three systems, and the parity work in stage 1 covers it.

use std::time::{Duration, Instant};

use crate::vision::Observation;

/// One grabbed frame, as the platform handed it over.
///
/// The channel order is BGRA everywhere, because that is what a Windows
/// `PrintWindow` grab produces through `GetDIBits` and what the equivalents
/// produce too; the luma weights below are written in that order.
#[derive(Debug, Clone, Default)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub bgra: Vec<u8>,
}

impl Frame {
    pub fn new(width: usize, height: usize, bgra: Vec<u8>) -> Self {
        Self {
            width,
            height,
            bgra,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || self.bgra.len() < self.width * self.height * 4
    }
}

/// Turns a stream of screenshots into a fixed-shape tensor.
///
/// Output is `(frame_stack + 1, size, size)` float32 in `[0, 1]`:
///
/// ```text
/// channels [0 .. n-1]  the last n frames, oldest first, as grayscale
/// channel  n           the motion channel, from the oldest and newest frames
/// ```
///
/// The difference channel costs a subtraction, and it is what lets the model see
/// motion without having to infer it from per-frame appearance alone, or from a
/// stack long enough to make the motion obvious.
pub struct FrameStack {
    pub size: usize,
    pub stack: usize,
    /// Grey frames plus the motion channel: what the model's input width is.
    pub channels: usize,
    /// The ring of grey frames, oldest first after the first `filled` entries.
    buffer: Vec<u8>,
    filled: usize,
    out: Observation,
    /// Work buffers, allocated once.
    luma: Vec<f32>,
    last_width: usize,
    last_height: usize,
}

impl FrameStack {
    pub fn new(size: usize, stack: usize) -> Self {
        let size = size.max(1);
        let stack = stack.max(1);
        Self {
            size,
            stack,
            channels: stack + 1,
            buffer: vec![0; stack * size * size],
            filled: 0,
            out: Observation::zeros(stack + 1, size, size),
            luma: Vec::new(),
            last_width: 0,
            last_height: 0,
        }
    }

    pub fn clear(&mut self) {
        self.buffer.fill(0);
        self.filled = 0;
    }

    /// How many frames have been pushed since the last clear, capped at `stack`.
    pub fn filled(&self) -> usize {
        self.filled
    }

    /// BGRA at source resolution -> `(size, size)` grey.
    ///
    /// The Python resizes in colour and then converts to grey with
    /// `cv2.cvtColor`; luma is a linear combination of the channels, so
    /// converting first and resizing after is the same picture for a fraction of
    /// the work. The weights are Rec. 601 in BGR order, which is what
    /// `COLOR_BGR2GRAY` uses.
    fn gray_into(&mut self, frame: &Frame, target: &mut [u8]) {
        if frame.width != self.last_width || frame.height != self.last_height {
            self.luma = vec![0.0; frame.width * frame.height];
            self.last_width = frame.width;
            self.last_height = frame.height;
        }
        for (index, pixel) in frame.bgra.chunks_exact(4).enumerate() {
            let blue = pixel[0] as f32;
            let green = pixel[1] as f32;
            let red = pixel[2] as f32;
            self.luma[index] = 0.114 * blue + 0.587 * green + 0.299 * red;
        }
        let resized = crate::vision::area_resize(
            &self.luma,
            frame.height,
            frame.width,
            self.size,
            self.size,
        );
        for (slot, value) in target.iter_mut().zip(resized) {
            *slot = value.round().clamp(0.0, 255.0) as u8;
        }
    }

    /// Add one frame and return the observation tensor.
    pub fn push(&mut self, frame: &Frame) -> &Observation {
        let plane = self.size * self.size;
        // Roll by one: the oldest frame falls off the front.
        if self.stack > 1 {
            self.buffer.copy_within(plane..self.stack * plane, 0);
        }
        let mut gray = vec![0u8; plane];
        self.gray_into(frame, &mut gray);
        self.buffer[(self.stack - 1) * plane..self.stack * plane].copy_from_slice(&gray);
        self.filled = (self.filled + 1).min(self.stack);
        self.observe()
    }

    /// The current observation, without advancing the stack.
    pub fn observe(&mut self) -> &Observation {
        let plane = self.size * self.size;
        let scale = 1.0 / 255.0;
        for channel in 0..self.stack {
            let source = &self.buffer[channel * plane..(channel + 1) * plane];
            let target = self.out.channel_mut(channel);
            for (slot, value) in target.iter_mut().zip(source) {
                *slot = *value as f32 * scale;
            }
        }
        // The motion channel. The Python subtracts into a `uint8` array, so a
        // large *decrease* in brightness wraps around to a small number rather
        // than showing up as a large change. That is reproduced here: it is what
        // the model was built on, and the synthetic game uses a true absolute
        // difference, so the two are deliberately different.
        let newest = &self.buffer[(self.stack - 1) * plane..self.stack * plane];
        let oldest = &self.buffer[..plane];
        let target = self.out.channel_mut(self.stack);
        for (slot, (new, old)) in target.iter_mut().zip(newest.iter().zip(oldest)) {
            *slot = new.wrapping_sub(*old) as f32 * scale;
        }
        &self.out
    }

    /// The observation as the model wants it: `(channels, size, size)`.
    pub fn observation(&self) -> &Observation {
        &self.out
    }
}

/// Holds the control loop at a target rate.
///
/// When the machine cannot keep up, the schedule slips instead of sprinting to
/// catch up: a bot that owes twenty frames must not deliver twenty steps of
/// input back to back, because the game would see a burst of input with no time
/// to simulate in between.
pub struct Pacer {
    pub target_fps: f32,
    pub extra_sleep: f32,
    /// How many of these steps one decision is held for. Only used to label the
    /// rate honestly: with `action_repeat = 2` the bot takes 20 game steps a
    /// second and 10 decisions a second, and reporting one number for both is
    /// how "it says 20 steps a second" ends up meaning nothing.
    pub decisions_per_step: usize,
    step: Duration,
    next: Instant,
    last_report: Instant,
    count: u64,
    pub overruns: u64,
    pub achieved_hz: f64,
}

impl Pacer {
    pub fn new(target_fps: f32, extra_sleep: f32, decisions_per_step: usize) -> Self {
        let target_fps = target_fps.max(0.0);
        let now = Instant::now();
        Self {
            target_fps,
            extra_sleep: extra_sleep.max(0.0),
            decisions_per_step: decisions_per_step.max(1),
            step: if target_fps > 0.0 {
                Duration::from_secs_f64(1.0 / target_fps as f64)
            } else {
                Duration::ZERO
            },
            next: now,
            last_report: now,
            count: 0,
            overruns: 0,
            achieved_hz: 0.0,
        }
    }

    pub fn reset(&mut self) {
        let now = Instant::now();
        self.next = now;
        self.last_report = now;
        self.count = 0;
    }

    /// Wait for this step's slot.
    pub fn tick(&mut self) {
        if !self.step.is_zero() {
            self.next += self.step;
            let now = Instant::now();
            if self.next > now {
                std::thread::sleep(self.next - now);
            } else if now.duration_since(self.next) > self.step {
                // Behind schedule: resynchronise rather than fire a burst.
                self.overruns += 1;
                self.next = now;
            }
        }
        if self.extra_sleep > 0.0 {
            std::thread::sleep(Duration::from_secs_f32(self.extra_sleep));
        }
        self.count += 1;
    }

    /// The `[Perf]` line, on the interval it is due, or `None`.
    pub fn report(&mut self, every: f64) -> Option<String> {
        let now = Instant::now();
        let window = now.duration_since(self.last_report).as_secs_f64();
        if window < every {
            return None;
        }
        self.achieved_hz = self.count as f64 / window.max(1e-6);
        self.count = 0;
        self.last_report = now;
        let mut line = format!("[Perf] {:5.1} game step(s)/s", self.achieved_hz);
        if self.decisions_per_step > 1 {
            line += &format!(
                "  = {:.1} decision(s)/s at action_repeat {}",
                self.achieved_hz / self.decisions_per_step as f64,
                self.decisions_per_step
            );
        }
        if self.target_fps > 0.0 {
            line += &format!(" (target {:.0})", self.target_fps);
            if self.achieved_hz < self.target_fps as f64 * 0.6 {
                line += "  <- behind target; lower frame_size or action_repeat";
            }
        }
        Some(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: usize, height: usize, value: u8) -> Frame {
        Frame::new(width, height, vec![value; width * height * 4])
    }

    #[test]
    fn a_frame_stack_produces_the_shape_the_model_wants() {
        let mut stack = FrameStack::new(8, 4);
        let observation = stack.push(&solid(64, 64, 128));
        assert_eq!(observation.shape(), (5, 8, 8));
        assert!(observation.data.iter().all(|value| *value >= 0.0 && *value <= 1.0));
    }

    #[test]
    fn the_stack_starts_black_and_fills_from_the_newest_end() {
        let mut stack = FrameStack::new(4, 3);
        let observation = stack.push(&solid(32, 32, 200)).clone();
        assert_eq!(stack.filled(), 1);
        // The real stack, unlike the synthetic game's, does not repeat the first
        // frame: the older slots are black until real frames arrive.
        assert_eq!(observation.channel(0)[0], 0.0);
        assert_eq!(observation.channel(1)[0], 0.0);
        let expected = (0.114 * 200.0 + 0.587 * 200.0 + 0.299 * 200.0) / 255.0;
        assert!(
            (observation.channel(2)[0] - expected).abs() < 0.01,
            "{}",
            observation.channel(2)[0]
        );
        // And the motion channel is the change from black to that frame.
        assert!((observation.channel(3)[0] - expected).abs() < 0.01);
    }

    #[test]
    fn the_oldest_frame_falls_off_the_front() {
        let mut stack = FrameStack::new(2, 2);
        stack.push(&solid(16, 16, 0));
        stack.push(&solid(16, 16, 100));
        let observation = stack.push(&solid(16, 16, 255)).clone();
        // The window now holds the 100 and 255 frames, oldest first.
        assert!((observation.channel(0)[0] - 100.0 / 255.0).abs() < 0.01);
        assert!((observation.channel(1)[0] - 1.0).abs() < 0.01);
        // And the motion channel is the change between them.
        assert!((observation.channel(2)[0] - 155.0 / 255.0).abs() < 0.01);
    }

    #[test]
    fn the_motion_channel_wraps_like_the_python_does() {
        // A bright frame followed by a black one: the Python's `uint8`
        // subtraction wraps, so the change reads as small rather than large.
        let mut stack = FrameStack::new(2, 2);
        stack.push(&solid(16, 16, 255));
        let observation = stack.push(&solid(16, 16, 0));
        let diff = observation.channel(2)[0];
        assert!(
            diff < 0.01,
            "expected the wraparound the Python produces, got {diff}"
        );
    }

    #[test]
    fn the_pacer_resynchronises_after_a_stall_instead_of_bursting() {
        // A 10 ms step, then a stall five slots long: the next tick must give up
        // the lost time rather than firing five steps back to back.
        let mut pacer = Pacer::new(100.0, 0.0, 1);
        std::thread::sleep(Duration::from_millis(50));
        pacer.tick();
        assert_eq!(pacer.overruns, 1, "a missed schedule must be counted");
        // And it is back on schedule: the following tick waits for its slot.
        let started = Instant::now();
        pacer.tick();
        assert!(
            started.elapsed() >= Duration::from_millis(5),
            "the next tick should wait ~10 ms, took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn the_pacer_reports_at_its_interval_and_not_before() {
        let mut pacer = Pacer::new(0.0, 0.0, 1);
        pacer.tick();
        assert!(pacer.report(15.0).is_none());
        let line = pacer.report(0.0).expect("a report is due immediately");
        assert!(line.starts_with("[Perf]"), "{line}");
    }
}
