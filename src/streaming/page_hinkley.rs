//! The Page-Hinkley online change-point detector.

/// The direction of a detected change in a signal's mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageHinkleyChange {
    /// The mean shifted upward.
    Increase,
    /// The mean shifted downward.
    Decrease,
}

/// A Page-Hinkley test: classic online detection of a change in the mean of a
/// scalar signal, in bounded memory and O(1) per update.
///
/// This is the genuinely-online paradigm the windowed monitors can't replicate:
/// feed it a live scalar (a running score, a prediction, a latency) one value at
/// a time and it signals when the mean shifts by more than `delta`, once the
/// accumulated evidence crosses `lambda`. On detection it resets, so it can
/// catch successive change-points.
///
/// * `delta` — the magnitude of change to tolerate before accumulating evidence
///   (a small slack that makes the test robust to noise).
/// * `lambda` — the detection threshold: larger means fewer false alarms but
///   slower detection.
///
/// # Example
/// ```
/// use driftwatch::streaming::{PageHinkleyDetector, PageHinkleyChange};
///
/// let mut ph = PageHinkleyDetector::new(0.5, 50.0);
/// let mut fired = None;
/// for i in 0..2000 {
///     let x = if i < 1000 { 0.0 } else { 5.0 }; // mean jumps at i = 1000
///     if let Some(change) = ph.update(x) {
///         fired = Some((i, change));
///         break;
///     }
/// }
/// let (at, change) = fired.expect("should detect the shift");
/// assert!(at >= 1000);
/// assert_eq!(change, PageHinkleyChange::Increase);
/// ```
#[derive(Clone, Debug)]
pub struct PageHinkleyDetector {
    delta: f64,
    lambda: f64,
    n: u64,
    mean: f64,
    m_inc: f64,
    min_inc: f64,
    m_dec: f64,
    max_dec: f64,
}

impl PageHinkleyDetector {
    /// Create a detector with the given tolerance `delta` and threshold `lambda`.
    pub fn new(delta: f64, lambda: f64) -> Self {
        Self {
            delta,
            lambda,
            n: 0,
            mean: 0.0,
            m_inc: 0.0,
            min_inc: 0.0,
            m_dec: 0.0,
            max_dec: 0.0,
        }
    }

    /// Feed one value. Returns the change direction if a change-point is detected
    /// at this step (after which the detector resets), otherwise `None`.
    pub fn update(&mut self, x: f64) -> Option<PageHinkleyChange> {
        if !x.is_finite() {
            return None;
        }
        self.n += 1;
        // Running mean over the samples seen since the last reset.
        self.mean += (x - self.mean) / self.n as f64;

        // Upward cumulative sum and its running minimum.
        self.m_inc += x - self.mean - self.delta;
        self.min_inc = self.min_inc.min(self.m_inc);
        // Downward cumulative sum and its running maximum.
        self.m_dec += x - self.mean + self.delta;
        self.max_dec = self.max_dec.max(self.m_dec);

        let ph_inc = self.m_inc - self.min_inc;
        let ph_dec = self.max_dec - self.m_dec;

        if ph_inc > self.lambda {
            self.reset();
            Some(PageHinkleyChange::Increase)
        } else if ph_dec > self.lambda {
            self.reset();
            Some(PageHinkleyChange::Decrease)
        } else {
            None
        }
    }

    /// Number of values seen since construction or the last detection.
    pub fn samples_since_reset(&self) -> u64 {
        self.n
    }

    /// Clear all accumulated state (called automatically on detection).
    pub fn reset(&mut self) {
        self.n = 0;
        self.mean = 0.0;
        self.m_inc = 0.0;
        self.min_inc = 0.0;
        self.m_dec = 0.0;
        self.max_dec = 0.0;
    }
}
