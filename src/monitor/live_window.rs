//! `LiveWindow`: a thread-safe, bounded buffer of live feature vectors.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A single row of live feature values (one observation across all features).
pub type FeatureVector = Vec<f64>;

/// How a [`LiveWindow`] decides which entries to retain.
#[derive(Clone, Copy, Debug)]
pub enum WindowMode {
    /// Keep the most recent `N` entries; evict the oldest when full.
    Count(usize),
    /// Keep entries pushed within the last [`Duration`]; older ones are dropped.
    /// More meaningful than a fixed count when traffic volume varies.
    Duration(Duration),
}

struct Entry {
    features: FeatureVector,
    at: Instant,
}

/// A bounded, thread-safe ring buffer accumulating live feature vectors.
///
/// This is the piece that makes `driftwatch` usable against a real serving
/// system rather than only static before/after datasets: request-handling
/// threads call [`push`](LiveWindow::push) concurrently, and a scheduler of the
/// caller's choosing calls [`snapshot`](LiveWindow::snapshot) periodically to
/// feed a [`DatasetMonitor`](crate::DatasetMonitor).
///
/// It is deliberately synchronous and framework-agnostic — it builds in no
/// scheduler and forces no async runtime on the core crate. Let the integration
/// layer (e.g. a `tokio::time::interval` loop in an `axum` server) own the
/// cadence.
///
/// # Example
/// ```
/// use driftwatch::{LiveWindow, WindowMode};
///
/// let window = LiveWindow::new(WindowMode::Count(2));
/// window.push(vec![1.0]);
/// window.push(vec![2.0]);
/// window.push(vec![3.0]); // evicts [1.0]
/// assert_eq!(window.snapshot(), vec![vec![2.0], vec![3.0]]);
/// ```
pub struct LiveWindow {
    buffer: Mutex<VecDeque<Entry>>,
    mode: WindowMode,
}

impl LiveWindow {
    /// Create an empty window with the given retention mode.
    pub fn new(mode: WindowMode) -> Self {
        Self {
            buffer: Mutex::new(VecDeque::new()),
            mode,
        }
    }

    /// Push one feature vector, evicting whatever the mode no longer retains.
    ///
    /// Safe to call concurrently from many threads.
    pub fn push(&self, features: FeatureVector) {
        self.push_at(features, Instant::now());
    }

    /// Push with an explicit timestamp (used internally and for deterministic
    /// tests of time-based eviction).
    pub(crate) fn push_at(&self, features: FeatureVector, at: Instant) {
        let mut buf = self.buffer.lock().expect("LiveWindow mutex poisoned");
        buf.push_back(Entry { features, at });
        evict(&mut buf, self.mode, at);
    }

    /// Snapshot the currently-retained feature vectors, oldest first.
    ///
    /// For a time-based window this also evicts anything now expired, so the
    /// snapshot reflects the window as of the call.
    pub fn snapshot(&self) -> Vec<FeatureVector> {
        let mut buf = self.buffer.lock().expect("LiveWindow mutex poisoned");
        evict(&mut buf, self.mode, Instant::now());
        buf.iter().map(|e| e.features.clone()).collect()
    }

    /// Extract a single feature column across all retained entries, given its
    /// index within each feature vector. Values from vectors too short to
    /// contain the index are skipped.
    ///
    /// Convenient for feeding a per-feature slice to
    /// [`DatasetMonitor::check`](crate::DatasetMonitor::check).
    pub fn column(&self, index: usize) -> Vec<f64> {
        let mut buf = self.buffer.lock().expect("LiveWindow mutex poisoned");
        evict(&mut buf, self.mode, Instant::now());
        buf.iter()
            .filter_map(|e| e.features.get(index).copied())
            .collect()
    }

    /// Number of entries currently retained (after any time-based eviction).
    pub fn len(&self) -> usize {
        let mut buf = self.buffer.lock().expect("LiveWindow mutex poisoned");
        evict(&mut buf, self.mode, Instant::now());
        buf.len()
    }

    /// Whether the window currently holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all entries.
    pub fn clear(&self) {
        self.buffer
            .lock()
            .expect("LiveWindow mutex poisoned")
            .clear();
    }
}

/// Apply the retention policy in place. `now` is the reference time for
/// duration-based eviction.
fn evict(buf: &mut VecDeque<Entry>, mode: WindowMode, now: Instant) {
    match mode {
        WindowMode::Count(n) => {
            while buf.len() > n {
                buf.pop_front();
            }
        }
        WindowMode::Duration(d) => {
            while let Some(front) = buf.front() {
                if now.duration_since(front.at) > d {
                    buf.pop_front();
                } else {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_mode_evicts_oldest() {
        let window = LiveWindow::new(WindowMode::Count(3));
        for i in 0..5 {
            window.push(vec![i as f64]);
        }
        assert_eq!(window.snapshot(), vec![vec![2.0], vec![3.0], vec![4.0]]);
        assert_eq!(window.len(), 3);
    }

    #[test]
    fn column_extracts_feature() {
        let window = LiveWindow::new(WindowMode::Count(10));
        window.push(vec![1.0, 10.0]);
        window.push(vec![2.0, 20.0]);
        assert_eq!(window.column(0), vec![1.0, 2.0]);
        assert_eq!(window.column(1), vec![10.0, 20.0]);
    }

    #[test]
    fn duration_mode_evicts_expired() {
        // Drive `evict` directly with synthetic timestamps for determinism.
        let t0 = Instant::now();
        let mut buf = VecDeque::new();
        buf.push_back(Entry {
            features: vec![1.0],
            at: t0,
        });
        buf.push_back(Entry {
            features: vec![2.0],
            at: t0 + Duration::from_secs(10),
        });
        // "Now" is 12s after t0: the first entry (12s old) is past a 5s window,
        // the second (2s old) is retained.
        evict(
            &mut buf,
            WindowMode::Duration(Duration::from_secs(5)),
            t0 + Duration::from_secs(12),
        );
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0].features, vec![2.0]);
    }
}
