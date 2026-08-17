//! `LiveWindow` must tolerate concurrent writes from many threads without data
//! races or panics.

use driftwatch::{LiveWindow, WindowMode};
use std::sync::Arc;
use std::thread;

#[test]
fn concurrent_pushes_do_not_panic_and_respect_capacity() {
    const THREADS: usize = 16;
    const PER_THREAD: usize = 1000;
    const CAP: usize = 256;

    let window = Arc::new(LiveWindow::new(WindowMode::Count(CAP)));
    let mut handles = Vec::new();
    for t in 0..THREADS {
        let w = Arc::clone(&window);
        handles.push(thread::spawn(move || {
            for i in 0..PER_THREAD {
                w.push(vec![(t * PER_THREAD + i) as f64]);
            }
        }));
    }
    for h in handles {
        h.join().expect("worker thread panicked");
    }

    // The ring buffer never exceeds its capacity, regardless of interleaving.
    assert_eq!(window.len(), CAP);
    assert_eq!(window.snapshot().len(), CAP);
}

#[test]
fn concurrent_push_and_snapshot_are_consistent() {
    let window = Arc::new(LiveWindow::new(WindowMode::Count(100)));
    let writer = {
        let w = Arc::clone(&window);
        thread::spawn(move || {
            for i in 0..5000 {
                w.push(vec![i as f64]);
            }
        })
    };
    let reader = {
        let w = Arc::clone(&window);
        thread::spawn(move || {
            for _ in 0..500 {
                // Snapshot must always be a valid, bounded view.
                assert!(w.snapshot().len() <= 100);
            }
        })
    };
    writer.join().unwrap();
    reader.join().unwrap();
}
