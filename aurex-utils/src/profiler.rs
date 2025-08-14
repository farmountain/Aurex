//! Lightweight profiling utilities.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use sysinfo::System;

/// Collected metrics for a single operation.
#[derive(Debug, Clone)]
pub struct OpRecord {
    pub name: &'static str,
    pub duration: Duration,
    pub memory_bytes: u64,
    pub gpu_counters: HashMap<String, u64>,
}

/// Profiler holding per-operation records.
#[derive(Default)]
pub struct Profiler {
    records: Vec<OpRecord>,
}

impl Profiler {
    /// Create a new empty profiler.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Profile a named operation by executing `f` and recording metrics.
    pub fn profile<F, R>(&mut self, name: &'static str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start_time = Instant::now();
        let start_mem = current_mem();
        let start_gpu = gpu_counters();
        let result = f();
        let end_time = Instant::now();
        let end_mem = current_mem();
        let end_gpu = gpu_counters();

        self.records.push(OpRecord {
            name,
            duration: end_time - start_time,
            memory_bytes: end_mem.saturating_sub(start_mem),
            gpu_counters: diff_counters(&start_gpu, &end_gpu),
        });

        result
    }

    /// Access collected operation records.
    pub fn records(&self) -> &[OpRecord] {
        &self.records
    }
}

/// Macro to profile an individual `TensorOps` call.
#[macro_export]
macro_rules! profile_tensor_op {
    ($prof:expr, $name:expr, $body:expr) => {{
        $prof.profile($name, || $body)
    }};
}

/// Macro to profile a runtime step.
#[macro_export]
macro_rules! profile_runtime_step {
    ($prof:expr, $name:expr, $body:expr) => {{
        $prof.profile($name, || $body)
    }};
}

/// Convenience wrapper for profiling a closure and printing the result.
pub fn profile<F: FnOnce()>(f: F) {
    let mut prof = Profiler::new();
    prof.profile("run", f);
    for record in prof.records() {
        println!(
            "{}: {:?} (mem: {} B, gpu: {:?})",
            record.name, record.duration, record.memory_bytes, record.gpu_counters
        );
    }
}

fn current_mem() -> u64 {
    let pid = sysinfo::get_current_pid().unwrap();
    let mut sys = System::new_all();
    sys.refresh_process(pid);
    sys.process(pid).map(|p| p.memory() * 1024).unwrap_or(0)
}

fn gpu_counters() -> HashMap<String, u64> {
    // Placeholder for real GPU counter collection.
    HashMap::new()
}

fn diff_counters(start: &HashMap<String, u64>, end: &HashMap<String, u64>) -> HashMap<String, u64> {
    let mut diff = HashMap::new();
    for (k, v_end) in end {
        let v_start = start.get(k).copied().unwrap_or(0);
        diff.insert(k.clone(), v_end.saturating_sub(v_start));
    }
    diff
}
