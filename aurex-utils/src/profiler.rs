//! Minimal profiler example.

pub fn profile<F: FnOnce()>(f: F) {
    let start = std::time::Instant::now();
    f();
    let dur = start.elapsed();
    println!("Execution took {:?}", dur);
}
