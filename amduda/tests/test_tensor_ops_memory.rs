use amduda::amduda_core::memory_tiering::{DeviceCapabilities, MemoryManager, MemoryTier};
use amduda::amduda_core::tensor_ops::{CpuFallback, TensorOps};
use serial_test::serial;

#[test]
#[serial]
fn tensor_ops_allocation_respects_tiers() {
    std::env::set_var("AMDUDA_HAS_GPU", "1");
    std::env::set_var("AMDUDA_HAS_NVME", "1");
    let caps = DeviceCapabilities::detect();
    let mut mgr = MemoryManager::new_with_limits(caps, 64, 64, 256);
    let ops = CpuFallback;

    let a = vec![1.0f32; 4];
    let b = vec![1.0f32; 4];
    let result = ops.matmul(&a, &b, 2, 2, 2);
    let bytes = result.len() * std::mem::size_of::<f32>();

    let tier = mgr.allocate(bytes);
    assert_eq!(tier, MemoryTier::Gpu);

    mgr.mark_cold(MemoryTier::Gpu, bytes);
    assert_eq!(mgr.allocate(64), MemoryTier::Gpu);
    assert_eq!(mgr.usage(), (64, bytes, 0));
}
