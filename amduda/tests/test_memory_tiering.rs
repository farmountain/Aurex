use amduda::amduda_core::memory_tiering::{DeviceCapabilities, MemoryManager, MemoryTier};
use serial_test::serial;

#[test]
#[serial]
fn gpu_caching_and_migration() {
    std::env::set_var("AMDUDA_HAS_GPU", "1");
    std::env::set_var("AMDUDA_HAS_NVME", "1");
    let caps = DeviceCapabilities::detect();
    let mut mgr = MemoryManager::new_with_limits(caps, 64, 64, 256);

    assert_eq!(mgr.allocate(32), MemoryTier::Gpu);
    assert_eq!(mgr.usage(), (32, 0, 0));

    assert_eq!(mgr.allocate(48), MemoryTier::Gpu);
    assert_eq!(mgr.usage(), (64, 16, 0));

    assert_eq!(mgr.allocate(64), MemoryTier::Gpu);
    assert_eq!(mgr.usage(), (64, 64, 16));
}

#[test]
#[serial]
fn cpu_and_nvme_fallback() {
    std::env::set_var("AMDUDA_HAS_GPU", "0");
    std::env::set_var("AMDUDA_HAS_NVME", "1");
    let caps = DeviceCapabilities::detect();
    let mut mgr = MemoryManager::new_with_limits(caps, 0, 64, 512);

    assert_eq!(mgr.allocate(32), MemoryTier::Cpu);
    assert_eq!(mgr.usage(), (0, 32, 0));

    assert_eq!(mgr.allocate(64), MemoryTier::Cpu);
    assert_eq!(mgr.usage(), (0, 64, 32));

    assert_eq!(mgr.allocate(128), MemoryTier::Nvme);
    assert_eq!(mgr.usage(), (0, 0, 224));
}

#[test]
#[serial]
fn manual_migration() {
    std::env::set_var("AMDUDA_HAS_GPU", "1");
    std::env::set_var("AMDUDA_HAS_NVME", "1");
    let caps = DeviceCapabilities::detect();
    let mut mgr = MemoryManager::new_with_limits(caps, 64, 64, 256);

    mgr.allocate(32);
    mgr.migrate(MemoryTier::Gpu, MemoryTier::Cpu, 16);
    assert_eq!(mgr.usage(), (16, 16, 0));

    mgr.migrate(MemoryTier::Cpu, MemoryTier::Nvme, 8);
    assert_eq!(mgr.usage(), (16, 8, 8));
}

#[test]
#[serial]
fn cold_data_evicted_first() {
    std::env::set_var("AMDUDA_HAS_GPU", "1");
    std::env::set_var("AMDUDA_HAS_NVME", "1");
    let caps = DeviceCapabilities::detect();
    let mut mgr = MemoryManager::new_with_limits(caps, 64, 64, 256);

    mgr.allocate(64);
    mgr.mark_cold(MemoryTier::Gpu, 32);
    assert_eq!(mgr.allocate(32), MemoryTier::Gpu);
    assert_eq!(mgr.usage(), (64, 32, 0));
}

#[test]
#[serial]
fn large_allocation_falls_back() {
    std::env::set_var("AMDUDA_HAS_GPU", "1");
    std::env::set_var("AMDUDA_HAS_NVME", "1");
    let caps = DeviceCapabilities::detect();
    let mut mgr = MemoryManager::new_with_limits(caps, 64, 64, 256);

    assert_eq!(mgr.allocate(200), MemoryTier::Nvme);
    assert_eq!(mgr.usage(), (0, 0, 200));
}
