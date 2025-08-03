//! Simulated multi-tier memory manager.
//!
//! The manager detects device capabilities and allocates memory across GPU,
//! CPU and NVMe tiers. When a tier is exhausted, data is migrated to the next
//! slower tier to act as a simple cache hierarchy.

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum MemoryTier {
    Gpu,
    Cpu,
    Nvme,
}

/// Runtime capabilities of the system.
#[derive(Debug, Clone, Copy)]
pub struct DeviceCapabilities {
    pub has_gpu: bool,
    pub has_nvme: bool,
}

impl DeviceCapabilities {
    /// Detects capabilities from environment variables.
    pub fn detect() -> Self {
        let has_gpu = std::env::var("AMDUDA_HAS_GPU")
            .map(|v| v == "1")
            .unwrap_or(false);
        let has_nvme = std::env::var("AMDUDA_HAS_NVME")
            .map(|v| v == "1")
            .unwrap_or(false);
        Self { has_gpu, has_nvme }
    }
}

/// Simple hierarchical memory manager.
#[derive(Debug)]
pub struct MemoryManager {
    caps: DeviceCapabilities,
    gpu_limit: usize,
    cpu_limit: usize,
    nvme_limit: usize,
    gpu_used: usize,
    cpu_used: usize,
    nvme_used: usize,
}

impl MemoryManager {
    /// Create a manager with default tier limits.
    pub fn new(caps: DeviceCapabilities) -> Self {
        Self::new_with_limits(caps, 1024, 8 * 1024, usize::MAX)
    }

    /// Create a manager with explicit limits for each tier.
    pub fn new_with_limits(
        caps: DeviceCapabilities,
        gpu_limit: usize,
        cpu_limit: usize,
        nvme_limit: usize,
    ) -> Self {
        Self {
            caps,
            gpu_limit,
            cpu_limit,
            nvme_limit,
            gpu_used: 0,
            cpu_used: 0,
            nvme_used: 0,
        }
    }

    /// Allocates memory using a caching hierarchy. New allocations prefer the
    /// fastest tier (GPU) and trigger migrations when space is required.
    pub fn allocate(&mut self, bytes: usize) -> MemoryTier {
        if self.caps.has_gpu {
            self.ensure_gpu_space(bytes);
            self.gpu_used += bytes;
            MemoryTier::Gpu
        } else {
            self.ensure_cpu_space(bytes);
            if self.cpu_used + bytes <= self.cpu_limit {
                self.cpu_used += bytes;
                MemoryTier::Cpu
            } else {
                self.nvme_used += bytes;
                MemoryTier::Nvme
            }
        }
    }

    /// Manually migrate data between tiers.
    pub fn migrate(&mut self, from: MemoryTier, to: MemoryTier, bytes: usize) {
        match (from, to) {
            (MemoryTier::Gpu, MemoryTier::Cpu) => {
                self.ensure_cpu_space(bytes);
                let moved = bytes.min(self.gpu_used);
                self.gpu_used -= moved;
                self.cpu_used += moved;
            }
            (MemoryTier::Cpu, MemoryTier::Gpu) => {
                if self.caps.has_gpu {
                    self.ensure_gpu_space(bytes);
                    let moved = bytes.min(self.cpu_used);
                    self.cpu_used -= moved;
                    self.gpu_used += moved;
                }
            }
            (MemoryTier::Cpu, MemoryTier::Nvme) => {
                if self.caps.has_nvme {
                    let moved = bytes.min(self.cpu_used);
                    self.cpu_used -= moved;
                    self.nvme_used += moved;
                }
            }
            (MemoryTier::Nvme, MemoryTier::Cpu) => {
                let moved = bytes.min(self.nvme_used);
                self.ensure_cpu_space(moved);
                self.nvme_used -= moved;
                self.cpu_used += moved;
            }
            _ => {}
        }
    }

    /// Current usage of each tier.
    pub fn usage(&self) -> (usize, usize, usize) {
        (self.gpu_used, self.cpu_used, self.nvme_used)
    }

    fn ensure_gpu_space(&mut self, bytes: usize) {
        if self.gpu_used + bytes <= self.gpu_limit {
            return;
        }
        let needed = self.gpu_used + bytes - self.gpu_limit;
        self.ensure_cpu_space(needed);
        let migrated = needed.min(self.gpu_used);
        self.gpu_used -= migrated;
        self.cpu_used += migrated;
    }

    fn ensure_cpu_space(&mut self, bytes: usize) {
        if self.cpu_used + bytes <= self.cpu_limit {
            return;
        }
        let needed = self.cpu_used + bytes - self.cpu_limit;
        if self.caps.has_nvme && self.nvme_used + needed <= self.nvme_limit {
            let migrated = needed.min(self.cpu_used);
            self.cpu_used -= migrated;
            self.nvme_used += migrated;
        } else {
            let dropped = needed.min(self.cpu_used);
            self.cpu_used -= dropped;
        }
    }
}

/// Convenience allocation using detected capabilities with default limits.
pub fn allocate(bytes: usize) -> MemoryTier {
    let caps = DeviceCapabilities::detect();
    let mut mgr = MemoryManager::new(caps);
    mgr.allocate(bytes)
}
