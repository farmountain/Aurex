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
    /// Total GPU memory available in bytes.
    pub gpu_mem: usize,
    /// Total CPU memory available in bytes.
    pub cpu_mem: usize,
    /// Total NVMe space available in bytes.
    pub nvme_mem: usize,
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
        let gpu_mem = std::env::var("AMDUDA_GPU_MEM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1024);
        let cpu_mem = std::env::var("AMDUDA_CPU_MEM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8 * 1024);
        let nvme_mem = std::env::var("AMDUDA_NVME_MEM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        Self {
            has_gpu,
            has_nvme,
            gpu_mem,
            cpu_mem,
            nvme_mem,
        }
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
    gpu_cold: usize,
    cpu_cold: usize,
    nvme_cold: usize,
}

impl MemoryManager {
    /// Create a manager with default tier limits.
    pub fn new(caps: DeviceCapabilities) -> Self {
        Self::new_with_limits(caps, caps.gpu_mem, caps.cpu_mem, caps.nvme_mem)
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
            gpu_cold: 0,
            cpu_cold: 0,
            nvme_cold: 0,
        }
    }

    /// Allocates memory using a caching hierarchy. New allocations prefer the
    /// fastest tier (GPU) and trigger migrations when space is required.
    pub fn allocate(&mut self, bytes: usize) -> MemoryTier {
        if self.caps.has_gpu && bytes <= self.gpu_limit {
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
                let cold = moved.min(self.gpu_cold);
                self.gpu_cold -= cold;
                self.cpu_cold += cold;
            }
            (MemoryTier::Cpu, MemoryTier::Gpu) => {
                if self.caps.has_gpu {
                    self.ensure_gpu_space(bytes);
                    let moved = bytes.min(self.cpu_used);
                    self.cpu_used -= moved;
                    self.gpu_used += moved;
                    let cold = moved.min(self.cpu_cold);
                    self.cpu_cold -= cold;
                    self.gpu_cold += cold;
                }
            }
            (MemoryTier::Cpu, MemoryTier::Nvme) => {
                if self.caps.has_nvme {
                    let moved = bytes.min(self.cpu_used);
                    self.cpu_used -= moved;
                    self.nvme_used += moved;
                    let cold = moved.min(self.cpu_cold);
                    self.cpu_cold -= cold;
                    self.nvme_cold += cold;
                }
            }
            (MemoryTier::Nvme, MemoryTier::Cpu) => {
                let moved = bytes.min(self.nvme_used);
                self.ensure_cpu_space(moved);
                self.nvme_used -= moved;
                self.cpu_used += moved;
                 let cold = moved.min(self.nvme_cold);
                 self.nvme_cold -= cold;
                 self.cpu_cold += cold;
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
        // migrate cold data first
        if self.gpu_cold > 0 {
            let cold_migrate = needed.min(self.gpu_cold);
            if cold_migrate > 0 {
                self.ensure_cpu_space(cold_migrate);
                self.gpu_cold -= cold_migrate;
                self.gpu_used -= cold_migrate;
                self.cpu_used += cold_migrate;
                self.cpu_cold += cold_migrate;
            }
        }
        if self.gpu_used + bytes <= self.gpu_limit {
            return;
        }
        let remaining = self.gpu_used + bytes - self.gpu_limit;
        self.ensure_cpu_space(remaining);
        let migrated = remaining.min(self.gpu_used);
        self.gpu_used -= migrated;
        self.cpu_used += migrated;
    }

    fn ensure_cpu_space(&mut self, bytes: usize) {
        if self.cpu_used + bytes <= self.cpu_limit {
            return;
        }
        let needed = self.cpu_used + bytes - self.cpu_limit;
        if self.caps.has_nvme && self.nvme_used + needed <= self.nvme_limit {
            // migrate cold bytes first
            if self.cpu_cold > 0 {
                let cold_migrate = needed.min(self.cpu_cold);
                self.cpu_cold -= cold_migrate;
                self.cpu_used -= cold_migrate;
                self.nvme_used += cold_migrate;
                self.nvme_cold += cold_migrate;
            }
            if self.cpu_used + bytes > self.cpu_limit {
                let remaining = self.cpu_used + bytes - self.cpu_limit;
                let migrated = remaining.min(self.cpu_used);
                self.cpu_used -= migrated;
                self.nvme_used += migrated;
            }
        } else {
            // drop cold bytes first
            if self.cpu_cold > 0 {
                let dropped = needed.min(self.cpu_cold);
                self.cpu_cold -= dropped;
                self.cpu_used -= dropped;
            }
            if self.cpu_used + bytes > self.cpu_limit {
                let remaining = self.cpu_used + bytes - self.cpu_limit;
                let dropped = remaining.min(self.cpu_used);
                self.cpu_used -= dropped;
            }
        }
    }
}

impl MemoryManager {
    /// Mark bytes in a tier as cold, making them candidates for eviction.
    pub fn mark_cold(&mut self, tier: MemoryTier, bytes: usize) {
        match tier {
            MemoryTier::Gpu => {
                let add = bytes.min(self.gpu_used - self.gpu_cold);
                self.gpu_cold += add;
            }
            MemoryTier::Cpu => {
                let add = bytes.min(self.cpu_used - self.cpu_cold);
                self.cpu_cold += add;
            }
            MemoryTier::Nvme => {
                let add = bytes.min(self.nvme_used - self.nvme_cold);
                self.nvme_cold += add;
            }
        }
    }

    /// Mark bytes in a tier as recently used (hot).
    pub fn mark_hot(&mut self, tier: MemoryTier, bytes: usize) {
        match tier {
            MemoryTier::Gpu => self.gpu_cold = self.gpu_cold.saturating_sub(bytes),
            MemoryTier::Cpu => self.cpu_cold = self.cpu_cold.saturating_sub(bytes),
            MemoryTier::Nvme => self.nvme_cold = self.nvme_cold.saturating_sub(bytes),
        }
    }
}

/// Convenience allocation using detected capabilities with default limits.
pub fn allocate(bytes: usize) -> MemoryTier {
    let caps = DeviceCapabilities::detect();
    let mut mgr = MemoryManager::new(caps);
    mgr.allocate(bytes)
}
