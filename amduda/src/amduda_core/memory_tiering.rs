//! Simulated multi-tier memory manager.

#[derive(Debug, PartialEq)]
pub enum MemoryTier {
    Gpu,
    Cpu,
    Nvme,
}

pub fn allocate(_bytes: usize) -> MemoryTier {
    // Placeholder strategy: always allocate on CPU for now.
    MemoryTier::Cpu
}
