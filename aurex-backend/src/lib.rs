//! Backend dispatch layer routing operations to device implementations.

pub mod dispatch;
pub mod vulkan_backend;

pub use dispatch::{Backend, Dispatcher, Workload};
pub use vulkan_backend::VulkanBackend;
